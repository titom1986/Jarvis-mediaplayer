import sys
import json
import requests
import time

from config import OLLAMA_URL, MODEL
from tools import radarr, sonarr, plex, catalog_sets


TOOLS = [
    *catalog_sets.TOOLS,
    plex.TOOL,
    radarr.TOOL,
    radarr.QUEUE_TOOL,
    radarr.REQUEST_TOOL,
    radarr.REQUEST_MOVIES_TOOL,
    sonarr.TOOL,
    sonarr.QUEUE_TOOL,
    sonarr.REQUEST_TOOL,
]


def execute_tool(name, args):
    if name == "catalog_title":
        return catalog_sets.title(args["query"], args["media_type"])
    if name == "catalog_person":
        return catalog_sets.person(args["name"], args["media_type"])
    if name == "catalog_genre":
        return catalog_sets.genre(args["name"], args["media_type"], args.get("source"))
    if name == "catalog_keyword_vocabulary":
        return catalog_sets.keyword_vocabulary(args["query"], args.get("limit", 12))
    if name == "catalog_keyword":
        return catalog_sets.keyword(args["name"], args["media_type"], args.get("source"), aliases=args.get("aliases"))
    if name == "catalog_years":
        return catalog_sets.years(args["year_from"], args["year_to"], args["media_type"], args.get("source"))
    if name == "catalog_combine":
        return catalog_sets.combine(args["operation"], args["sets"])
    if name == "catalog_subtract":
        return catalog_sets.subtract(args["source"], args["remove"])
    if name == "catalog_results":
        return catalog_sets.results(args["set"], args.get("limit", 10))

    if name == "plex_status":
        return plex.status(args["title"])
    if name == "radarr_status":
        return radarr.status(args["title"])
    if name == "radarr_queue_status":
        return radarr.queue_status(args["title"])
    if name == "radarr_request_movie":
        return radarr.request_movie(args["tmdb_id"], french=args.get("french", False))
    if name == "sonarr_status":
        return sonarr.status(args["title"])
    if name == "sonarr_queue_status":
        return sonarr.queue_status(args["title"])
    if name == "sonarr_request_series":
        return sonarr.request_series(
            args["tvdb_id"], scope=args.get("scope", "series"),
            season=args.get("season"), episode=args.get("episode"),
            french=args.get("french", False), force=args.get("force", False)
        )
    return {"error": f"Outil inconnu : {name}"}


def _ollama_perf(payload, wall_s):
    return {
        "wall_s": round(wall_s, 3),
        "load_ms": round((payload.get("load_duration") or 0) / 1_000_000, 1),
        "prompt_eval_ms": round((payload.get("prompt_eval_duration") or 0) / 1_000_000, 1),
        "eval_ms": round((payload.get("eval_duration") or 0) / 1_000_000, 1),
        "prompt_tokens": payload.get("prompt_eval_count"),
        "output_tokens": payload.get("eval_count"),
    }


PENDING_CONFIRMATION = None


def _is_confirmation(text):
    return (text or "").strip().casefold() in {
        "oui", "yes", "ok", "okay", "confirme", "je confirme", "vas-y", "vas y", "go"
    }


def _execute_pending_confirmation():
    global PENDING_CONFIRMATION
    plan = PENDING_CONFIRMATION
    PENDING_CONFIRMATION = None
    if not plan:
        return None
    results = []
    for movie in plan["movies"]:
        result = radarr.request_movie(movie["tmdb_id"], french=plan["french"])
        results.append({"title": movie["title"], **result})
        if result.get("error"):
            break
    return results


CATALOG_CONSTRAINT_TOOLS = {
    "catalog_person", "catalog_genre", "catalog_keyword", "catalog_years"
}


def _compose_catalog_batch(entries):
    """Compose one LLM constraint batch; Python owns execution order."""
    include_groups = {}
    exclude_groups = {}

    for index, (name, args) in enumerate(entries):
        is_exclude = args.get("exclude", False)
        if is_exclude and "group" not in args:
            group = f"exclude-{index}"
        else:
            group = int(args.get("group", 0))
        target = exclude_groups if is_exclude else include_groups
        target.setdefault(group, []).append((name, args))

    print("[PLAN] batch", json.dumps({
        "include_groups": {str(k): [name for name, _ in v] for k, v in include_groups.items()},
        "exclude_groups": {str(k): [name for name, _ in v] for k, v in exclude_groups.items()},
    }, ensure_ascii=False, sort_keys=True))

    if not include_groups:
        return {"error": "catalogue batch has no include constraint"}

    def execute_groups(groups, source=None):
        handles = []
        for group_name, members in groups.items():
            print("[PLAN] group_start", json.dumps({"group": str(group_name), "source": source, "constraints": [{"tool": n, "args": a} for n, a in members]}, ensure_ascii=False, sort_keys=True))
            result = catalog_sets.execute_constraint_group(members, source=source)
            print("[PLAN] group_done", json.dumps({"group": str(group_name), "source": source, "result": result}, ensure_ascii=False, sort_keys=True))
            if result.get("error"):
                return result
            handles.append(result["set"])
        if len(handles) == 1:
            value = catalog_sets._get(handles[0])
            return {"set": handles[0], "count": len(value["ids"])}
        return catalog_sets.combine("union", handles)

    included = execute_groups(include_groups)
    print("[PLAN] included", json.dumps(included, ensure_ascii=False, sort_keys=True))
    if included.get("error") or not exclude_groups:
        return included
    # Exclusions can only remove included candidates, so refine each exclusion
    # group from the included subset instead of broad-scanning the catalogue.
    excluded = execute_groups(exclude_groups, source=included["set"])
    print("[PLAN] excluded_union", json.dumps(excluded, ensure_ascii=False, sort_keys=True))
    if excluded.get("error"):
        return excluded
    final = catalog_sets.subtract(included["set"], excluded["set"])
    print("[PLAN] final_after_exclusions", json.dumps(final, ensure_ascii=False, sort_keys=True))
    return final

def _render_catalogue_results(grounded):
    results = grounded.get("results", [])
    if not results:
        return "Aucun résultat."
    blocks = []
    for item in results:
        title = item.get("title") or "Titre inconnu"
        date = item.get("releaseDate") or ""
        year = date[:4] if date else ""
        rating = item.get("rating")
        heading = f"{title} ({year})" if year else title
        if rating is not None:
            heading += f" — {rating}/10"
        overview = (item.get("overview") or "").strip()
        blocks.append(heading + (f"\n{overview}" if overview else ""))
    return "\n\n".join(blocks)


def _download_state(item):
    """Deterministically interpret Arr queue fields without hiding import state."""
    status = (item.get("status") or "").casefold()
    tracked_status = (item.get("trackedDownloadStatus") or "").casefold()
    tracked_state = (item.get("trackedDownloadState") or "").casefold()

    if tracked_status and tracked_status not in {"ok", "none"}:
        return f"{item.get('trackedDownloadStatus')}"
    if status == "completed":
        if tracked_state:
            return f"téléchargement terminé — traitement Sonarr/Radarr : {item.get('trackedDownloadState')}"
        return "téléchargement terminé — encore présent dans la queue"
    if status:
        return item.get("status")
    if tracked_state:
        return item.get("trackedDownloadState")
    return "en cours"


def _progress(item):
    size, left = item.get("size"), item.get("sizeleft")
    if isinstance(size, (int, float)) and size > 0 and isinstance(left, (int, float)):
        return max(0, min(100, round((size - left) * 100 / size)))
    return None


def _render_terminal_tool(name, result):
    if result.get("error"):
        return result["error"]
    if name == "radarr_request_movie":
        if result.get("alreadyExists"):
            return f'{result.get("title", "Film")} est déjà présent dans Radarr.'
        if result.get("added"):
            quality = result.get("qualityProfile")
            suffix = f" — {quality}" if quality else ""
            return f'{result.get("title", "Film")} mis en téléchargement{suffix}.'
    if name == "radarr_queue_status":
        title = result.get("title", "Film")
        if not result.get("found"):
            return f"{title} est introuvable dans Radarr."
        if not result.get("inQueue"):
            return f"{title} — aucun téléchargement en cours."
        state = _download_state(result)
        progress = _progress(result)
        parts = [f"{title} — {state}"]
        if progress is not None and (result.get("status") or "").casefold() != "completed":
            parts.append(f"{progress} %")
        if result.get("timeleft"):
            parts.append(f'restant : {result["timeleft"]}')
        return " — ".join(parts) + "."
    if name == "radarr_status":
        title = result.get("title", "Film")
        if not result.get("found"):
            return f"{title} est introuvable dans Radarr."
        if result.get("inQueue"):
            return f"{title} — téléchargement en cours."
        if result.get("hasFile"):
            return f"{title} — disponible."
        return f"{title} — présent dans Radarr, sans fichier."
    if name == "sonarr_queue_status":
        title = result.get("title", "Série")
        if not result.get("found"):
            return f"{title} est introuvable dans Sonarr."
        if not result.get("inQueue"):
            return f"{title} — aucun téléchargement en cours."
        items = result.get("items", [])
        states = {}
        for item in items:
            state = _download_state(item)
            states[state] = states.get(state, 0) + 1
        parts = [f"{count} {state}" for state, count in states.items()]
        active = [
            progress for item in items
            if (item.get("status") or "").casefold() != "completed"
            for progress in [_progress(item)]
            if progress is not None
        ]
        if active:
            parts.append(f"progression moyenne : {round(sum(active) / len(active))} %")
        return f"{title} — " + " ; ".join(parts) + "."
    if name == "sonarr_status":
        title = result.get("title", "Série")
        if not result.get("found"):
            return f"{title} est introuvable dans Sonarr."
        return f'{title} — {result.get("episodeFileCount", 0)}/{result.get("episodeCount", 0)} épisodes disponibles.'
    if name == "plex_status":
        title = result.get("title", "Média")
        if not result.get("found"):
            return f"{title} est introuvable dans Plex."
        return f"{title} — disponible dans Plex."
    return None


def _model_config():
    # Preserve every model's native Ollama tool template. Semantic operating
    # instructions live in the common tool schemas, not in model-specific prompts.
    payload = {"think": False} if MODEL.casefold().startswith("qwen3") else {}
    return None, payload


def run_agent(question):
    global PENDING_CONFIRMATION
    if PENDING_CONFIRMATION is not None:
        if _is_confirmation(question):
            results = _execute_pending_confirmation()
            lines = []
            for item in results or []:
                if item.get("error"):
                    lines.append(f'{item.get("title", "Film")} — {item["error"]}')
                elif item.get("alreadyExists"):
                    lines.append(f'{item.get("title", "Film")} — déjà présent dans Radarr.')
                elif item.get("added"):
                    lines.append(f'{item.get("title", "Film")} — mis en téléchargement.')
            final = "\n".join(lines) or "Aucune action exécutée."
            print("\n═══ RÉPONSE ═══\n")
            print(final)
            return final
        # Any non-confirmation cancels the stale plan before handling the new request.
        PENDING_CONFIRMATION = None

    total_started = time.perf_counter()
    catalog_sets.reset()
    pending_catalog_batch = []
    grounded_keyword_labels = set()
    grounded_catalogue_results = None
    terminal_content = None

    system, model_payload = _model_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    print("\n--- Agent ---")
    final_content = ""
    llm_calls = []
    tool_calls = 0

    # Atomic tools need several cheap turns; the cap prevents a broken model
    # from looping forever without embedding any query-specific workflow here.
    for step in range(16):
        request_payload = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 220},
        }
        request_payload.update(model_payload)

        call_started = time.perf_counter()
        response = requests.post(OLLAMA_URL, json=request_payload, timeout=None)
        response.raise_for_status()
        payload = response.json()
        perf = _ollama_perf(payload, time.perf_counter() - call_started)
        llm_calls.append(perf)
        print(f"[PERF] agent_llm_{step + 1}", json.dumps(perf, ensure_ascii=False))

        message = payload["message"]
        calls = message.get("tool_calls", [])
        if not calls:
            if pending_catalog_batch:
                final_content = "Je n'ai pas pu finaliser la recherche catalogue : des contraintes déclarées n'ont pas été exécutées."
            else:
                final_content = message.get("content", "").strip()
            break

        messages.append(message)
        constraint_calls = [
            call for call in calls
            if call["function"]["name"] in CATALOG_CONSTRAINT_TOOLS
        ]
        other_calls = [
            call for call in calls
            if call["function"]["name"] not in CATALOG_CONSTRAINT_TOOLS
        ]
        pending_tool_messages = []

        if constraint_calls:
            # Constraint calls are declarations, never execution. This makes the
            # planner independent of whether the model emits one call per turn or
            # several calls in parallel.
            for call in constraint_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                print(f"> {name}({args})")
                tool_calls += 1

                if name == "catalog_keyword":
                    labels = [args.get("name"), *(args.get("aliases") or [])]
                    ungrounded = [
                        value for value in labels if value and
                        catalog_sets.media_search._norm(value) not in grounded_keyword_labels
                    ]
                    if ungrounded:
                        vocabulary = catalog_sets.keyword_vocabulary(args["name"])
                        real_labels = [item["name"] for item in vocabulary.get("keywords", [])]
                        grounded_keyword_labels.update(
                            catalog_sets.media_search._norm(value) for value in real_labels
                        )
                        result = {
                            "grounding_required": True,
                            "concept": args["name"],
                            "catalogue_labels": real_labels,
                            "error": vocabulary.get("error"),
                            "diagnostic": vocabulary.get("response"),
                            "required_action": (
                                "Choose only real labels matching the user's concept. "
                                "If insufficient, call catalog_keyword_vocabulary with another "
                                "English wording. Then redeclare catalog_keyword with grounded "
                                "name/aliases. Keep all other constraints; they are already pending."
                            ),
                        }
                        pending_tool_messages.append((name, result))
                        continue

                entry = (name, args)
                if entry not in pending_catalog_batch:
                    pending_catalog_batch.append(entry)
                pending_tool_messages.append((name, {
                    "accepted": True,
                    "pending_constraints": len(pending_catalog_batch),
                    "required_action": "Declare any remaining constraints, then call catalog_execute once.",
                }))
        if other_calls:
            for call in other_calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                print(f"> {name}({args})")
                started = time.perf_counter()

                if name == "catalog_title":
                    try:
                        result = execute_tool(name, args)
                    except Exception as exc:
                        result = {"error": str(exc)}
                    if not result.get("error"):
                        # A named title is already grounded catalogue data. It is
                        # deliberately kept separate from semantic discovery: words
                        # inside the title are not constraints.
                        grounded_catalogue_results = {
                            "count": len(result.get("results", [])),
                            "results": result.get("results", []),
                        }
                        result["grounded_results"] = grounded_catalogue_results
                        result["required_action"] = (
                            "The named target is grounded. Continue with the user's requested "
                            "status/action using only an ID from grounded_results. Do not reinterpret "
                            "words inside the title as catalogue constraints."
                        )
                elif name == "catalog_execute":
                    if not pending_catalog_batch:
                        result = {"error": "no pending catalogue constraints"}
                    else:
                        try:
                            result = _compose_catalog_batch(pending_catalog_batch)
                        except Exception as exc:
                            result = {"error": str(exc)}
                        if not result.get("error") and result.get("set"):
                            grounded_catalogue_results = catalog_sets.results(result["set"], limit=10)
                            result["grounded_results"] = grounded_catalogue_results
                            result["results_loaded"] = True
                            result["response_contract"] = (
                                "These grounded_results are the only catalogue media you may name "
                                "as search results. Titles, dates, ratings, and descriptions must "
                                "come only from grounded_results."
                            )
                            if args.get("purpose") != "action":
                                terminal_content = _render_catalogue_results(grounded_catalogue_results)
                        pending_catalog_batch = []
                elif pending_catalog_batch and name in {
                    "plex_status", "radarr_status", "radarr_queue_status",
                    "radarr_request_movie", "radarr_request_movies", "sonarr_status", "sonarr_queue_status", "sonarr_request_series"
                }:
                    result = {
                        "error": "catalogue constraints are still pending",
                        "required_action": "Call catalog_execute before status or media actions.",
                    }
                else:
                    if name == "radarr_request_movies":
                        movies = args.get("movies") or []
                        grounded_movies = {
                            item.get("id"): item for item in (grounded_catalogue_results or {}).get("results", [])
                            if item.get("mediaType") == "movie"
                        }
                        ids = [item.get("tmdb_id") for item in movies]
                        if len(movies) < 2 or len(set(ids)) != len(ids) or any(mid not in grounded_movies for mid in ids):
                            result = {"error": "Plan multi-films invalide ou non entièrement groundé."}
                        else:
                            # Titles are replaced with grounded catalogue titles; the model cannot
                            # smuggle arbitrary display/action targets into the confirmation.
                            plan_movies = [
                                {"tmdb_id": mid, "title": grounded_movies[mid].get("title") or str(mid)}
                                for mid in ids
                            ]
                            if args.get("exclude_existing_plex", False):
                                plan_movies = [
                                    movie for movie in plan_movies
                                    if not plex.status(movie["title"]).get("found", False)
                                ]
                            if not plan_movies:
                                result = {"confirmationRequired": False, "count": 0}
                                terminal_content = "Tous les films correspondants sont déjà disponibles dans Plex."
                                pending_tool_messages.append((name, result))
                                print("<", json.dumps(result, ensure_ascii=False))
                                tool_calls += 1
                                continue
                            if len(plan_movies) == 1:
                                try:
                                    result = radarr.request_movie(
                                        plan_movies[0]["tmdb_id"], french=bool(args.get("french", False))
                                    )
                                except Exception as exc:
                                    result = {"error": str(exc)}
                                terminal_content = _render_terminal_tool("radarr_request_movie", result)
                                pending_tool_messages.append((name, result))
                                print("<", json.dumps(result, ensure_ascii=False))
                                tool_calls += 1
                                continue
                            PENDING_CONFIRMATION = {
                                "movies": plan_movies,
                                "french": bool(args.get("french", False)),
                            }
                            language = " en français" if PENDING_CONFIRMATION["french"] else ""
                            result = {"confirmationRequired": True, "count": len(plan_movies)}
                            terminal_content = (
                                f'Je vais télécharger{language} : ' +
                                ", ".join(item["title"] for item in plan_movies) +
                                ". Tu confirmes ?"
                            )
                    elif name == "sonarr_request_series":
                        tv_items = [
                            item for item in (grounded_catalogue_results or {}).get("results", [])
                            if item.get("mediaType") == "tv"
                        ]
                        requested_tmdb = args.get("tmdb_id")
                        grounded = next((item for item in tv_items if item.get("id") == requested_tmdb), None)
                        if grounded is None:
                            result = {
                                "error": "Série non vérifiée dans les résultats catalogue de cette requête.",
                                "required_action": "Ground the requested TV series before requesting it."
                            }
                        else:
                            try:
                                details = catalog_sets.seerr.media_details(requested_tmdb, "tv")
                                external = details.get("externalIds") or {}
                                tvdb_id = external.get("tvdbId")
                                if not tvdb_id:
                                    result = {"error": "TVDB ID vérifié absent des détails Seerr pour cette série."}
                                else:
                                    write_args = dict(args)
                                    write_args.pop("tmdb_id", None)
                                    write_args["tvdb_id"] = int(tvdb_id)
                                    result = execute_tool(name, write_args)
                            except Exception as exc:
                                result = {"error": str(exc)}
                    elif name == "radarr_request_movie":
                        allowed_ids = {
                            item.get("id") for item in (grounded_catalogue_results or {}).get("results", [])
                            if item.get("mediaType") == "movie"
                        }
                        if args.get("tmdb_id") not in allowed_ids:
                            result = {
                                "error": "Film non vérifié dans les résultats catalogue de cette requête.",
                                "required_action": "Ground the requested movie with catalogue tools before requesting it."
                            }
                        else:
                            try:
                                result = execute_tool(name, args)
                            except Exception as exc:
                                result = {"error": str(exc)}
                    else:
                        try:
                            result = execute_tool(name, args)
                        except Exception as exc:
                            result = {"error": str(exc)}
                    if name in {
                        "plex_status", "radarr_status", "radarr_queue_status",
                        "radarr_request_movie", "radarr_request_movies", "sonarr_status", "sonarr_queue_status", "sonarr_request_series"
                    } and terminal_content is None:
                        terminal_content = _render_terminal_tool(name, result)
                    if name == "catalog_keyword_vocabulary" and not result.get("error"):
                        grounded_keyword_labels.update(
                            catalog_sets.media_search._norm(item["name"])
                            for item in result.get("keywords", []) if item.get("name")
                        )

                print("<", json.dumps(result, ensure_ascii=False))
                print(
                    f"[PERF] tool {name}",
                    json.dumps({"wall_s": round(time.perf_counter() - started, 3)}, ensure_ascii=False),
                )
                tool_calls += 1
                pending_tool_messages.append((name, result))

        if terminal_content is not None:
            final_content = terminal_content
            break

        for name, result in pending_tool_messages:
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })

    if not final_content:
        final_content = "Je n'ai pas pu produire une réponse finale vérifiée."

    print("[PERF] total", json.dumps({
        "wall_s": round(time.perf_counter() - total_started, 3),
        "ollama_calls": len(llm_calls),
        "tool_calls": tool_calls,
    }, ensure_ascii=False))
    print("\n═══ RÉPONSE ═══\n")
    print(final_content)


if __name__ == "__main__":
    print(f"Seed Agent - {MODEL}")
    print()

    if len(sys.argv) > 1:
        run_agent(" ".join(sys.argv[1:]).strip())
    else:
        while True:
            try:
                question = input("> ").strip()
                if question.lower() in {"exit", "quit", "/bye"}:
                    break
                if question:
                    run_agent(question)
            except (KeyboardInterrupt, EOFError):
                print()
                break
