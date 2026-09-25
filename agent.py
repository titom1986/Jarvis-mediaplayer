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
    sonarr.TOOL,
]


def execute_tool(name, args):
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

def _model_config():
    # Preserve every model's native Ollama tool template. Semantic operating
    # instructions live in the common tool schemas, not in model-specific prompts.
    payload = {"think": False} if MODEL.casefold().startswith("qwen3") else {}
    return None, payload


def run_agent(question):
    total_started = time.perf_counter()
    catalog_sets.reset()
    pending_catalog_batch = []
    grounded_keyword_labels = set()
    grounded_catalogue_results = None

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
        batch_is_constraints = all(
            call["function"]["name"] in CATALOG_CONSTRAINT_TOOLS for call in calls
        )
        pending_tool_messages = []

        if batch_is_constraints:
            # Constraint calls are declarations, never execution. This makes the
            # planner independent of whether the model emits one call per turn or
            # several calls in parallel.
            for call in calls:
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
        else:
            for call in calls:
                name = call["function"]["name"]
                args = call["function"].get("arguments") or {}
                print(f"> {name}({args})")
                started = time.perf_counter()

                if name == "catalog_execute":
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
                        pending_catalog_batch = []
                else:
                    try:
                        result = execute_tool(name, args)
                    except Exception as exc:
                        result = {"error": str(exc)}
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
