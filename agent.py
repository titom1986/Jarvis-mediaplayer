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
    if name == "catalog_keyword":
        return catalog_sets.keyword(args["name"], args["media_type"], args.get("source"))
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
    """Compose one LLM batch deterministically from group/exclude annotations."""
    include_groups = {}
    exclude_groups = {}

    for index, (args, result) in enumerate(entries):
        if result.get("error"):
            return {"error": result["error"]}
        if result.get("found") is False:
            return {"error": f"unresolved catalogue constraint: {result}"}
        handle = result.get("set")
        if not handle:
            return {"error": f"catalogue constraint produced no set: {result}"}
        is_exclude = args.get("exclude", False)
        # Includes default to one AND group. Independent exclusions default to
        # separate groups, therefore their union is removed (A - (E1 OR E2)).
        # An explicit shared group can still express a compound exclusion.
        if is_exclude and "group" not in args:
            group = f"exclude-{index}"
        else:
            group = int(args.get("group", 0))
        target = exclude_groups if is_exclude else include_groups
        target.setdefault(group, []).append(handle)

    if not include_groups:
        return {"error": "catalogue batch has no include constraint"}

    def compose_groups(groups):
        handles = []
        for members in groups.values():
            if len(members) == 1:
                handles.append(members[0])
            else:
                combined = catalog_sets.combine("intersection", members)
                if combined.get("error"):
                    return combined
                handles.append(combined["set"])
        if len(handles) == 1:
            value = catalog_sets._get(handles[0])
            return {"set": handles[0], "count": len(value["ids"])}
        return catalog_sets.combine("union", handles)

    included = compose_groups(include_groups)
    if included.get("error"):
        return included

    if not exclude_groups:
        return included

    excluded = compose_groups(exclude_groups)
    if excluded.get("error"):
        return excluded
    return catalog_sets.subtract(included["set"], excluded["set"])


def _model_config():
    # Preserve every model's native Ollama tool template. Semantic operating
    # instructions live in the common tool schemas, not in model-specific prompts.
    payload = {"think": False} if MODEL.casefold().startswith("qwen3") else {}
    return None, payload


def run_agent(question):
    total_started = time.perf_counter()
    catalog_sets.reset()

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
            final_content = message.get("content", "").strip()
            break

        messages.append(message)
        batch_is_constraints = (
            len(calls) > 1
            and all(call["function"]["name"] in CATALOG_CONSTRAINT_TOOLS for call in calls)
        )
        batch_entries = []
        pending_tool_messages = []

        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            print(f"> {name}({args})")
            started = time.perf_counter()
            try:
                # A batch contains independent declarations. Ignore stale/source
                # handles so all constraints are materialized on equal footing.
                exec_args = dict(args)
                if batch_is_constraints:
                    exec_args.pop("source", None)
                result = execute_tool(name, exec_args)
            except Exception as exc:
                result = {"error": str(exc)}
            print("<", json.dumps(result, ensure_ascii=False))
            print(
                f"[PERF] tool {name}",
                json.dumps({"wall_s": round(time.perf_counter() - started, 3)}, ensure_ascii=False),
            )
            tool_calls += 1
            if batch_is_constraints:
                batch_entries.append((args, result))
            pending_tool_messages.append((name, result))

        if batch_is_constraints:
            composed = _compose_catalog_batch(batch_entries)
            print("< composed", json.dumps(composed, ensure_ascii=False))
            # Attach the deterministic batch result to the final tool response.
            # The model only needs this final handle for catalog_results.
            pending_tool_messages[-1][1]["composed"] = composed

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
            except KeyboardInterrupt:
                print()
                break
