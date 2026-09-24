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


def _model_config():
    name = MODEL.casefold()
    # Keep native default system messages for models whose Ollama templates use
    # them to teach the tool protocol. Qwen accepts a compact custom policy.
    if name.startswith(("granite3.3", "phi4-mini", "ministral-3")):
        return None, {}
    payload = {"think": False} if name.startswith("qwen3") else {}
    return (
        "Tu es JARVIS, l'agent d'un media center. Utilise les outils pour toute "
        "information dépendant du catalogue ou des services et n'invente pas leurs données. "
        "Les outils catalog_* créent des ensembles opaques : garde leurs handles courts, "
        "raffine un ensemble existant avec source quand tu ajoutes une contrainte, combine par intersection pour des contraintes simultanées indépendantes, union pour des "
        "alternatives, et catalog_subtract pour une exclusion. Termine une recherche avec "
        "catalog_results. N'ajoute/télécharge un média que sur demande explicite. "
        "Réponds dans la langue de l'utilisateur.",
        payload,
    )


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
        response = requests.post(OLLAMA_URL, json=request_payload, timeout=180)
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
        for call in calls:
            name = call["function"]["name"]
            args = call["function"].get("arguments") or {}
            print(f"> {name}({args})")
            started = time.perf_counter()
            try:
                result = execute_tool(name, args)
            except Exception as exc:
                result = {"error": str(exc)}
            print("<", json.dumps(result, ensure_ascii=False))
            print(
                f"[PERF] tool {name}",
                json.dumps({"wall_s": round(time.perf_counter() - started, 3)}, ensure_ascii=False),
            )
            tool_calls += 1
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
