import json
import requests
import time

from config import OLLAMA_URL, MODEL
from planner import extract_intent, compile_media_search
from tools import radarr, sonarr, plex, media_search


TOOLS = [
    radarr.TOOL,
    radarr.QUEUE_TOOL,
    radarr.REQUEST_TOOL,
    sonarr.TOOL,
    plex.TOOL,
    media_search.MEDIA_SEARCH_TOOL,
]


def execute_tool(name, args):
    if name == "radarr_status":
        return radarr.status(args["title"])

    if name == "radarr_queue_status":
        return radarr.queue_status(args["title"])

    if name == "radarr_request_movie":
        return radarr.request_movie(args["tmdb_id"], french=args.get("french", False))

    if name == "sonarr_status":
        return sonarr.status(args["title"])

    if name == "plex_status":
        return plex.status(args["title"])

    if name == "seerr_media_search":
        return media_search.media_search(
            media_type=args["media_type"],
            include=args["include"],
            exclude=args.get("exclude")
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


def run_agent(question):
    total_started = time.perf_counter()

    intent = extract_intent(question)
    intent_perf = intent.pop("_perf", {})
    print("\n[PERF] intent", json.dumps(intent_perf, ensure_ascii=False))
    compiled_search = compile_media_search(intent)

    preverified = []
    if compiled_search is not None:
        search_started = time.perf_counter()
        result = media_search.media_search(**compiled_search)
        search_wall = time.perf_counter() - search_started
        print("\n--- Recherche structurée ---")
        print("> seerr_media_search(" + str(compiled_search) + ")")
        print("<", json.dumps(result, ensure_ascii=False))
        print("[PERF] media_search", json.dumps({
            "wall_s": round(search_wall, 3),
            **result.get("_perf", {})
        }, ensure_ascii=False))
        preverified.append({
            "tool": "seerr_media_search",
            "arguments": compiled_search,
            "result": result,
        })

    messages = [
        {
            "role": "system",
            "content": (
                "Tu administres un serveur multimédia et réponds dans la langue de l'utilisateur. "
                "La recherche catalogue est déjà vérifiée et ordonnée. Ne rappelle jamais seerr_media_search. "
                "IMPORTANT pour avoid_watched : Plex sert uniquement à exclure un candidat qui est À LA FOIS "
                "found=true ET watched=true. Un candidat found=false dans Plex RESTE VALIDE. "
                "Un candidat found=true et watched=false RESTE VALIDE. "
                "Teste les candidats dans l'ordre jusqu'au premier valide, puis recommande-le. "
                "Si download=false, n'appelle jamais Radarr pour ajouter/télécharger. "
                "Si download=true, vérifie Plex puis Radarr avant tout ajout. "
                "french_download ne doit être transmis à Radarr que s'il est vrai. "
                "Quand tu disposes d'un candidat valide, réponds directement : aucun autre outil n'est nécessaire."
            )
        },
        {
            "role": "user",
            "content": (
                "Demande : " + question + "\n"
                "Intent structuré : " + json.dumps(intent, ensure_ascii=False) + "\n"
                "Recherche catalogue déjà vérifiée : " + json.dumps(preverified, ensure_ascii=False)
            )
        }
    ]

    results = list(preverified)
    print("\n--- Plan agent ---")
    final_content = ""
    llm_calls = []

    for step in range(8):
        call_started = time.perf_counter()
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": [radarr.TOOL, radarr.QUEUE_TOOL, radarr.REQUEST_TOOL, sonarr.TOOL, plex.TOOL],
                "stream": False,
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_predict": 160}
            },
            timeout=180
        )
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
            args = call["function"]["arguments"]
            print(f"> {name}({args})")

            tool_started = time.perf_counter()
            result = execute_tool(name, args)
            tool_wall = time.perf_counter() - tool_started
            print("<", json.dumps(result, ensure_ascii=False))
            print(f"[PERF] tool {name}", json.dumps({"wall_s": round(tool_wall, 3)}, ensure_ascii=False))

            results.append({"tool": name, "arguments": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    if not final_content:
        final_content = "Je n'ai pas pu produire une réponse finale vérifiée."

    print("[PERF] total", json.dumps({
        "wall_s": round(time.perf_counter() - total_started, 3),
        "ollama_calls": 1 + len(llm_calls)
    }, ensure_ascii=False))

    print("\n═══ RÉPONSE ═══\n")
    print(final_content)

if __name__ == "__main__":
    print("Seed Agent - Qwen3 4B local")
    print("Lecture seule")
    print()

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
