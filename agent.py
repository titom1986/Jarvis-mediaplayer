import sys
import json
import requests
import time

from config import OLLAMA_URL, MODEL
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

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un agent multimédia. Comprends la demande puis choisis toi-même les outils nécessaires. "
                "Ne transforme jamais une préférence en condition plus stricte que celle demandée. "
                "Pour seerr_media_search : toutes les contraintes cumulatives appartiennent à UN même groupe include ; "
                "plusieurs groupes include signifient uniquement des alternatives OR explicitement demandées. "
                "Chaque information appartient à une seule catégorie : personne→people, genre de catalogue→genres, "
                "thème/concept→keywords, période→dates. Ne duplique jamais une valeur entre catégories. "
                "Une décennie est inclusive : années 90→1990..1999. "
                "exclude porte déjà la négation : mets le concept POSITIF dans exclude, par exemple "
                "« pas dystopique »→exclude keyword « dystopia », jamais « not dystopian ». "
                "Les résultats de seerr_media_search sont triés par note puis nombre de votes. "
                "Si l'utilisateur demande d'éviter les médias déjà vus dans Plex, found=false reste un candidat valide, "
                "found=true/watched=false reste valide, seul found=true/watched=true doit être évité. "
                "N'ajoute/télécharge un média que si l'utilisateur le demande explicitement. "
                "Pour un téléchargement français, french=true uniquement si le français est explicitement demandé. "
                "Après chaque résultat d'outil, décide librement si un autre outil est nécessaire ou si tu peux répondre. "
                "Réponds dans la langue de l'utilisateur."
            )
        },
        {"role": "user", "content": question}
    ]

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
                "tools": TOOLS,
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

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    if not final_content:
        final_content = "Je n'ai pas pu produire une réponse finale vérifiée."

    print("[PERF] total", json.dumps({
        "wall_s": round(time.perf_counter() - total_started, 3),
        "ollama_calls": len(llm_calls)
    }, ensure_ascii=False))

    print("\n═══ RÉPONSE ═══\n")
    print(final_content)

if __name__ == "__main__":
    print("Seed Agent - Qwen3 4B local")
    print("Lecture seule")
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
