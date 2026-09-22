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


def run_agent(question):
    # Conversation interne du planner.
    # Il peut appeler des outils, observer leurs résultats,
    # puis décider d'en appeler d'autres.
    intent = extract_intent(question)
    compiled_search = compile_media_search(intent)

    # Le LLM extrait la sémantique ; Python compile la logique booléenne.
    # Seerr/Plex/Radarr restent interrogés en temps réel : aucun cache métier.
    preverified = []
    if compiled_search is not None:
        result = media_search.media_search(**compiled_search)
        print("\n--- Recherche structurée ---")
        print("> seerr_media_search(" + str(compiled_search) + ")")
        print("<", json.dumps(result, ensure_ascii=False))
        preverified.append({
            "tool": "seerr_media_search",
            "arguments": compiled_search,
            "result": result,
        })

    messages = [
        {
            "role": "system",
            "content": (
                "Tu administres un serveur multimédia. Réponds dans la langue de l'utilisateur. "
                "L'intention de recherche et la recherche catalogue ont déjà été traitées de façon structurée. "
                "Ne rappelle jamais seerr_media_search. Utilise uniquement Plex/Radarr/Sonarr pour vérifier "
                "l'état frais du serveur. Si avoid_watched est vrai, vérifie dans Plex les candidats utiles "
                "dans l'ordre fourni. Si download est faux, n'ajoute jamais de média. Si download est vrai, "
                "vérifie Plex puis Radarr avant tout ajout. french_download ne doit être transmis à Radarr "
                "que s'il est vrai. Quand la demande est vérifiée, n'appelle plus d'outil."
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

    # Maximum de cycles pour empêcher une boucle infinie.
    for step in range(8):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": [radarr.TOOL, radarr.QUEUE_TOOL, radarr.REQUEST_TOOL, sonarr.TOOL, plex.TOOL],
                "stream": False,
                "keep_alive": "30m",
                "options": {
                    "num_predict": 120
                }
            },
            timeout=180
        )
        response.raise_for_status()

        message = response.json()["message"]
        calls = message.get("tool_calls", [])

        # Aucun nouvel outil : le planner considère la recherche terminée.
        if not calls:
            break

        # Le modèle doit revoir ses propres appels au tour suivant.
        messages.append(message)

        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            print(f"> {name}({args})")

            result = execute_tool(name, args)

            print("<", json.dumps(result, ensure_ascii=False))

            results.append({
                "tool": name,
                "arguments": args,
                "result": result
            })

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False)
            })

    # Aucun outil n'était nécessaire.
    if not results:
        synthesis_input = "Aucun résultat d'outil n'était nécessaire."
    else:
        synthesis_input = json.dumps(results, ensure_ascii=False)

    # Synthèse indépendante : aucun tool disponible ici.
    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "Réponds directement en français à la demande de l'utilisateur. "
                "Pour toute information concernant le serveur multimédia, utilise "
                "uniquement les résultats vérifiés fournis. "
                "N'invente aucune information. "
                "Ne mentionne pas le fonctionnement interne, les outils ou le raisonnement. "
                "Si les résultats ne permettent pas de conclure sur un point, dis-le clairement. "
                "Sois concis par défaut."
            )
        },
        {
            "role": "user",
            "content": (
                "Demande : " + question + "\n\n"
                "Résultats vérifiés :\n" + synthesis_input
            )
        }
    ]

    final_response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": synthesis_messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 120
            }
        },
        timeout=180
    )
    final_response.raise_for_status()

    content = final_response.json()["message"].get("content", "").strip()

    print("\n═══ RÉPONSE ═══\n")
    print(content or "Aucune réponse générée.")

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
