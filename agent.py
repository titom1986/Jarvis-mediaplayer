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


def run_agent(question):
    # Conversation interne du planner.
    # Il peut appeler des outils, observer leurs résultats,
    # puis décider d'en appeler d'autres.
    messages = [
        {
            "role": "system",
            "content": (
                 "Tu administres un serveur multimédia. "
                "Réponds dans la langue de l'utilisateur. "
                "Analyse toute la demande avant de choisir un outil. "
                "Pour rechercher un film ou une série selon son contenu, son casting, "
                "son genre ou sa période, utilise seerr_media_search. "
                "Traduis toi-même le sens de la demande en critères adaptés aux métadonnées : "
                "personnes, genres, mots-clés et dates. "
                "Dans seerr_media_search, une demande cumulative forme UN SEUL groupe : personne + genre + thème + période "
                "restent ensemble dans ce groupe. Toutes ses contraintes sont en AND, même si elles appartiennent "
                "à des catégories différentes. Les groupes distincts sont exclusivement des alternatives OR explicites "
                "(par exemple « soit ... soit ... »). Une simple succession de critères ne crée jamais plusieurs groupes. "
                "Classe les personnes dans people, les genres de catalogue dans genres, les thèmes ou concepts "
                "dans keywords et les périodes dans dates. Dans exclude, exprime le critère lui-même sans mot de négation : "
                "« pas dystopique » devient par exemple le concept positif « dystopia », jamais « not dystopian ». "
                "Une décennie est inclusive : les années 90 vont de 1990 à 1999. Respecte exactement la logique booléenne exprimée. "
                "N'ajoute aucune contrainte qu'il n'a pas demandée. "
                "Utilise les autres outils uniquement pour vérifier l'état réel du serveur "
                "comme la disponibilité Plex, Radarr, Sonarr ou les téléchargements. "
                "Ne confonds jamais présence dans le catalogue avec disponibilité dans Plex. "
                "Quand plusieurs candidats sont retournés, utilise leurs notes et nombres de votes seulement si la demande "
                "demande une recommandation ou un classement. Si l'utilisateur demande d'éviter les éléments déjà vus, "
                "vérifie les candidats dans Plex, dans l'ordre utile, jusqu'à en trouver un admissible. "
                "N'utilise radarr_request_movie que si l'utilisateur a explicitement demandé de télécharger ou ajouter le film. "
                "Avant cette action, vérifie d'abord Plex puis Radarr afin d'éviter un ajout inutile ou en double. "
                "Après chaque résultat, décide si une autre vérification est réellement nécessaire. "
                "Quand la demande est entièrement vérifiée, n'appelle plus d'outil."
            )
        },
        {
            "role": "user",
            "content": question
        }
    ]

    results = []

    print("\n--- Plan agent ---")

    # Maximum de cycles pour empêcher une boucle infinie.
    for step in range(8):
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
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
