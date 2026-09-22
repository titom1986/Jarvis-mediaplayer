import os
import requests

PLEX_URL = os.environ.get("PLEX_URL", "http://127.0.0.1:32400")
PLEX_TOKEN = os.environ.get("PLEX_TOKEN")


def status(title):
    if not PLEX_TOKEN:
        return {"error": "PLEX_TOKEN non configuré"}

    try:
        response = requests.get(
            f"{PLEX_URL}/hubs/search",
            params={
                "query": title,
                "X-Plex-Token": PLEX_TOKEN,
            },
            headers={"Accept": "application/json"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = []

        for hub in data.get("MediaContainer", {}).get("Hub", []):
            for item in hub.get("Metadata", []):
                item_title = item.get("title", "")

                if title.casefold() != item_title.casefold():
                    continue

                results.append({
                    "title": item_title,
                    "type": item.get("type"),
                    "year": item.get("year"),
                    "ratingKey": item.get("ratingKey"),
                    "library": item.get("librarySectionTitle"),
                    "addedAt": item.get("addedAt"),
                })

        if not results:
            return {"found": False, "title": title}

        return {
            "found": True,
            "title": title,
            "results": results[:10],
        }

    except Exception as e:
        return {"error": str(e), "title": title}

TOOL = {
    "type": "function",
    "function": {
        "name": "plex_status",
        "description": "Recherche un film ou une série dans Plex et retourne les éléments correspondants.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titre du film ou de la série à rechercher"
                }
            },
            "required": ["title"]
        }
    }
}
