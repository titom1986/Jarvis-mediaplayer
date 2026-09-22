import requests

from config import SERVICES


def status(title):
    cfg = SERVICES["sonarr"]

    if not cfg["api_key"]:
        return {"error": "SONARR_API_KEY non configurée"}

    try:
        response = requests.get(
            f'{cfg["url"]}/api/v3/series',
            headers={"X-Api-Key": cfg["api_key"]},
            timeout=10
        )
        response.raise_for_status()

        wanted = title.casefold()

        for series in response.json():
            if series.get("title", "").casefold() == wanted:
                stats = series.get("statistics", {})

                return {
                    "found": True,
                    "id": series.get("id"),
                    "title": series.get("title"),
                    "year": series.get("year"),
                    "status": series.get("status"),
                    "monitored": series.get("monitored"),
                    "path": series.get("path"),
                    "seasonCount": stats.get("seasonCount"),
                    "episodeCount": stats.get("episodeCount"),
                    "episodeFileCount": stats.get("episodeFileCount"),
                    "percentOfEpisodes": stats.get("percentOfEpisodes")
                }

        return {
            "found": False,
            "title": title
        }

    except requests.RequestException as e:
        return {"error": f"Erreur Sonarr : {e}"}


TOOL = {
    "type": "function",
    "function": {
        "name": "sonarr_status",
        "description":
            "Recherche une série TV dans Sonarr et retourne son état réel.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titre de la série TV"
                }
            },
            "required": ["title"]
        }
    }
}
