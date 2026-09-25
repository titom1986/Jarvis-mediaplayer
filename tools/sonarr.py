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


def queue_status(title):
    cfg = SERVICES["sonarr"]
    if not cfg["api_key"]:
        return {"error": "SONARR_API_KEY non configurée"}
    headers = {"X-Api-Key": cfg["api_key"]}
    try:
        series_response = requests.get(
            f'{cfg["url"]}/api/v3/series', headers=headers, timeout=10
        )
        series_response.raise_for_status()
        wanted = title.casefold()
        series = next(
            (item for item in series_response.json()
             if item.get("title", "").casefold() == wanted),
            None,
        )
        if series is None:
            return {"found": False, "title": title, "inQueue": False}

        queue_response = requests.get(
            f'{cfg["url"]}/api/v3/queue',
            headers=headers,
            params={"page": 1, "pageSize": 100},
            timeout=10,
        )
        queue_response.raise_for_status()
        records = queue_response.json().get("records", [])
        items = [item for item in records if item.get("seriesId") == series.get("id")]
        if not items:
            return {"found": True, "title": series.get("title"), "inQueue": False, "items": []}

        return {
            "found": True,
            "title": series.get("title"),
            "inQueue": True,
            "items": [{
                "id": item.get("id"),
                "episodeId": item.get("episodeId"),
                "seasonNumber": item.get("seasonNumber"),
                "episode": item.get("episode"),
                "title": item.get("title"),
                "status": item.get("status"),
                "trackedDownloadStatus": item.get("trackedDownloadStatus"),
                "trackedDownloadState": item.get("trackedDownloadState"),
                "size": item.get("size"),
                "sizeleft": item.get("sizeleft"),
                "estimatedCompletionTime": item.get("estimatedCompletionTime"),
                "quality": item.get("quality"),
                "languages": item.get("languages"),
                "protocol": item.get("protocol"),
                "downloadClient": item.get("downloadClient"),
            } for item in items],
        }
    except requests.RequestException as e:
        return {"error": f"Erreur Sonarr : {e}"}


QUEUE_TOOL = {
    "type": "function",
    "function": {
        "name": "sonarr_queue_status",
        "description": "Vérifie les téléchargements actuellement présents dans la queue Sonarr pour une série et retourne leurs états réels.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Titre exact de la série TV"}
            },
            "required": ["title"]
        }
    }
}


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
