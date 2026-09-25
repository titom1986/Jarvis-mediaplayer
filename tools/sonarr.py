import re
import unicodedata

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


def _title_key(value):
    value = unicodedata.normalize("NFKD", value or "").casefold()
    return "".join(ch for ch in value if ch.isalnum())


def queue_status(title):
    cfg = SERVICES["sonarr"]
    if not cfg["api_key"]:
        return {"error": "SONARR_API_KEY non configurée"}
    headers = {"X-Api-Key": cfg["api_key"]}
    try:
        # Queue is the source of truth for a download-status question. Resolve
        # its native seriesId values back to Sonarr series instead of requiring
        # the user's title to exactly equal Sonarr's stored punctuation.
        queue_response = requests.get(
            f'{cfg["url"]}/api/v3/queue',
            headers=headers,
            params={"page": 1, "pageSize": 100},
            timeout=10,
        )
        queue_response.raise_for_status()
        records = queue_response.json().get("records", [])

        series_response = requests.get(
            f'{cfg["url"]}/api/v3/series', headers=headers, timeout=10
        )
        series_response.raise_for_status()
        series_by_id = {
            item.get("id"): item for item in series_response.json()
            if item.get("id") is not None
        }

        wanted = _title_key(title)
        queued_series_ids = {item.get("seriesId") for item in records}
        series = next(
            (series_by_id[series_id] for series_id in queued_series_ids
             if series_id in series_by_id
             and _title_key(series_by_id[series_id].get("title")) == wanted),
            None,
        )
        if series is None:
            series = next(
                (item for item in series_by_id.values()
                 if _title_key(item.get("title")) == wanted),
                None,
            )
        if series is None:
            return {"found": False, "title": title, "inQueue": False}

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
