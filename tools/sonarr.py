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

        # First match the actual queue release title. Release names append
        # season/episode/quality tokens, so the normalized requested title is
        # expected as a prefix/subsequence at the beginning of that release.
        queue_match = next(
            (item for item in records
             if wanted and _title_key(item.get("title")).startswith(wanted)),
            None,
        )
        if queue_match is not None:
            series = series_by_id.get(queue_match.get("seriesId"))
        else:
            series = None

        queued_series_ids = {item.get("seriesId") for item in records}
        if series is None:
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


def request_series(tvdb_id, scope="series", season=None, episode=None, french=False, force=False):
    """Add/resolve a TVDB series, then search the requested series/season/episode."""
    cfg = SERVICES["sonarr"]
    if not cfg["api_key"]:
        return {"error": "SONARR_API_KEY non configurée"}
    if scope not in {"series", "season", "episode"}:
        return {"error": f"Granularité Sonarr invalide : {scope}"}
    if scope in {"season", "episode"} and season is None:
        return {"error": "Le numéro de saison est requis"}
    if scope == "episode" and episode is None:
        return {"error": "Le numéro d'épisode est requis"}

    headers = {"X-Api-Key": cfg["api_key"]}
    try:
        series_response = requests.get(f'{cfg["url"]}/api/v3/series', headers=headers, timeout=10)
        series_response.raise_for_status()
        series = next((x for x in series_response.json() if x.get("tvdbId") == tvdb_id), None)
        added = False

        if series is None:
            lookup = requests.get(
                f'{cfg["url"]}/api/v3/series/lookup',
                headers=headers, params={"term": f"tvdb:{tvdb_id}"}, timeout=10
            )
            lookup.raise_for_status()
            matches = lookup.json()
            if not matches:
                return {"error": f"Série TVDB {tvdb_id} introuvable dans Sonarr"}
            candidate = matches[0]

            roots = requests.get(f'{cfg["url"]}/api/v3/rootfolder', headers=headers, timeout=10)
            roots.raise_for_status()
            profiles = requests.get(f'{cfg["url"]}/api/v3/qualityprofile', headers=headers, timeout=10)
            profiles.raise_for_status()
            root_path = cfg.get("root_folder")
            profile_name = cfg.get("french_profile") if french else cfg.get("default_profile")
            root = next((x for x in roots.json() if x.get("path") == root_path), None)
            profile = next((x for x in profiles.json() if x.get("name") == profile_name), None)
            if root is None:
                return {"error": f"Root folder Sonarr introuvable/non configuré : {root_path}"}
            if profile is None:
                return {"error": f"Quality profile Sonarr introuvable/non configuré : {profile_name}"}

            payload = {
                **candidate,
                "qualityProfileId": profile["id"],
                "rootFolderPath": root["path"],
                "monitored": True,
                "seasonFolder": True,
                "addOptions": {"searchForMissingEpisodes": False},
            }
            add = requests.post(f'{cfg["url"]}/api/v3/series', headers=headers, json=payload, timeout=15)
            add.raise_for_status()
            series = add.json()
            added = True

        series_id = series.get("id")
        command = None
        command_body = None

        if scope == "series":
            command = "SeriesSearch"
            command_body = {"name": command, "seriesId": series_id}
        elif scope == "season":
            command = "SeasonSearch"
            command_body = {"name": command, "seriesId": series_id, "seasonNumber": season}
        else:
            episodes_response = requests.get(
                f'{cfg["url"]}/api/v3/episode',
                headers=headers, params={"seriesId": series_id}, timeout=10
            )
            episodes_response.raise_for_status()
            target = next(
                (x for x in episodes_response.json()
                 if x.get("seasonNumber") == season and x.get("episodeNumber") == episode),
                None
            )
            if target is None:
                return {"error": f"Épisode S{season:02d}E{episode:02d} introuvable dans Sonarr"}
            # Explicit episode requests, including re-requests, deliberately run
            # EpisodeSearch even when a file already exists.
            command = "EpisodeSearch"
            command_body = {"name": command, "episodeIds": [target["id"]]}

        search = requests.post(
            f'{cfg["url"]}/api/v3/command', headers=headers, json=command_body, timeout=15
        )
        search.raise_for_status()
        command_result = search.json()
        return {
            "requested": True, "added": added, "title": series.get("title"),
            "tvdbId": tvdb_id, "scope": scope, "season": season, "episode": episode,
            "force": force, "command": command, "commandId": command_result.get("id"),
        }
    except requests.RequestException as e:
        return {"error": f"Erreur Sonarr : {e}"}


REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "sonarr_request_series",
        "description": (
            "Ajoute si nécessaire une série TV identifiée par son TVDB ID puis lance une recherche Sonarr. "
            "scope=series pour toute la série, season pour une saison entière, episode pour un épisode précis. "
            "Une demande explicite d'un épisode lance EpisodeSearch même si cet épisode possède déjà un fichier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tmdb_id": {"type": "integer", "description": "TMDB ID exact de la série retourné par le catalogue Seerr"},
                "scope": {"type": "string", "enum": ["series", "season", "episode"]},
                "season": {"type": "integer", "minimum": 0},
                "episode": {"type": "integer", "minimum": 1},
                "french": {"type": "boolean", "default": False},
                "force": {"type": "boolean", "default": False, "description": "true when user explicitly asks to re-download/re-request"}
            },
            "required": ["tmdb_id", "scope"]
        }
    }
}

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
