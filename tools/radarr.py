import requests

from config import SERVICES


def _get_movies_and_queue():
    cfg = SERVICES["radarr"]

    if not cfg["api_key"]:
        raise RuntimeError("RADARR_API_KEY non configurée")

    headers = {"X-Api-Key": cfg["api_key"]}

    movies_response = requests.get(
        f'{cfg["url"]}/api/v3/movie',
        headers=headers,
        timeout=10
    )
    movies_response.raise_for_status()

    queue_response = requests.get(
        f'{cfg["url"]}/api/v3/queue',
        headers=headers,
        params={"page": 1, "pageSize": 100},
        timeout=10
    )
    queue_response.raise_for_status()

    return movies_response.json(), queue_response.json().get("records", [])


def _find_movie(movies, title):
    wanted = title.casefold()

    return next(
        (
            movie for movie in movies
            if movie.get("title", "").casefold() == wanted
        ),
        None
    )


def status(title):
    try:
        movies, queue = _get_movies_and_queue()
        movie = _find_movie(movies, title)

        if movie is None:
            return {"found": False, "title": title}

        movie_id = movie.get("id")
        queue_item = next(
            (item for item in queue if item.get("movieId") == movie_id),
            None
        )

        return {
            "found": True,
            "id": movie_id,
            "title": movie.get("title"),
            "originalTitle": movie.get("originalTitle"),
            "year": movie.get("year"),
            "monitored": movie.get("monitored"),
            "hasFile": movie.get("hasFile"),
            "status": movie.get("status"),
            "path": movie.get("path"),
            "inQueue": queue_item is not None
        }

    except (requests.RequestException, RuntimeError) as e:
        return {"error": f"Erreur Radarr : {e}"}


def queue_status(title):
    try:
        movies, queue = _get_movies_and_queue()
        movie = _find_movie(movies, title)

        if movie is None:
            return {"found": False, "title": title, "inQueue": False}

        movie_id = movie.get("id")
        queue_item = next(
            (item for item in queue if item.get("movieId") == movie_id),
            None
        )

        if queue_item is None:
            return {
                "found": True,
                "title": movie.get("title"),
                "inQueue": False
            }

        return {
            "found": True,
            "title": movie.get("title"),
            "inQueue": True,
            "status": queue_item.get("status"),
            "trackedDownloadStatus": queue_item.get("trackedDownloadStatus"),
            "trackedDownloadState": queue_item.get("trackedDownloadState"),
            "size": queue_item.get("size"),
            "sizeleft": queue_item.get("sizeleft"),
            "timeleft": queue_item.get("timeleft")
        }

    except (requests.RequestException, RuntimeError) as e:
        return {"error": f"Erreur Radarr : {e}"}


def _select_request_settings(cfg, root_folders, quality_profiles, french):
    root_path = cfg.get("root_folder")
    profile_name = cfg.get("french_profile") if french else cfg.get("default_profile")

    if not root_path:
        return None, None, "RADARR_ROOT_FOLDER non configuré"
    if not profile_name:
        key = "RADARR_FRENCH_PROFILE" if french else "RADARR_DEFAULT_PROFILE"
        return None, None, f"{key} non configuré"

    root = next((item for item in root_folders if item.get("path") == root_path), None)
    if root is None:
        return None, None, f"Root folder Radarr introuvable : {root_path}"

    profile = next(
        (item for item in quality_profiles if item.get("name") == profile_name),
        None
    )
    if profile is None:
        return None, None, f"Quality profile Radarr introuvable : {profile_name}"

    return root, profile, None


def request_movie(tmdb_id, french=False):
    """Ajoute un film TMDB à Radarr et lance sa recherche."""
    cfg = SERVICES["radarr"]
    if not cfg["api_key"]:
        return {"error": "RADARR_API_KEY non configurée"}

    headers = {"X-Api-Key": cfg["api_key"]}

    try:
        lookup = requests.get(
            f'{cfg["url"]}/api/v3/movie/lookup/tmdb',
            headers=headers,
            params={"tmdbId": tmdb_id},
            timeout=10
        )
        lookup.raise_for_status()
        movie = lookup.json()

        existing = status(movie.get("title", ""))
        if existing.get("found"):
            return {
                "added": False,
                "alreadyExists": True,
                **existing
            }

        roots = requests.get(
            f'{cfg["url"]}/api/v3/rootfolder',
            headers=headers,
            timeout=10
        )
        roots.raise_for_status()
        root_folders = roots.json()

        profiles = requests.get(
            f'{cfg["url"]}/api/v3/qualityprofile',
            headers=headers,
            timeout=10
        )
        profiles.raise_for_status()
        quality_profiles = profiles.json()

        root, profile, error = _select_request_settings(
            cfg, root_folders, quality_profiles, french
        )
        if error:
            return {"error": error}

        payload = {
            **movie,
            "qualityProfileId": profile["id"],
            "rootFolderPath": root["path"],
            "monitored": True,
            "addOptions": {"searchForMovie": True},
        }

        response = requests.post(
            f'{cfg["url"]}/api/v3/movie',
            headers=headers,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        added = response.json()

        return {
            "added": True,
            "id": added.get("id"),
            "tmdbId": added.get("tmdbId"),
            "title": added.get("title"),
            "year": added.get("year"),
            "monitored": added.get("monitored"),
            "qualityProfile": profile.get("name"),
            "rootFolderPath": root.get("path"),
        }

    except requests.RequestException as e:
        return {"error": f"Erreur Radarr : {e}"}


TOOL = {
    "type": "function",
    "function": {
        "name": "radarr_status",
        "description": "Recherche un film dans Radarr et retourne son état dans la bibliothèque Radarr.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Titre du film"}
            },
            "required": ["title"]
        }
    }
}


QUEUE_TOOL = {
    "type": "function",
    "function": {
        "name": "radarr_queue_status",
        "description": "Vérifie si un film est actuellement dans la queue de téléchargement de Radarr et retourne l'état du téléchargement.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Titre du film"}
            },
            "required": ["title"]
        }
    }
}


REQUEST_TOOL = {
    "type": "function",
    "function": {
        "name": "radarr_request_movie",
        "description": (
            "Ajoute dans Radarr un film identifié par son identifiant TMDB et lance immédiatement sa recherche. "
            "À utiliser uniquement si l'utilisateur demande explicitement de télécharger ou d'ajouter le film. "
            "Le profil qualité standard est utilisé par défaut ; french=true uniquement si l'utilisateur demande explicitement une version française."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tmdb_id": {
                    "type": "integer",
                    "description": "Identifiant TMDB exact du film"
                },
                "french": {
                    "type": "boolean",
                    "description": "true uniquement si l'utilisateur demande explicitement une version française",
                    "default": False
                }
            },
            "required": ["tmdb_id"]
        }
    }
}
