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
            return {
                "found": False,
                "title": title
            }

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
            return {
                "found": False,
                "title": title,
                "inQueue": False
            }

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


TOOL = {
    "type": "function",
    "function": {
        "name": "radarr_status",
        "description": "Recherche un film dans Radarr et retourne son état dans la bibliothèque Radarr.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titre du film"
                }
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
                "title": {
                    "type": "string",
                    "description": "Titre du film"
                }
            },
            "required": ["title"]
        }
    }
}
