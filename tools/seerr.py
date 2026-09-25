import requests
from urllib.parse import quote
import json
import time
from config import SERVICES

SEERR_URL = (SERVICES["seerr"]["url"] or "http://127.0.0.1:5055").rstrip("/")
SEERR_API_KEY = SERVICES["seerr"]["api_key"]

_PERSON_NAMES = {}


def _get(url, **kwargs):
    """GET with diagnostic timing. Never logs headers or API keys."""
    started = time.perf_counter()
    try:
        response = requests.get(url, **kwargs)
        elapsed = time.perf_counter() - started
        safe_path = url.split("/api/v1/", 1)[-1] if "/api/v1/" in url else url
        print("[SEERR]", json.dumps({"path": safe_path, "params": kwargs.get("params"), "status": response.status_code, "wall_s": round(elapsed, 3)}, ensure_ascii=False, sort_keys=True))
        return response
    except Exception as exc:
        elapsed = time.perf_counter() - started
        safe_path = url.split("/api/v1/", 1)[-1] if "/api/v1/" in url else url
        print("[SEERR]", json.dumps({"path": safe_path, "params": kwargs.get("params"), "error": type(exc).__name__, "wall_s": round(elapsed, 3)}, ensure_ascii=False, sort_keys=True))
        raise

TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_search",
        "description": (
            "Recherche un film ou une série dans le catalogue Seerr. "
            "Un résultat trouvé signifie uniquement que Seerr connaît ce média. "
            "Cela NE signifie PAS que le média est disponible, téléchargé, demandé, "
            "présent dans Plex, Radarr ou Sonarr. "
            "Utilise cet outil pour découvrir ou identifier un média."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Titre ou termes à rechercher"
                }
            },
            "required": ["query"]
        }
    }
}

def search(query):
    if not SEERR_API_KEY:
        return {"error": "SEERR_API_KEY non configurée"}

    try:
        r = _get(
            f"{SEERR_URL}/api/v1/search",
            headers={"X-Api-Key": SEERR_API_KEY},
            params={"query": query},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()

        results = []

        for item in data.get("results", [])[:5]:
            if item.get("mediaType") == "person" and item.get("id"):
                _PERSON_NAMES[item["id"]] = item.get("name") or item.get("title") or query

            results.append({
                "mediaType": item.get("mediaType"),
                "id": item.get("id"),
                "title": item.get("title") or item.get("name"),
                "originalTitle": item.get("originalTitle") or item.get("originalName"),
                "releaseDate": item.get("releaseDate") or item.get("firstAirDate"),
            })

        return {
            "found": bool(results),
            "query": query,
        "results": results[:20]
        }

    except Exception as e:
        return {"error": str(e), "query": query}


PERSON_CREDITS_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_person_credits",
        "description": (
            "Récupère la filmographie d'une personne identifiée dans Seerr. "
            "Utilise cet outil après seerr_search lorsqu'un résultat de type person "
            "a fourni son identifiant. Permet de trouver des films ou séries avec "
            "un acteur, une actrice ou une autre personnalité."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "integer",
                    "description": "Identifiant de la personne retourné par Seerr"
                }
            },
            "required": ["person_id"]
        }
    }
}


def person_credits(person_id, limit=20):
    if not SEERR_API_KEY:
        return {"error": "SEERR_API_KEY non configurée"}

    r = _get(
        f"{SEERR_URL}/api/v1/person/{person_id}/combined_credits",
        headers={"X-Api-Key": SEERR_API_KEY},
        timeout=15,
    )
    r.raise_for_status()

    data = r.json()
    results = []

    for item in data.get("cast", []):
        results.append({
            "mediaType": item.get("mediaType"),
            "id": item.get("id"),
            "title": item.get("title") or item.get("name"),
            "releaseDate": item.get("releaseDate") or item.get("firstAirDate"),
            "character": item.get("character"),
            "genreIds": item.get("genreIds", []),
        })

    results.sort(
        key=lambda x: x.get("releaseDate") or "",
        reverse=True
    )

    return {
        "personId": person_id,
        "count": len(results),
        "results": results if limit is None else results[:limit]
    }


def genres(media_type):
    """Liste les genres officiels TMDB exposés par Seerr."""
    if media_type not in ("movie", "tv"):
        return []

    if not SEERR_API_KEY:
        return []

    r = _get(
        f"{SEERR_URL}/api/v1/genres/{media_type}",
        headers={"X-Api-Key": SEERR_API_KEY},
        params={"language": "en"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()

    return data if isinstance(data, list) else data.get("genres", [])


def search_keyword(query):
    """Résout un nom de mot-clé en identifiant TMDB via Seerr."""
    if not SEERR_API_KEY:
        return {"error": "SEERR_API_KEY non configurée"}

    try:
        # Seerr validates this endpoint before its query parser and requires
        # reserved characters (including spaces) to already be percent-encoded.
        # Supplying the value through requests' params= is rejected by Seerr even
        # though requests correctly serializes it on the wire, so build only this
        # query value explicitly and keep all other endpoints on params=.
        encoded_query = quote(query, safe="")
        r = _get(
            f"{SEERR_URL}/api/v1/search/keyword?query={encoded_query}&page=1",
            headers={"X-Api-Key": SEERR_API_KEY},
            timeout=15,
        )
        if not r.ok:
            body = (r.text or "").strip()
            return {
                "error": f"Recherche keyword Seerr impossible : HTTP {r.status_code}",
                "query": query,
                "response": body[:1000],
            }
        return r.json()
    except requests.RequestException as e:
        return {
            "error": f"Recherche keyword Seerr impossible : {e}",
            "query": query,
        }


def discover(
    media_type,
    page=1,
    genre_ids=None,
    keyword_ids=None,
    date_from=None,
    date_to=None,
    sort_by="popularity.desc",
    exclude_keyword_ids=None,
):
    """Interroge l'API Discover de Seerr avec des filtres TMDB natifs."""
    if not SEERR_API_KEY:
        return {"error": "SEERR_API_KEY non configurée"}

    if media_type not in ("movie", "tv"):
        return {"error": f"Type de média invalide : {media_type}"}

    params = {"page": page, "sortBy": sort_by}

    if genre_ids:
        params["genre"] = ",".join(str(value) for value in genre_ids)

    if keyword_ids:
        params["keywords"] = ",".join(str(value) for value in keyword_ids)

    if exclude_keyword_ids:
        params["excludeKeywords"] = ",".join(str(value) for value in exclude_keyword_ids)

    if media_type == "movie":
        path = "movies"
        if date_from:
            params["primaryReleaseDateGte"] = date_from
        if date_to:
            params["primaryReleaseDateLte"] = date_to
    else:
        path = "tv"
        if date_from:
            params["firstAirDateGte"] = date_from
        if date_to:
            params["firstAirDateLte"] = date_to

    r = _get(
        f"{SEERR_URL}/api/v1/discover/{path}",
        headers={"X-Api-Key": SEERR_API_KEY},
        params=params,
        timeout=15,
    )
    r.raise_for_status()
    return r.json()


def media_details(media_id, media_type="movie"):
    if not SEERR_API_KEY:
        return {"error": "SEERR_API_KEY non configurée"}

    if media_type not in ("movie", "tv"):
        return {"error": f"Type de média invalide : {media_type}"}

    r = _get(
        f"{SEERR_URL}/api/v1/{media_type}/{media_id}",
        headers={"X-Api-Key": SEERR_API_KEY},
        timeout=15,
    )
    r.raise_for_status()

    data = r.json()

    return {
        "mediaType": media_type,
        "id": data.get("id"),
        "title": data.get("title") or data.get("name"),
        "originalTitle": data.get("originalTitle") or data.get("originalName"),
        "releaseDate": data.get("releaseDate") or data.get("firstAirDate"),
        "genres": [
            g.get("name")
            for g in data.get("genres", [])
            if g.get("name")
        ],
        "keywords": [
            k.get("name")
            for k in data.get("keywords", [])
            if k.get("name")
        ],
        "rating": data.get("voteAverage"),
        "voteCount": data.get("voteCount"),
        "overview": data.get("overview") or "",
    }

MEDIA_DETAILS_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_details",
        "description": (
            "Récupère les détails sémantiques d'un film ou d'une série identifié dans Seerr : "
            "genres, mots-clés et synopsis. "
            "Utilise cet outil après avoir identifié un média lorsque la demande contient "
            "des critères de contenu, thème, ambiance ou genre, par exemple Noël, romance, "
            "horreur, guerre, famille, espace, etc. "
            "Ne l'utilise pas pour une simple vérification de disponibilité."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "media_id": {
                    "type": "integer",
                    "description": "Identifiant du média retourné par Seerr"
                },
                "media_type": {
                    "type": "string",
                    "enum": ["movie", "tv"],
                    "description": "Type du média retourné par Seerr"
                }
            },
            "required": ["media_id", "media_type"]
        }
    }
}

def person_media_catalog(person_id):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    credits = person_credits(person_id, limit=None)

    if "error" in credits:
        return credits

    candidates = [
        c for c in credits.get("results", [])
        if c.get("mediaType") in ("movie", "tv") and c.get("id")
    ]

    def inspect(credit):
        try:
            details = media_details(
                credit["id"],
                credit["mediaType"]
            )

            if "error" in details:
                return None

            return {
                "mediaType": credit["mediaType"],
                "id": credit["id"],
                "title": details.get("title"),
                "releaseDate": details.get("releaseDate"),
                "genres": details.get("genres", []),
                "keywords": details.get("keywords", []),
            }

        except Exception:
            # Certains IDs renvoyés par combined_credits ne sont
            # plus résolus correctement par Seerr.
            return None

    results = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(inspect, c) for c in candidates]

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    results.sort(
        key=lambda x: x.get("releaseDate") or "",
        reverse=True
    )

    return {
        "personId": person_id,
        "scanned": len(candidates),
        "count": len(results),
        "results": results,
    }


PERSON_MEDIA_CATALOG_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_person_media_catalog",
        "description": (
            "Retourne la filmographie détaillée d'une personne depuis Seerr. "
            "Fournit pour chaque film ou série son identifiant, son titre, "
            "sa date de sortie, ses genres et ses mots-clés. "
            "Utilise cet outil lorsqu'une demande nécessite de raisonner sur "
            "la filmographie d'une personne, par exemple selon le sujet, "
            "le genre, la période ou la présence d'autres personnes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "person_id": {
                    "type": "integer",
                    "description": "Identifiant Seerr de la personne"
                }
            },
            "required": ["person_id"]
        }
    }
}
