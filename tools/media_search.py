from concurrent.futures import ThreadPoolExecutor, as_completed

from tools import seerr


def _norm(value):
    return str(value or "").strip().casefold()


def _resolve_person(name):
    result = seerr.search(name)

    if "error" in result:
        return None

    wanted = _norm(name)

    for item in result.get("results", []):
        if (
            item.get("mediaType") == "person"
            and _norm(item.get("title")) == wanted
        ):
            return item

    for item in result.get("results", []):
        if item.get("mediaType") == "person":
            return item

    return None


def _credits_ids(person_id, media_type):
    credits = seerr.person_credits(person_id, limit=None)

    if "error" in credits:
        return set()

    return {
        item["id"]
        for item in credits.get("results", [])
        if item.get("id") and item.get("mediaType") == media_type
    }


def _person_ids(names, media_type):
    """
    Toutes les personnes d'un groupe sont en AND :
    intersection de leurs filmographies.
    """
    credit_sets = []

    for name in names or []:
        person = _resolve_person(name)

        if not person:
            return set()

        credit_sets.append(
            _credits_ids(person["id"], media_type)
        )

    if not credit_sets:
        return None

    result = credit_sets[0].copy()

    for ids in credit_sets[1:]:
        result &= ids

    return result


def _match_values(required, actual):
    """
    Toutes les valeurs d'une catégorie sont en AND.
    """
    if not required:
        return True

    actual = {_norm(v) for v in actual if v}

    return all(
        _norm(value) in actual
        for value in required
    )


def _match_dates(required, release_date):
    """
    Toutes les contraintes de date d'un groupe sont en AND.
    """
    if not required:
        return True

    if not release_date:
        return False

    try:
        year = int(str(release_date)[:4])
    except (TypeError, ValueError):
        return False

    for condition in required:
        if not isinstance(condition, dict):
            return False

        year_from = condition.get("from")
        year_to = condition.get("to")

        if year_from is not None and year < int(year_from):
            return False

        if year_to is not None and year > int(year_to):
            return False

    return True


def _match_metadata(group, media):
    """
    AND entre toutes les conditions du groupe.
    people est traité séparément via les filmographies.
    """
    if not _match_values(
        group.get("genres", []),
        media.get("genres", [])
    ):
        return False

    if not _match_values(
        group.get("keywords", []),
        media.get("keywords", [])
    ):
        return False

    if not _match_dates(
        group.get("dates", []),
        media.get("releaseDate")
    ):
        return False

    return True


def _group_candidate_ids(group, media_type):
    """
    Point d'entrée d'un groupe par ses personnes.
    """
    return _person_ids(
        group.get("people", []),
        media_type
    )


def media_search(media_type, include=None, exclude=None):
    if media_type not in ("movie", "tv"):
        return {"error": "media_type doit être 'movie' ou 'tv'"}

    include = include or []
    exclude = exclude or []

    if not include:
        return {"error": "include doit contenir au moins un groupe"}

    # V1 : chaque groupe include doit avoir au moins une personne,
    # afin d'obtenir un ensemble fini de candidats depuis Seerr.
    include_candidates = []

    for group in include:
        ids = _group_candidate_ids(group, media_type)

        if ids is None:
            return {
                "error": (
                    "Chaque groupe include doit actuellement contenir "
                    "au moins une personne."
                )
            }

        include_candidates.append((group, ids))

    # OR entre les groupes include.
    candidate_ids = set()

    for _, ids in include_candidates:
        candidate_ids |= ids

    # Prépare les intersections people de chaque groupe exclude.
    exclude_people = []

    for group in exclude:
        ids = _group_candidate_ids(group, media_type)

        # Pas de people dans ce groupe = aucune contrainte people.
        exclude_people.append((group, ids))

    def group_matches(group, people_ids, media_id, media):
        # Si le groupe contient people, le média doit appartenir
        # à leur intersection.
        if people_ids is not None and media_id not in people_ids:
            return False

        return _match_metadata(group, media)

    def inspect(media_id):
        try:
            media = seerr.media_details(media_id, media_type)

            if "error" in media:
                return None

            # OR entre les groupes include.
            included = any(
                media_id in ids
                and _match_metadata(group, media)
                for group, ids in include_candidates
            )

            if not included:
                return None

            # OR entre les groupes exclude.
            excluded = any(
                group_matches(
                    group,
                    people_ids,
                    media_id,
                    media
                )
                for group, people_ids in exclude_people
            )

            if excluded:
                return None

            return {
                "mediaType": media.get("mediaType"),
                "id": media.get("id"),
                "title": media.get("title"),
                "releaseDate": media.get("releaseDate"),
                "genres": media.get("genres", []),
                "keywords": media.get("keywords", []),
            }

        except Exception:
            # Seerr contient actuellement quelques IDs de crédits
            # que son endpoint détail ne résout plus.
            return None

    matches = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(inspect, media_id)
            for media_id in candidate_ids
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                matches.append(result)

    matches.sort(
        key=lambda x: x.get("releaseDate") or "",
        reverse=True
    )

    return {
        "mediaType": media_type,
        "include": include,
        "exclude": exclude,
        "candidates": len(candidate_ids),
        "count": len(matches),
        "results": matches,
    }


MEDIA_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": (
            "Recherche des films ou séries selon des groupes de critères. "
            "Toutes les conditions d'un groupe sont reliées par AND. "
            "Les groupes sont reliés par OR. "
            "include et exclude ont exactement la même structure. "
            "Un groupe peut combiner people, genres, keywords et dates. "
            "N'ajoute aucun critère non demandé."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "media_type": {
                    "type": "string",
                    "enum": ["movie", "tv"]
                },
                "include": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "people": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "dates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "integer"},
                                        "to": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                },
                "exclude": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "people": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "dates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "from": {"type": "integer"},
                                        "to": {"type": "integer"}
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "required": ["media_type", "include"]
        }
    }
}
