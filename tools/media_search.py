from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter
import re
import re

from tools import seerr


def _norm(value):
    # Lexical normalization only: catalogue labels such as "Science Fiction"
    # and model output such as "science-fiction" should resolve identically.
    value = str(value or "").strip().casefold()
    return re.sub(r"[\\s_-]+", " ", value)


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


def _resolve_genre_ids(names, media_type):
    if not names:
        return []

    available = {
        _norm(item.get("name")): item.get("id")
        for item in seerr.genres(media_type)
        if item.get("id")
    }

    ids = []
    for name in names:
        genre_id = available.get(_norm(name))
        if genre_id is None:
            return None
        ids.append(genre_id)

    return ids


def _resolve_keyword_ids(names):
    ids = []

    for name in names or []:
        result = seerr.search_keyword(name)

        if "error" in result:
            return None

        wanted = _norm(name)
        exact = next(
            (
                item
                for item in result.get("results", [])
                if _norm(item.get("name")) == wanted
            ),
            None,
        )

        if not exact or not exact.get("id"):
            return None

        ids.append(exact["id"])

    return ids


def _date_bounds(dates):
    year_from = None
    year_to = None

    for condition in dates or []:
        if not isinstance(condition, dict):
            continue

        if condition.get("from") is not None:
            value = int(condition["from"])
            year_from = max(year_from, value) if year_from is not None else value

        if condition.get("to") is not None:
            value = int(condition["to"])
            year_to = min(year_to, value) if year_to is not None else value

    return year_from, year_to


def _discover_ids(group, media_type, exclude_keyword_ids=None, max_pages=100):
    genre_ids = _resolve_genre_ids(group.get("genres", []), media_type)
    keyword_ids = _resolve_keyword_ids(group.get("keywords", []))

    if genre_ids is None or keyword_ids is None:
        return set()

    year_from, year_to = _date_bounds(group.get("dates", []))

    date_from = f"{year_from}-01-01" if year_from is not None else None
    date_to = f"{year_to}-12-31" if year_to is not None else None

    ids = set()
    page = 1

    while page <= max_pages:
        result = seerr.discover(
            media_type,
            page=page,
            genre_ids=genre_ids,
            keyword_ids=keyword_ids,
            date_from=date_from,
            date_to=date_to,
            exclude_keyword_ids=exclude_keyword_ids,
        )

        if "error" in result:
            break

        ids.update(
            item["id"]
            for item in result.get("results", [])
            if item.get("id")
        )

        total_pages = min(
            int(result.get("totalPages") or result.get("total_pages") or 1),
            max_pages,
        )

        if page >= total_pages:
            break

        page += 1

    return ids


def _group_candidate_ids(group, media_type):
    """
    Une personne est une optimisation de sélection, pas une obligation.
    Sans people, Seerr Discover construit l'ensemble de candidats.
    """
    people_ids = _person_ids(group.get("people", []), media_type)

    if people_ids is not None:
        return people_ids

    return _discover_ids(group, media_type)


def media_search(media_type, include=None, exclude=None):
    if media_type not in ("movie", "tv"):
        return {"error": "media_type doit être 'movie' ou 'tv'"}

    include = include or []
    exclude = exclude or []

    if not include:
        return {"error": "include doit contenir au moins un groupe"}

    include_candidates = []

    # Une exclusion composée uniquement de keywords peut être poussée directement
    # dans TMDB/Seerr Discover. Cela évite de récupérer puis enrichir des milliers
    # de médias uniquement pour les éliminer ensuite.
    native_exclude_keyword_ids = []
    residual_exclude = []
    for group in exclude:
        if set(group) <= {"keywords"} and group.get("keywords"):
            ids = _resolve_keyword_ids(group["keywords"])
            if ids is not None:
                native_exclude_keyword_ids.extend(ids)
                continue
        residual_exclude.append(group)

    timings = {}
    t_candidates = perf_counter()

    for group in include:
        people_ids = _person_ids(group.get("people", []), media_type)
        has_metadata_filters = any(
            group.get(key) for key in ("genres", "keywords", "dates")
        )

        if people_ids is not None and has_metadata_filters:
            # La filmographie est déjà un petit ensemble ciblé. Enrichir ces IDs
            # puis appliquer les critères est beaucoup moins coûteux que parcourir
            # toutes les pages Discover d'un genre/période très large.
            ids = people_ids
        elif people_ids is not None:
            ids = people_ids
        else:
            ids = _discover_ids(
                group,
                media_type,
                exclude_keyword_ids=native_exclude_keyword_ids,
            )
        include_candidates.append((group, ids))

    timings["candidate_selection_s"] = round(perf_counter() - t_candidates, 3)

    # OR entre les groupes include.
    candidate_ids = set()

    for _, ids in include_candidates:
        candidate_ids |= ids

    # Prépare les intersections people de chaque groupe exclude.
    exclude_people = []

    for group in residual_exclude:
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
                "rating": media.get("rating"),
                "voteCount": media.get("voteCount"),
            }

        except Exception:
            # Seerr contient actuellement quelques IDs de crédits
            # que son endpoint détail ne résout plus.
            return None

    matches = []
    t_details = perf_counter()

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(inspect, media_id)
            for media_id in candidate_ids
        ]

        for future in as_completed(futures):
            result = future.result()
            if result:
                matches.append(result)

    timings["details_filter_s"] = round(perf_counter() - t_details, 3)

    matches.sort(
        key=lambda x: (
            x.get("rating") or 0,
            x.get("voteCount") or 0,
        ),
        reverse=True
    )

    return {
        "mediaType": media_type,
        "include": include,
        "exclude": exclude,
        "candidates": len(candidate_ids),
        "count": len(matches),
        "results": matches,
        "_perf": timings,
    }


MEDIA_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": (
            "Recherche des films ou séries selon des groupes de critères. "
            "Toutes les contraintes people, genres, keywords et dates d'un même groupe "
            "sont simultanées (AND). Les groupes sont des alternatives (OR). "
            "Crée un nouveau groupe uniquement si la demande exprime une alternative. "
            "include et exclude ont exactement la même structure. "
            "IMPORTANT : exclude porte déjà la négation. Les valeurs placées dans exclude doivent donc toujours "
            "nommer positivement la propriété à retirer, sans 'not', 'non', 'pas', 'sans' ou autre négation. "
            "Exemple : pour exclure les films dystopiques, utilise exclude=[{'keywords':['dystopia']}], "
            "jamais un keyword comme 'not dystopian'. "
            "Un groupe peut combiner people, genres, keywords et dates et people n'est jamais obligatoire. "
            "people contient les personnes, genres les genres de catalogue, keywords les thèmes/concepts "
            "et dates les périodes. N'ajoute aucun critère non demandé."
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
                                "items": {
                                    "type": "string",
                                    "description": "Thème ou concept positif à inclure, sans opérateur logique."
                                }
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
                                "description": "Concepts à exclure. Chaque valeur nomme positivement le concept interdit ; ne jamais inclure de négation dans la chaîne.",
                                "items": {
                                    "type": "string",
                                    "description": "Nom positif du concept exclu, par exemple 'dystopia', jamais 'not dystopian'."
                                }
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
