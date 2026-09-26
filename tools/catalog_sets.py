"""Atomic, handle-based catalogue operations for small tool-calling models.

The model decides what each natural-language constraint means.  Python only
creates candidate sets and performs deterministic set algebra on opaque handles.
Handles are request-local: call reset() before each user request.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from threading import Lock
import json
import time

from tools import media_search, seerr

_sets = {}
_keyword_vocabulary = {}
_seq = count(1)
_lock = Lock()


def _trace(event, **fields):
    """Diagnostic telemetry for deterministic catalogue execution; never sent to the LLM."""
    print("[CATALOG]", event, json.dumps(fields, ensure_ascii=False, sort_keys=True))


def reset():
    global _seq
    with _lock:
        _sets.clear()
        _keyword_vocabulary.clear()
        _seq = count(1)


def _store(ids, media_type, label):
    ids = set(ids or [])
    with _lock:
        handle = f"s{next(_seq)}"
        _sets[handle] = {"ids": ids, "media_type": media_type}
    return {"set": handle, "count": len(ids), "label": label}


def _get(handle):
    value = _sets.get(handle)
    if value is None:
        raise ValueError(f"unknown set handle: {handle}")
    return value


def _same_type(handles):
    values = [_get(h) for h in handles]
    types = {v["media_type"] for v in values}
    if len(types) != 1:
        raise ValueError("set handles have different media types")
    return values, next(iter(types))


def title(query, media_type):
    """Resolve an explicitly named media target without interpreting words inside its title."""
    result = seerr.search(query)
    if result.get("error"):
        return result
    matches = [
        item for item in result.get("results", [])
        if item.get("mediaType") == media_type and item.get("id") is not None
    ]
    return {
        "query": query,
        "media_type": media_type,
        "found": bool(matches),
        "results": matches[:5],
    }


def person(name, media_type):
    resolved = media_search._resolve_person(name)
    if not resolved:
        return {"found": False, "name": name}
    ids = media_search._credits_ids(resolved["id"], media_type)
    return {"found": True, **_store(ids, media_type, f"person:{name}")}


def _filter_existing(source, predicate, label):
    try:
        value = _get(source)
    except ValueError as exc:
        return {"error": str(exc)}

    ids = list(value["ids"])

    def matches(media_id):
        try:
            item = seerr.media_details(media_id, value["media_type"])
            return media_id if "error" not in item and predicate(item) else None
        except Exception:
            return None

    kept = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(matches, media_id) for media_id in ids]
        for future in as_completed(futures):
            media_id = future.result()
            if media_id is not None:
                kept.add(media_id)
    return _store(kept, value["media_type"], label)


def genre(name, media_type, source=None):
    if source:
        try:
            value = _get(source)
        except ValueError as exc:
            return {"error": str(exc)}
        if value["media_type"] != media_type:
            return {"error": "source handle has different media type"}
        wanted = media_search._norm(name)
        return _filter_existing(
            source,
            lambda item: wanted in {media_search._norm(v) for v in item.get("genres", [])},
            f"genre:{name}",
        )
    if media_search._resolve_genre_ids([name], media_type) is None:
        return {"error": f"unresolved genre: {name}"}
    ids = media_search._discover_ids({"genres": [name]}, media_type)
    return _store(ids, media_type, f"genre:{name}")


def keyword_vocabulary(query, limit=12):
    """Expose Seerr/TMDB keyword labels for semantic selection by the model."""
    result = seerr.search_keyword(query)
    if "error" in result:
        return result
    items = []
    for item in result.get("results", [])[:max(1, min(int(limit), 20))]:
        if item.get("id") and item.get("name"):
            entry = {"id": item["id"], "name": item["name"]}
            items.append(entry)
            _keyword_vocabulary[media_search._norm(item["name"])] = item["id"]
    return {"query": query, "count": len(items), "keywords": items}

def keyword(name, media_type, source=None, aliases=None):
    if source:
        try:
            value = _get(source)
        except ValueError as exc:
            return {"error": str(exc)}
        if value["media_type"] != media_type:
            return {"error": "source handle has different media type"}
        wanted = {media_search._norm(v) for v in [name, *(aliases or [])] if v}
        return _filter_existing(
            source,
            lambda item: bool(wanted & {media_search._norm(v) for v in item.get("keywords", [])}),
            f"keyword:{name}",
        )
    if media_search._resolve_keyword_ids([name]) is None:
        return {"error": f"unresolved keyword: {name}"}
    ids = media_search._discover_ids({"keywords": [name]}, media_type)
    return _store(ids, media_type, f"keyword:{name}")


def years(year_from, year_to, media_type, source=None):
    if source:
        try:
            value = _get(source)
        except ValueError as exc:
            return {"error": str(exc)}
        if value["media_type"] != media_type:
            return {"error": "source handle has different media type"}
        return _filter_existing(
            source,
            lambda item: media_search._match_dates(
                [{"from": year_from, "to": year_to}], item.get("releaseDate")
            ),
            f"years:{year_from}-{year_to}",
        )
    ids = media_search._discover_ids(
        {"dates": [{"from": year_from, "to": year_to}]}, media_type
    )
    return _store(ids, media_type, f"years:{year_from}-{year_to}")


def estimate_constraint(kind, args):
    """Return a cheap cardinality estimate without materializing all Discover pages."""
    media_type = args["media_type"]
    if kind == "catalog_person":
        resolved = media_search._resolve_person(args["name"])
        if not resolved:
            return {"error": f"unresolved person: {args['name']}"}
        ids = media_search._credits_ids(resolved["id"], media_type)
        return {"count": len(ids), "seed_ids": ids}

    group = {}
    if kind == "catalog_genre":
        if media_search._resolve_genre_ids([args["name"]], media_type) is None:
            return {"error": f"unresolved genre: {args['name']}"}
        group["genres"] = [args["name"]]
    elif kind == "catalog_keyword":
        names = [args["name"], *(args.get("aliases") or [])]
        resolved = []
        for candidate in names:
            grounded_id = _keyword_vocabulary.get(media_search._norm(candidate))
            if grounded_id is not None:
                resolved.append(grounded_id)
                continue
            ids = media_search._resolve_keyword_ids([candidate])
            if ids:
                resolved.extend(ids)
        if not resolved:
            return {"error": f"unresolved keyword: {args['name']}"}
        # Concept aliases are alternatives, so estimate their union rather than
        # treating them as an AND conjunction.
        counts = []
        for keyword_id in dict.fromkeys(resolved):
            result = seerr.discover(media_type, page=1, keyword_ids=[keyword_id])
            if "error" in result:
                return {"error": result["error"]}
            value = result.get("totalResults")
            if value is None:
                value = result.get("total_results")
            if value is None:
                value = len(result.get("results", []))
            counts.append(int(value))
        return {"count": sum(counts), "keyword_ids": list(dict.fromkeys(resolved))}
    elif kind == "catalog_years":
        group["dates"] = [{"from": args["year_from"], "to": args["year_to"]}]
    else:
        return {"error": f"unsupported constraint: {kind}"}

    genre_ids = media_search._resolve_genre_ids(group.get("genres", []), media_type)
    keyword_ids = media_search._resolve_keyword_ids(group.get("keywords", []))
    if genre_ids is None or keyword_ids is None:
        return {"error": "unresolved catalogue constraint"}
    year_from, year_to = media_search._date_bounds(group.get("dates", []))
    result = seerr.discover(
        media_type,
        page=1,
        genre_ids=genre_ids,
        keyword_ids=keyword_ids,
        date_from=f"{year_from}-01-01" if year_from is not None else None,
        date_to=f"{year_to}-12-31" if year_to is not None else None,
    )
    if "error" in result:
        return {"error": result["error"]}
    # Seerr/TMDB Discover exposes totalResults. Fall back to the first-page
    # cardinality only when the server omits it; the estimate is used for order,
    # never for correctness.
    count_value = result.get("totalResults")
    if count_value is None:
        count_value = result.get("total_results")
    if count_value is None:
        count_value = len(result.get("results", []))
    return {"count": int(count_value)}


def materialize_constraint(kind, args, source=None, seed_ids=None):
    """Materialize one constraint, optionally refining an existing set."""
    media_type = args["media_type"]
    if source is not None:
        if kind == "catalog_person":
            resolved = media_search._resolve_person(args["name"])
            if not resolved:
                return {"error": f"unresolved person: {args['name']}"}
            person_ids = media_search._credits_ids(resolved["id"], media_type)
            src = _get(source)
            return _store(src["ids"] & person_ids, media_type, f"person:{args['name']}")
        if kind == "catalog_genre":
            return genre(args["name"], media_type, source)
        if kind == "catalog_keyword":
            return keyword(args["name"], media_type, source, aliases=args.get("aliases"))
        if kind == "catalog_years":
            return years(args["year_from"], args["year_to"], media_type, source)
        return {"error": f"unsupported constraint: {kind}"}

    if kind == "catalog_keyword" and args.get("_keyword_ids"):
        ids = set()
        for keyword_id in args["_keyword_ids"]:
            page = 1
            while page <= 100:
                result = seerr.discover(media_type, page=page, keyword_ids=[keyword_id])
                if "error" in result:
                    return {"error": result["error"]}
                ids.update(item["id"] for item in result.get("results", []) if item.get("id"))
                total_pages = min(int(result.get("totalPages") or result.get("total_pages") or 1), 100)
                if page >= total_pages:
                    break
                page += 1
        return _store(ids, media_type, f"keyword:{args['name']}")
    if seed_ids is not None:
        label = kind.replace("catalog_", "") + ":seed"
        return _store(seed_ids, media_type, label)
    if kind == "catalog_person":
        return person(args["name"], media_type)
    if kind == "catalog_genre":
        return genre(args["name"], media_type)
    if kind == "catalog_keyword":
        return keyword(args["name"], media_type, aliases=args.get("aliases"))
    if kind == "catalog_years":
        return years(args["year_from"], args["year_to"], media_type)
    return {"error": f"unsupported constraint: {kind}"}


def execute_constraint_group(entries, source=None):
    """Choose the cheapest seed, or refine an existing candidate set."""
    if not entries:
        return {"error": "constraint group is empty"}
    media_types = {args.get("media_type") for _, args in entries}
    if len(media_types) != 1:
        return {"error": "constraint group has different media types"}

    # A supplied source is already the cheapest possible search space. Do not
    # run global Discover estimates: refine that request-local subset directly.
    if source is not None:
        try:
            src = _get(source)
        except ValueError as exc:
            return {"error": str(exc)}
        if src["media_type"] not in media_types:
            return {"error": "source handle has different media type"}
        current = {"set": source, "count": len(src["ids"])}
        _trace("refine_group_start", source=source, source_count=current["count"], constraints=len(entries))
        for kind, args in entries:
            before = current["count"]
            started = time.perf_counter()
            current = materialize_constraint(kind, args, source=current["set"])
            elapsed = time.perf_counter() - started
            _trace("refine", kind=kind, args=args, before=before, after=current.get("count"), wall_s=round(elapsed, 3), error=current.get("error"))
            if current.get("error"):
                return current
            if current.get("count") == 0:
                break
        return current

    estimates = []
    for index, (kind, args) in enumerate(entries):
        started = time.perf_counter()
        estimate = estimate_constraint(kind, args)
        elapsed = time.perf_counter() - started
        _trace("estimate", kind=kind, args=args, count=estimate.get("count"), seed_ids=len(estimate.get("seed_ids", [])) if estimate.get("seed_ids") is not None else None, wall_s=round(elapsed, 3), error=estimate.get("error"))
        if estimate.get("error"):
            return estimate
        estimates.append((estimate["count"], index, kind, args, estimate))

    _, seed_index, seed_kind, seed_args, seed_estimate = min(
        estimates, key=lambda x: (x[0], x[1])
    )
    started = time.perf_counter()
    seed_materialize_args = dict(seed_args)
    if seed_estimate.get("keyword_ids"):
        seed_materialize_args["_keyword_ids"] = seed_estimate["keyword_ids"]
    current = materialize_constraint(
        seed_kind, seed_materialize_args, seed_ids=seed_estimate.get("seed_ids")
    )
    _trace("seed", kind=seed_kind, args=seed_args, estimated_count=seed_estimate["count"], actual_count=current.get("count"), set=current.get("set"), wall_s=round(time.perf_counter() - started, 3), error=current.get("error"))
    if current.get("error"):
        return current

    # Refine in ascending estimated cardinality. This changes cost only, never
    # boolean semantics: every remaining constraint is still ANDed.
    for _, index, kind, args, _ in sorted(estimates, key=lambda x: (x[0], x[1])):
        if index == seed_index:
            continue
        before = current["count"]
        started = time.perf_counter()
        current = materialize_constraint(kind, args, source=current["set"])
        _trace("refine", kind=kind, args=args, before=before, after=current.get("count"), wall_s=round(time.perf_counter() - started, 3), error=current.get("error"))
        if current.get("error"):
            return current
        if current.get("count") == 0:
            break
    return current

def combine(operation, sets):
    if not sets:
        return {"error": "sets must not be empty"}
    try:
        values, media_type = _same_type(sets)
    except ValueError as exc:
        return {"error": str(exc)}

    if operation == "intersection":
        # Start from the smallest materialized subset. This is purely a
        # cardinality optimization; Python assigns no semantic priority.
        ordered = sorted(values, key=lambda value: len(value["ids"]))
        ids = ordered[0]["ids"].copy()
        for value in ordered[1:]:
            ids &= value["ids"]
    elif operation == "union":
        ids = set()
        for value in values:
            ids |= value["ids"]
    else:
        return {"error": "operation must be intersection or union"}
    return _store(ids, media_type, operation)


def subtract(source, remove):
    try:
        src = _get(source)
        rem = _get(remove)
    except ValueError as exc:
        return {"error": str(exc)}
    if src["media_type"] != rem["media_type"]:
        return {"error": "set handles have different media types"}
    return _store(src["ids"] - rem["ids"], src["media_type"], "difference")


def results(handle, limit=10):
    total_started = time.perf_counter()
    try:
        value = _get(handle)
    except ValueError as exc:
        return {"error": str(exc)}

    ids = list(value["ids"])
    details = []
    _trace("results_start", set=handle, candidate_count=len(ids), limit=limit)

    def inspect(media_id):
        try:
            item = seerr.media_details(media_id, value["media_type"])
            return None if "error" in item else item
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(inspect, media_id) for media_id in ids]
        for future in as_completed(futures):
            item = future.result()
            if item:
                details.append(item)

    details.sort(
        key=lambda x: (x.get("rating") or 0, x.get("voteCount") or 0),
        reverse=True,
    )
    # Keep the LLM context compact, but include the catalogue synopsis so the
    # final natural-language answer can describe results without model-memory
    # fabrication. Semantic filtering fields stay internal.
    compact = []
    for item in details[:max(1, min(int(limit), 20))]:
        compact.append({
            "mediaType": item.get("mediaType"),
            "id": item.get("id"),
            "title": item.get("title"),
            "releaseDate": item.get("releaseDate"),
            "rating": item.get("rating"),
            "voteCount": item.get("voteCount"),
            "overview": item.get("overview"),
        })
    result = {
        "set": handle,
        "count": len(details),
        "results": compact,
    }
    _trace("results_done", set=handle, candidate_count=len(ids), detail_count=len(details), returned_count=len(compact), wall_s=round(time.perf_counter() - total_started, 3))
    return result


def _tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


TITLE_TOOL = _tool(
    "catalog_title",
    "Resolve a media explicitly identified or named by the user. Use this when the user refers to a specific movie or TV series by title, including when they want to add, download, check, watch, or re-request it. The title is an opaque target: NEVER reinterpret words inside a named title as person, genre, theme, keyword, or year constraints. Use discovery constraints only when the user describes media they have not identified by title.",
    {
        "query": {"type": "string", "description": "The media title as named by the user; preserve the title wording rather than decomposing its words."},
        "media_type": {"type": "string", "enum": ["movie", "tv"]},
    },
    ["query", "media_type"],
)

PERSON_TOOL = _tool(
    "catalog_person",
    "Declare one person constraint for a movie or TV search. This is declarative only: Python records it without scanning the catalogue. Declare every constraint from the user request, then call catalog_execute once.",
    {
        "name": {"type": "string"},
        "media_type": {"type": "string", "enum": ["movie", "tv"]},
        "group": {"type": "integer", "minimum": 0, "description": "AND group number. Same group means simultaneous constraints; different groups mean explicit OR alternatives. Default 0."},
        "exclude": {"type": "boolean", "description": "True only when explicitly excluded. Separate excluded constraints are OR by default; give them the same explicit group only when the user excludes their conjunction. Default false."},
    },
    ["name", "media_type"],
)

GENRE_TOOL = _tool(
    "catalog_genre",
    "Declare one movie or TV genre constraint. Use the canonical catalogue genre name, normally English (for example Science Fiction). This is declarative only: Python records it without scanning the catalogue. Declare every constraint from the user request, then call catalog_execute once.",
    {
        "name": {"type": "string"},
        "media_type": {"type": "string", "enum": ["movie", "tv"]},
        "source": {"type": "string", "description": "Optional existing candidate-set handle to refine."},
        "group": {"type": "integer", "minimum": 0, "description": "AND group number. Use the same group for simultaneous constraints; different groups only for explicit OR alternatives. Default 0."},
        "exclude": {"type": "boolean", "description": "True only when explicitly excluded. Separate excluded constraints are OR by default; give them the same explicit group only when the user excludes their conjunction. Default false."},
    },
    ["name", "media_type"],
)

KEYWORD_VOCABULARY_TOOL = _tool(
    "catalog_keyword_vocabulary",
    "Look up real Seerr/TMDB keyword labels for a theme or concept before declaring catalog_keyword when the catalogue wording is uncertain. This tool returns vocabulary only; choose semantically appropriate returned labels yourself. Do not treat every returned label as equivalent.",
    {
        "query": {"type": "string", "description": "Short English concept or lexical variant to look up in the catalogue keyword vocabulary."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    ["query"],
)

KEYWORD_TOOL = _tool(
    "catalog_keyword",
    "Declare one movie or TV theme/concept constraint using real catalogue keyword labels. If the catalogue wording for a semantic concept is uncertain, call catalog_keyword_vocabulary first and then use the semantically appropriate returned label(s). aliases are OR-equivalent labels for the same requested concept. This is declarative only: Python records it without scanning the catalogue. Declare every constraint from the user request, then call catalog_execute once.",
    {
        "name": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}, "maxItems": 5, "description": "Optional English catalogue keyword labels that express the same requested concept. They are OR alternatives, never additional constraints. Use them when a concept can be represented by several catalogue labels."},
        "media_type": {"type": "string", "enum": ["movie", "tv"]},
        "source": {"type": "string", "description": "Optional existing candidate-set handle to refine."},
        "group": {"type": "integer", "minimum": 0, "description": "AND group number. Use the same group for simultaneous constraints; different groups only for explicit OR alternatives. Default 0."},
        "exclude": {"type": "boolean", "description": "True only when explicitly excluded. Separate excluded constraints are OR by default; give them the same explicit group only when the user excludes their conjunction. Default false."},
    },
    ["name", "media_type"],
)

YEARS_TOOL = _tool(
    "catalog_years",
    "Declare one inclusive release-year constraint. This is declarative only: Python records it without scanning the catalogue. Declare every constraint from the user request, then call catalog_execute once.",
    {
        "year_from": {"type": "integer"},
        "year_to": {"type": "integer"},
        "media_type": {"type": "string", "enum": ["movie", "tv"]},
        "source": {"type": "string", "description": "Optional existing candidate-set handle to refine."},
        "group": {"type": "integer", "minimum": 0, "description": "AND group number. Use the same group for simultaneous constraints; different groups only for explicit OR alternatives. Default 0."},
        "exclude": {"type": "boolean", "description": "True only when explicitly excluded. Separate excluded constraints are OR by default; give them the same explicit group only when the user excludes their conjunction. Default false."},
    },
    ["year_from", "year_to", "media_type"],
)

COMBINE_TOOL = _tool(
    "catalog_combine",
    "Combine candidate-set handles. intersection means all constraints; union means alternatives.",
    {
        "operation": {"type": "string", "enum": ["intersection", "union"]},
        "sets": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    ["operation", "sets"],
)

SUBTRACT_TOOL = _tool(
    "catalog_subtract",
    "Remove every item in one candidate set from another candidate set.",
    {
        "source": {"type": "string"},
        "remove": {"type": "string"},
    },
    ["source", "remove"],
)

EXECUTE_TOOL = _tool(
    "catalog_execute",
    "Execute all catalogue constraints declared for the current user request. Call exactly once after every include/exclude constraint has been declared and any uncertain keyword has been grounded. Python chooses the cheapest subset and performs all set algebra; do not combine or subtract catalogue sets yourself. You MUST classify the user's requested outcome: purpose=results when they only want to see/find/list catalogue results; purpose=action when the catalogue results are inputs to any further operation such as checking another service, adding, requesting, downloading, or modifying media.",
    {
        "purpose": {
            "type": "string",
            "enum": ["results", "action"],
            "description": "Required outcome classification. action means at least one further service/tool operation is needed after catalogue discovery."
        }
    },
    ["purpose"],
)

RESULTS_TOOL = _tool(
    "catalog_results",
    "Return the best-rated media in a candidate set. Call after set operations are complete.",
    {
        "set": {"type": "string"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 20},
    },
    ["set"],
)

TOOLS = [
    TITLE_TOOL, PERSON_TOOL, GENRE_TOOL, KEYWORD_VOCABULARY_TOOL, KEYWORD_TOOL, YEARS_TOOL,
    EXECUTE_TOOL,
]
