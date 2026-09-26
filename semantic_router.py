"""Hierarchical semantic media router.

Qwen is deliberately asked small semantic questions. The first stage identifies
what kind of reference the user made; only named-media requests need a second
decision between action and status. No title, verb, or phrase is hard-coded.
"""
import requests

from config import OLLAMA_URL, MODEL

REFERENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "route_media_reference",
        "description": (
            "Classify what the user's request is ABOUT. "
            "named_media: one specific movie, series, season or episode is identified by title/name. "
            "discovery: media must be found or selected from descriptive criteria such as person, genre, theme, era or properties. "
            "transfer: the request asks about progress/state/completion of a download already in progress or previously requested."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["named_media", "discovery", "transfer"]}
            },
            "required": ["kind"],
            "additionalProperties": False,
        },
    },
}

NAMED_INTENT_TOOL = {
    "type": "function",
    "function": {
        "name": "route_named_media_intent",
        "description": (
            "The request is already known to concern specific named media. "
            "Classify only the user's intended operation. "
            "action: the user wants the named media obtained, added, requested, downloaded, re-downloaded, or an episode/season fetched. "
            "status: the user only asks whether the named media is present, available, known, or already in the library/service."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["action", "status"]}
            },
            "required": ["intent"],
            "additionalProperties": False,
        },
    },
}

FAMILY_TOOL_NAMES = {
    "named_action": {
        "catalog_title", "radarr_request_movie", "radarr_request_movies", "sonarr_request_series",
    },
    "discovery": {
        "catalog_person", "catalog_genre", "catalog_keyword_vocabulary", "catalog_keyword",
        "catalog_years", "catalog_execute", "radarr_request_movie", "radarr_request_movies",
        "sonarr_request_series", "plex_status",
    },
    "status": {
        "catalog_title", "plex_status", "radarr_status", "sonarr_status",
    },
    "download_status": {
        "catalog_title", "radarr_queue_status", "sonarr_queue_status",
    },
}


def _classify(question, tool, argument, allowed):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "tools": [tool],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 48},
    }
    if MODEL.casefold().startswith("qwen3"):
        payload["think"] = False
    response = requests.post(OLLAMA_URL, json=payload, timeout=None)
    response.raise_for_status()
    message = response.json().get("message") or {}
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    fn = calls[0].get("function") or {}
    if fn.get("name") != tool["function"]["name"]:
        return None
    value = (fn.get("arguments") or {}).get(argument)
    return value if value in allowed else None


def route(question):
    """Return the final agent family, or None for safe full-tool fallback."""
    kind = _classify(
        question, REFERENCE_TOOL, "kind",
        {"named_media", "discovery", "transfer"},
    )
    if kind == "discovery":
        return "discovery"
    if kind == "transfer":
        return "download_status"
    if kind != "named_media":
        return None

    intent = _classify(
        question, NAMED_INTENT_TOOL, "intent", {"action", "status"}
    )
    if intent == "action":
        return "named_action"
    if intent == "status":
        return "status"
    return None


def tools_for_family(all_tools, family):
    names = FAMILY_TOOL_NAMES.get(family)
    if not names:
        return all_tools
    selected = [tool for tool in all_tools if tool["function"]["name"] in names]
    return selected or all_tools
