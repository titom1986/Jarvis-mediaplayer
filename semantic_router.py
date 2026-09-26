"""Semantic request-family router.

The router classifies meaning, never phrases or media identities.  It intentionally
has no lexical rules: Qwen sees four broad capabilities and selects one.  Low
confidence/no tool call falls back to the full agent toolset.
"""
import requests

from config import OLLAMA_URL, MODEL

ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "route_media_request",
        "description": (
            "Classify the user's media request by semantic goal. "
            "named_action: act on a specific movie/series/title the user names, including a season/episode of it. "
            "discovery: find/select media from descriptive constraints such as actor, genre, theme, era, or other criteria, "
            "including when the selected result will later be downloaded. "
            "status: ask whether named media exists/is available in Plex, Radarr or Sonarr. "
            "download_status: ask about progress/state of an already requested download."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "enum": ["named_action", "discovery", "status", "download_status"],
                }
            },
            "required": ["family"],
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


def route(question):
    """Return a semantic family, or None so the caller can safely fall back."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": question}],
        "tools": [ROUTE_TOOL],
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
    if fn.get("name") != "route_media_request":
        return None
    family = (fn.get("arguments") or {}).get("family")
    return family if family in FAMILY_TOOL_NAMES else None


def tools_for_family(all_tools, family):
    names = FAMILY_TOOL_NAMES.get(family)
    if not names:
        return all_tools
    selected = [tool for tool in all_tools if tool["function"]["name"] in names]
    return selected or all_tools
