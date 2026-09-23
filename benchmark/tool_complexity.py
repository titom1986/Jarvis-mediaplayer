import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:11435"
CHAT = BASE + "/api/chat"
SHOW = BASE + "/api/show"

# No custom system prompt here. This suite deliberately preserves each
# installed model's native Ollama template/tool instructions.
PROFILES = {
    "qwen3": {"think": False},
    "granite3.3": {},
    "phi4-mini": {},
    "ministral-3": {},
}

GROUP_PROPERTIES = {
    "people": {"type": "array", "items": {"type": "string"}},
    "genres": {"type": "array", "items": {"type": "string"}},
    "keywords": {"type": "array", "items": {"type": "string"}},
    "dates": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "from": {"type": "integer"},
                "to": {"type": "integer"},
            },
            "required": ["from", "to"],
            "additionalProperties": False,
        },
    },
}

MEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": (
            "Search movies or TV shows. Criteria inside one include group are all "
            "required together (AND). Multiple include groups are alternatives (OR). "
            "exclude contains positive properties that must be rejected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "media_type": {"type": "string", "enum": ["movie", "tv"]},
                "include": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": GROUP_PROPERTIES,
                        "additionalProperties": False,
                    },
                },
                "exclude": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": GROUP_PROPERTIES,
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["media_type", "include"],
            "additionalProperties": False,
        },
    },
}

PLEX_TOOL = {
    "type": "function",
    "function": {
        "name": "plex_status",
        "description": "Return presence and watched status for one exact title in Plex.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}

# Escalation isolates the first structural level at which a model fails.
CASES = [
    {
        "name": "L1_one_string",
        "tools": [PLEX_TOOL],
        "prompt": "Check Armageddon in Plex using the available tool. Do not guess.",
        "expected_name": "plex_status",
        "expected_args": {"title": "Armageddon"},
    },
    {
        "name": "L2_scalar_plus_array",
        "tools": [MEDIA_TOOL],
        "prompt": "Search for movies with genre Thriller using the available tool.",
        "expected_name": "seerr_media_search",
        "expected_args": {
            "media_type": "movie",
            "include": [{"genres": ["Thriller"]}],
        },
    },
    {
        "name": "L3_array_object_date",
        "tools": [MEDIA_TOOL],
        "prompt": "Search for thriller movies released from 2020 through 2025 using the available tool.",
        "expected_name": "seerr_media_search",
        "expected_args": {
            "media_type": "movie",
            "include": [{"genres": ["Thriller"], "dates": [{"from": 2020, "to": 2025}]}],
        },
    },
    {
        "name": "L4_and_group",
        "tools": [MEDIA_TOOL],
        "prompt": "Search for 1990 through 1999 science fiction movies with Bruce Willis using the available tool.",
        "expected_name": "seerr_media_search",
        "expected_args": {
            "media_type": "movie",
            "include": [{
                "people": ["Bruce Willis"],
                "genres": ["Science Fiction"],
                "dates": [{"from": 1990, "to": 1999}],
            }],
        },
    },
    {
        "name": "L5_include_exclude",
        "tools": [MEDIA_TOOL],
        "prompt": "Search for 1990 through 1999 science fiction movies with Bruce Willis, excluding dystopia, using the available tool.",
        "expected_name": "seerr_media_search",
        "expected_args": {
            "media_type": "movie",
            "include": [{
                "people": ["Bruce Willis"],
                "genres": ["Science Fiction"],
                "dates": [{"from": 1990, "to": 1999}],
            }],
            "exclude": [{"keywords": ["dystopia"]}],
        },
    },
    {
        "name": "L6_or_groups",
        "tools": [MEDIA_TOOL],
        "prompt": "Search for movies with either Bruce Willis or Brad Pitt using the available tool.",
        "expected_name": "seerr_media_search",
        "expected_args": {
            "media_type": "movie",
            "include": [
                {"people": ["Bruce Willis"]},
                {"people": ["Brad Pitt"]},
            ],
        },
    },
    {
        "name": "L7_choose_tool",
        "tools": [MEDIA_TOOL, PLEX_TOOL],
        "prompt": "Is Armageddon in my Plex library? Use the appropriate available tool. Do not search the movie catalogue.",
        "expected_name": "plex_status",
        "expected_args": {"title": "Armageddon"},
    },
]


def profile(model):
    for prefix, value in PROFILES.items():
        if model.startswith(prefix):
            return value
    return {}


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
        return json.loads(raw), time.perf_counter() - started, None
    except urllib.error.HTTPError as e:
        return None, time.perf_counter() - started, f"HTTP {e.code}: {e.read().decode(errors='replace')}"
    except Exception as e:
        return None, time.perf_counter() - started, repr(e)


def show(model):
    data, wall, err = post(SHOW, {"model": model}, 60)
    if err:
        return {"error": err}
    return {
        "capabilities": data.get("capabilities"),
        "details": data.get("details"),
        "template": data.get("template"),
        "wall_s": round(wall, 3),
    }


def args_of(tc):
    raw = ((tc or {}).get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def strict_equal(actual, expected):
    # Intentional exact structural comparison. Extra/misplaced fields fail.
    return actual == expected


def run_case(model, case):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "tools": case["tools"],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 180},
    }
    payload.update(profile(model))
    raw, wall, err = post(CHAT, payload)
    result = {"wall_s": round(wall, 3), "error": err, "raw": raw}
    if err or not raw:
        result["ok"] = False
        result["failure_class"] = "transport"
        return result

    msg = raw.get("message") or {}
    calls = msg.get("tool_calls") or []
    result["done_reason"] = raw.get("done_reason")
    result["output_tokens"] = raw.get("eval_count")
    if not calls:
        result["ok"] = False
        result["failure_class"] = "protocol_no_tool_call"
        result["content"] = msg.get("content")
        return result
    if len(calls) != 1:
        result["ok"] = False
        result["failure_class"] = "protocol_wrong_call_count"
        return result

    tc = calls[0]
    name = (tc.get("function") or {}).get("name")
    actual = args_of(tc)
    result["actual_name"] = name
    result["actual_args"] = actual
    if name != case["expected_name"]:
        result["ok"] = False
        result["failure_class"] = "wrong_tool"
    elif not strict_equal(actual, case["expected_args"]):
        result["ok"] = False
        result["failure_class"] = "schema_or_semantics"
    else:
        result["ok"] = True
        result["failure_class"] = None
    return result


def main():
    models = sys.argv[1:] or [
        "qwen3:1.7b",
        "granite3.3:2b",
        "phi4-mini:3.8b",
        "ministral-3:3b",
    ]
    report = {
        "purpose": "tool-call schema complexity ladder",
        "rules": {
            "custom_system_prompt": False,
            "temperature": 0,
            "strict_exact_arguments": True,
            "real_services": False,
        },
        "models": [],
    }
    for model in models:
        entry = {"model": model, "show": show(model), "cases": []}
        for case in CASES:
            r = run_case(model, case)
            entry["cases"].append({"name": case["name"], **r})
        report["models"].append(entry)

    out = json.dumps(report, ensure_ascii=False, indent=2)
    print(out)
    # Durable capture even if terminal scrollback/tee is lost.
    with open("tool_complexity_result.json", "w", encoding="utf-8") as f:
        f.write(out + "\n")


if __name__ == "__main__":
    main()
