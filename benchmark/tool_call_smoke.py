import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:11435"
CHAT = BASE + "/api/chat"
SHOW = BASE + "/api/show"

TOOL = {
    "type": "function",
    "function": {
        "name": "plex_status",
        "description": "Return whether one exact movie title is present in Plex and whether it has been watched.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Exact movie title"}
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}

# Keep the task deliberately trivial. This smoke test measures native Ollama
# tool-call transport/template compatibility, not media reasoning.
USER = "Use the plex_status tool to check whether the movie Armageddon is in Plex and whether I have watched it. Do not guess."

# Model-specific settings are limited to documented native conventions.
# Granite is tested in its normal tool mode and with its documented thinking
# mechanisms so we do not reject it because of one integration convention.
PROFILES = {
    "qwen3": [
        ("native_no_think", {"think": False}, []),
    ],
    "granite3.3": [
        ("native", {}, []),
        ("documented_think", {"think": True}, []),
        ("documented_control_thinking", {}, [{"role": "control", "content": "thinking"}]),
    ],
    "phi4-mini": [
        ("native", {}, []),
    ],
    "ministral-3": [
        ("native", {}, []),
    ],
}


def profile_variants(model):
    for prefix, variants in PROFILES.items():
        if model.startswith(prefix):
            return variants
    return [("native", {}, [])]


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
        body = e.read().decode(errors="replace")
        return None, time.perf_counter() - started, f"HTTP {e.code}: {body}"
    except Exception as e:
        return None, time.perf_counter() - started, repr(e)


def show_model(model):
    data, wall, err = post(SHOW, {"model": model}, timeout=60)
    if err:
        return {"error": err, "wall_s": round(wall, 3)}
    return {
        "wall_s": round(wall, 3),
        "capabilities": data.get("capabilities"),
        "details": data.get("details"),
        "template": data.get("template"),
    }


def tool_calls(message):
    return (message or {}).get("tool_calls") or []


def function_args(tc):
    fn = tc.get("function") or {}
    raw = fn.get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def run_variant(model, variant, extra_payload, prefix_messages):
    base = {
        "model": model,
        "stream": False,
        "keep_alive": "30m",
        "tools": [TOOL],
        "options": {"temperature": 0, "num_predict": 120},
    }
    base.update(extra_payload)

    messages = prefix_messages + [{"role": "user", "content": USER}]
    first_payload = dict(base, messages=messages)
    first, wall1, err1 = post(CHAT, first_payload)

    result = {
        "model": model,
        "variant": variant,
        "first_wall_s": round(wall1, 3),
        "first_error": err1,
        "first_raw": first,
    }
    if err1 or not first:
        result["pass_single_tool"] = False
        result["pass_roundtrip"] = False
        return result

    msg1 = first.get("message") or {}
    calls = tool_calls(msg1)
    valid = []
    for tc in calls:
        fn = tc.get("function") or {}
        valid.append(fn.get("name") == "plex_status" and function_args(tc) == {"title": "Armageddon"})
    result["pass_single_tool"] = len(calls) == 1 and all(valid)

    if not result["pass_single_tool"]:
        result["pass_roundtrip"] = False
        return result

    # Follow Ollama's documented multi-turn pattern: append the complete
    # assistant message, then a tool-role result, then call chat again.
    messages.append(msg1)
    messages.append({
        "role": "tool",
        "tool_name": "plex_status",
        "content": json.dumps({
            "found": True,
            "title": "Armageddon",
            "viewCount": 1,
            "watched": True,
        }),
    })
    second_payload = dict(base, messages=messages)
    second, wall2, err2 = post(CHAT, second_payload)
    result["second_wall_s"] = round(wall2, 3)
    result["second_error"] = err2
    result["second_raw"] = second

    if err2 or not second:
        result["pass_roundtrip"] = False
        return result

    msg2 = second.get("message") or {}
    text = (msg2.get("content") or "").casefold()
    result["pass_roundtrip"] = (
        not tool_calls(msg2)
        and "armageddon" in text
        and any(word in text for word in ("watched", "vu", "regard", "visionn"))
    )
    return result


def main():
    models = sys.argv[1:] or [
        "qwen3:1.7b",
        "granite3.3:2b",
        "phi4-mini:3.8b",
        "ministral-3:3b",
    ]

    report = {
        "purpose": "native Ollama tool-calling compatibility smoke test",
        "models": [],
    }

    for model in models:
        entry = {"model": model, "show": show_model(model), "variants": []}
        for variant, extra_payload, prefix_messages in profile_variants(model):
            entry["variants"].append(
                run_variant(model, variant, extra_payload, prefix_messages)
            )
        report["models"].append(entry)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
