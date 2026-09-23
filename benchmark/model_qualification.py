import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:11435"
CHAT = BASE + "/api/chat"

# Only documented/native runtime differences. Semantic prompts and tools are
# identical for every model.
PROFILES = {
    "qwen3": {"think": False},
    "granite3.3": {},
    "phi4-mini": {},
    "ministral-3": {},
}

GROUP = {
    "type": "object",
    "properties": {
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
            },
        },
    },
}

MEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": (
            "Recherche des films ou séries selon des groupes de critères. "
            "Toutes les contraintes people, genres, keywords et dates d'un même groupe "
            "sont simultanées (AND). Les groupes sont des alternatives (OR). "
            "Crée un nouveau groupe uniquement si la demande exprime une alternative. "
            "include et exclude ont exactement la même structure. "
            "exclude porte déjà la négation: les valeurs y sont toujours positives. "
            "people contient les personnes, genres les genres de catalogue, keywords "
            "les thèmes/concepts et dates les périodes. N'ajoute aucun critère non demandé."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "media_type": {"type": "string", "enum": ["movie", "tv"]},
                "include": {"type": "array", "items": GROUP},
                "exclude": {"type": "array", "items": GROUP},
            },
            "required": ["media_type", "include"],
        },
    },
}

PLEX_TOOL = {
    "type": "function",
    "function": {
        "name": "plex_status",
        "description": "Retourne si un titre exact est présent dans Plex et s'il a été vu.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
}

# Qualification set: natural French, varied entities and structures. The held-out
# Bruce Willis / 90s / not 98 / time-travel / Scarlett Johansson sentence is
# intentionally NOT present here; it remains reserved for final E2E validation.
CASES = [
    {
        "name": "Q1_simple_genre",
        "prompt": "Trouve-moi des films d'horreur.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {"media_type": "movie", "include": [{"genres": ["Horror"]}]}),
    },
    {
        "name": "Q2_person_genre_period_and",
        "prompt": "Je cherche un thriller avec Morgan Freeman sorti entre 1995 et 2005.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {
            "media_type": "movie",
            "include": [{"people": ["Morgan Freeman"], "genres": ["Thriller"], "dates": [{"from": 1995, "to": 2005}]}],
        }),
    },
    {
        "name": "Q3_positive_keyword",
        "prompt": "Je voudrais une série de science-fiction sur les voyages spatiaux.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {
            "media_type": "tv",
            "include": [{"genres": ["Science Fiction"], "keywords": ["space travel"]}],
        }),
        "semantic_aliases": {"space travel": ["space travel", "space exploration", "spaceflight"]},
    },
    {
        "name": "Q4_exclusion",
        "prompt": "Un film d'action des années 2010, mais sans zombies.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {
            "media_type": "movie",
            "include": [{"genres": ["Action"], "dates": [{"from": 2010, "to": 2019}]}],
            "exclude": [{"keywords": ["zombie"]}],
        }),
        "semantic_aliases": {"zombie": ["zombie", "zombies"]},
    },
    {
        "name": "Q5_or_people",
        "prompt": "Trouve un film avec Tom Hanks ou Denzel Washington.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {
            "media_type": "movie",
            "include": [{"people": ["Tom Hanks"]}, {"people": ["Denzel Washington"]}],
        }),
    },
    {
        "name": "Q6_exclude_person",
        "prompt": "Je veux une comédie avec Adam Sandler mais sans Jennifer Aniston.",
        "tools": [MEDIA_TOOL],
        "expect": ("seerr_media_search", {
            "media_type": "movie",
            "include": [{"people": ["Adam Sandler"], "genres": ["Comedy"]}],
            "exclude": [{"people": ["Jennifer Aniston"]}],
        }),
    },
    {
        "name": "Q7_tool_choice",
        "prompt": "Est-ce que Interstellar est dans mon Plex ?",
        "tools": [MEDIA_TOOL, PLEX_TOOL],
        "expect": ("plex_status", {"title": "Interstellar"}),
    },
]

# One round-trip case verifies actual agent mechanics independently of media
# filtering: tool selection -> tool result consumption -> grounded final answer.
ROUNDTRIP = {
    "name": "Q8_roundtrip",
    "prompt": "Est-ce que Blade Runner 2049 est dans mon Plex et est-ce que je l'ai déjà vu ?",
    "tools": [MEDIA_TOOL, PLEX_TOOL],
    "expect": ("plex_status", {"title": "Blade Runner 2049"}),
    "result": {"found": True, "title": "Blade Runner 2049", "viewCount": 0, "watched": False},
}


def profile(model):
    for prefix, p in PROFILES.items():
        if model.startswith(prefix):
            return p
    return {}


def post(payload, timeout=300):
    req = urllib.request.Request(
        CHAT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data, round(time.perf_counter() - t, 3), None
    except urllib.error.HTTPError as e:
        return None, round(time.perf_counter() - t, 3), f"HTTP {e.code}: {e.read().decode(errors='replace')}"
    except Exception as e:
        return None, round(time.perf_counter() - t, 3), repr(e)


def args_of(call):
    raw = ((call or {}).get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def normalize(v):
    if isinstance(v, str):
        return v.strip().casefold()
    if isinstance(v, list):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items()}
    return v


def semantic_equal(actual, expected, aliases=None):
    # Structure remains strict. Only explicitly declared vocabulary aliases are
    # accepted, because catalogue labels can have equivalent wording.
    a, e = normalize(actual), normalize(expected)
    aliases = aliases or {}
    for canonical, variants in aliases.items():
        allowed = {x.casefold() for x in variants}
        def canon(x):
            if isinstance(x, str) and x in allowed:
                return canonical.casefold()
            if isinstance(x, list):
                return [canon(y) for y in x]
            if isinstance(x, dict):
                return {k: canon(y) for k, y in x.items()}
            return x
        a, e = canon(a), canon(e)
    return a == e


def first_call_result(model, case):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": case["prompt"]}],
        "tools": case["tools"],
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 180},
    }
    payload.update(profile(model))
    raw, wall, err = post(payload)
    out = {"wall_s": wall, "error": err, "raw": raw}
    if err or not raw:
        return out | {"ok": False, "failure": "transport"}
    msg = raw.get("message") or {}
    calls = msg.get("tool_calls") or []
    out["done_reason"] = raw.get("done_reason")
    out["eval_count"] = raw.get("eval_count")
    if len(calls) != 1:
        return out | {"ok": False, "failure": "no_or_multiple_tool_calls", "content": msg.get("content")}
    name = (calls[0].get("function") or {}).get("name")
    args = args_of(calls[0])
    exp_name, exp_args = case["expect"]
    out["actual_name"] = name
    out["actual_args"] = args
    if name != exp_name:
        return out | {"ok": False, "failure": "wrong_tool"}
    if not semantic_equal(args, exp_args, case.get("semantic_aliases")):
        return out | {"ok": False, "failure": "wrong_arguments"}
    return out | {"ok": True, "failure": None, "assistant_message": msg}


def run_roundtrip(model):
    case = ROUNDTRIP
    first = first_call_result(model, case)
    if not first["ok"]:
        first.pop("assistant_message", None)
        return {"first": first, "roundtrip_ok": False}

    assistant = first.pop("assistant_message")
    messages = [
        {"role": "user", "content": case["prompt"]},
        assistant,
        {"role": "tool", "tool_name": "plex_status", "content": json.dumps(case["result"])},
    ]
    payload = {
        "model": model, "messages": messages, "tools": case["tools"],
        "stream": False, "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 180},
    }
    payload.update(profile(model))
    raw, wall, err = post(payload)
    final = {"wall_s": wall, "error": err, "raw": raw}
    if err or not raw:
        return {"first": first, "second": final, "roundtrip_ok": False}
    msg = raw.get("message") or {}
    text = (msg.get("content") or "").casefold()
    calls = msg.get("tool_calls") or []
    grounded = (
        not calls
        and "blade runner 2049" in text
        and any(x in text for x in ["pas vu", "pas encore vu", "non vu", "not watched", "haven't watched", "have not watched"])
    )
    final["grounded_final"] = grounded
    return {"first": first, "second": final, "roundtrip_ok": grounded}


def main():
    models = sys.argv[1:] or ["qwen3:1.7b", "granite3.3:2b", "phi4-mini:3.8b", "ministral-3:3b"]
    report = {
        "purpose": "fair native model qualification before held-out E2E",
        "held_out_e2e_used": False,
        "rules": {
            "same_semantic_tasks": True,
            "same_tool_contracts": True,
            "custom_system_prompt": False,
            "only_documented_native_runtime_differences": True,
            "temperature": 0,
            "real_services": False,
        },
        "models": [],
    }
    for model in models:
        entry = {"model": model, "cases": []}
        for case in CASES:
            r = first_call_result(model, case)
            r.pop("assistant_message", None)
            entry["cases"].append({"name": case["name"], **r})
        entry["roundtrip"] = run_roundtrip(model)
        entry["passed_first_turn"] = sum(1 for x in entry["cases"] if x["ok"])
        entry["total_first_turn"] = len(entry["cases"])
        report["models"].append(entry)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    with open("model_qualification_result.json", "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
