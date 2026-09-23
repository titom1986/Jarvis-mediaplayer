import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:11435"
CHAT = BASE + "/api/chat"

MODELS = [
    "qwen3:4b-instruct",
    "qwen3:1.7b",
    "granite3.3:2b",
    "phi4-mini:3.8b",
    "ministral-3:3b",
]

# Runtime differences only where the model/runtime documents them. No model gets
# a semantic hint, worked example, or a different tool contract.
def model_payload(model):
    if model.startswith("qwen3"):
        return {"think": False}
    return {}

GROUP = {
    "type": "object",
    "properties": {
        "people": {"type": "array", "items": {"type": "string"}},
        "genres": {"type": "array", "items": {"type": "string"}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "from": {"type": "integer"},
                "to": {"type": "integer"},
            },
        }},
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
            "exclude porte déjà la négation : ses valeurs nomment positivement la propriété à retirer. "
            "people contient les personnes, genres les genres de catalogue, keywords les thèmes/concepts "
            "et dates les périodes. N'ajoute aucun critère non demandé."
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
        "description": "Retourne la présence dans Plex et l'état vu/non vu d'un titre précis.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
        },
    },
}

# Intentionally tiny and generic. Tool schemas carry the API contract; the
# system prompt only defines the agent's role and grounding rule.
SYSTEM = (
    "Tu es JARVIS, l'agent d'un media center. Utilise les outils disponibles "
    "quand ils sont nécessaires pour répondre à la demande. N'invente jamais "
    "les données des services. Réponds dans la langue de l'utilisateur."
)

# FINAL held-out sentence supplied by the user. Do not use it to tune the prompt,
# schemas or model-specific profiles after seeing results.
PROMPT = (
    "Je veux un film de sf de Bruce Willis des années 90 mais pas de 98 "
    "avec du voyage dans le temps mais sans Scarlett Johansson"
)

EXPECTED_SEARCH = {
    "media_type": "movie",
    "include": [{
        "people": ["Bruce Willis"],
        "genres": ["Science Fiction"],
        "keywords": ["time travel"],
        "dates": [{"from": 1990, "to": 1999}],
    }],
    "exclude": [
        {"dates": [{"from": 1998, "to": 1998}]},
        {"people": ["Scarlett Johansson"]},
    ],
}

# Representative output of the deterministic search AFTER all requested filters.
# Multiple plausible matches/distractors force the agent to consume tool data
# rather than simply echoing the only title returned.
SEARCH_RESULT = {
    "mediaType": "movie",
    "count": 3,
    "results": [
        {
            "mediaType": "movie", "id": 63, "title": "Twelve Monkeys",
            "releaseDate": "1995-12-29", "genres": ["Science Fiction", "Thriller"],
            "keywords": ["time travel"], "rating": 7.6, "voteCount": 8700,
        },
        {
            "mediaType": "movie", "id": 18, "title": "Looper",
            "releaseDate": "2012-09-28", "genres": ["Science Fiction", "Thriller"],
            "keywords": ["time travel"], "rating": 6.9, "voteCount": 10300,
        },
        {
            "mediaType": "movie", "id": 95, "title": "Armageddon",
            "releaseDate": "1998-07-01", "genres": ["Science Fiction", "Action"],
            "keywords": ["asteroid"], "rating": 6.8, "voteCount": 7900,
        },
    ],
}

# The real deterministic media_search would have removed Looper/Armageddon.
# They are deliberately retained in the mock payload only as poisoned distractors:
# a correct agent must rely on the tool contract/result semantics and recommend
# the actual matching result, not use memorized film knowledge to choose another.
# This makes hallucination visible without asking Python to do semantic reasoning.
SEARCH_RESULT["results"] = [SEARCH_RESULT["results"][0]]
SEARCH_RESULT["count"] = 1

TOOLS = [MEDIA_TOOL, PLEX_TOOL]


def post(payload, timeout=600):
    req = urllib.request.Request(
        CHAT,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data, round(time.perf_counter() - start, 3), None
    except urllib.error.HTTPError as e:
        return None, round(time.perf_counter() - start, 3), f"HTTP {e.code}: {e.read().decode(errors='replace')}"
    except Exception as e:
        return None, round(time.perf_counter() - start, 3), repr(e)


def arguments(tc):
    raw = ((tc or {}).get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return raw
    return raw


def norm(v):
    if isinstance(v, str):
        return v.strip().casefold()
    if isinstance(v, list):
        return [norm(x) for x in v]
    if isinstance(v, dict):
        return {k: norm(x) for k, x in v.items()}
    return v


def search_semantics(args):
    """Strict structure, tolerant only to catalogue vocabulary spelling."""
    if not isinstance(args, dict) or args.get("media_type") != "movie":
        return False, "media_type"
    inc, exc = args.get("include"), args.get("exclude")
    if not isinstance(inc, list) or len(inc) != 1 or not isinstance(inc[0], dict):
        return False, "include_shape"
    if not isinstance(exc, list) or not all(isinstance(g, dict) for g in exc):
        return False, "exclude_shape"

    g = norm(inc[0])
    if g.get("people") != ["bruce willis"]:
        return False, "include_people"
    if g.get("genres") not in (["science fiction"], ["sci-fi"], ["sf"]):
        return False, "include_genre"
    if g.get("keywords") not in (["time travel"], ["time-travel"], ["voyage dans le temps"]):
        return False, "include_keyword"
    if g.get("dates") != [{"from": 1990, "to": 1999}]:
        return False, "include_dates"

    has_1998 = any(norm(x.get("dates")) == [{"from": 1998, "to": 1998}] for x in exc)
    has_scarlett = any(norm(x.get("people")) == ["scarlett johansson"] for x in exc)
    if not has_1998:
        return False, "exclude_1998"
    if not has_scarlett:
        return False, "exclude_scarlett"
    # Exclusion criteria are OR groups. Putting both in one group would mean
    # "exclude only 1998 Scarlett films", which is not the user's request.
    if any("dates" in x and "people" in x for x in exc):
        return False, "exclude_wrong_and"
    return True, None


def execute(tc):
    fn = ((tc or {}).get("function") or {}).get("name")
    args = arguments(tc)
    if fn == "seerr_media_search":
        ok, why = search_semantics(args)
        if not ok:
            return {"error": "invalid search arguments", "detail": why}
        return SEARCH_RESULT
    if fn == "plex_status":
        title = args.get("title") if isinstance(args, dict) else None
        # Plex is available to the agent but was not requested by this held-out
        # prompt. Calling it is not fatal, but it is unnecessary work.
        return {"found": False, "title": title}
    return {"error": "unknown tool"}


def run(model):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT},
    ]
    trace = []
    total_wall = 0.0
    total_eval = 0.0
    total_tokens = 0

    for turn in range(1, 7):
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 240},
        }
        payload.update(model_payload(model))
        data, wall, err = post(payload)
        total_wall += wall
        if err or not data:
            trace.append({"turn": turn, "wall_s": wall, "error": err})
            break

        total_tokens += data.get("eval_count") or 0
        total_eval += (data.get("eval_duration") or 0) / 1e9
        msg = data.get("message") or {}
        calls = msg.get("tool_calls") or []
        row = {
            "turn": turn, "wall_s": wall, "done_reason": data.get("done_reason"),
            "eval_count": data.get("eval_count"), "content": msg.get("content") or "",
            "tool_calls": [],
        }
        messages.append(msg)

        if not calls:
            trace.append(row)
            break

        for tc in calls:
            fn = ((tc or {}).get("function") or {}).get("name")
            args = arguments(tc)
            result = execute(tc)
            row["tool_calls"].append({"name": fn, "arguments": args, "result": result})
            messages.append({
                "role": "tool",
                "tool_name": fn,
                "content": json.dumps(result, ensure_ascii=False),
            })
        trace.append(row)

    final = trace[-1].get("content", "") if trace else ""
    media_calls = [
        c for t in trace for c in t.get("tool_calls", [])
        if c.get("name") == "seerr_media_search"
    ]
    plex_calls = [
        c for t in trace for c in t.get("tool_calls", [])
        if c.get("name") == "plex_status"
    ]
    search_ok = False
    search_failure = "missing_search"
    if media_calls:
        search_ok, search_failure = search_semantics(media_calls[0].get("arguments"))

    f = final.casefold()
    final_ok = "twelve monkeys" in f or "12 monkeys" in f or "12 singes" in f
    grounded = final_ok and "armageddon" not in f and "looper" not in f

    return {
        "model": model,
        "search_ok": search_ok,
        "search_failure": search_failure,
        "final_ok": final_ok,
        "grounded": grounded,
        "unnecessary_plex_calls": len(plex_calls),
        "turns": len(trace),
        "wall_s": round(total_wall, 3),
        "output_tokens": total_tokens,
        "tok_s": round(total_tokens / total_eval, 2) if total_eval else 0,
        "pass": search_ok and grounded,
        "final": final,
        "trace": trace,
    }


def main():
    models = sys.argv[1:] or MODELS
    report = {
        "purpose": "final held-out generic E2E model selection",
        "prompt": PROMPT,
        "rules": {
            "same_system_semantics": True,
            "same_tools": True,
            "same_tool_results": True,
            "temperature": 0,
            "model_specific_semantic_hints": False,
            "qwen_native_think_disabled": True,
        },
        "models": [],
    }

    for model in models:
        print(f"=== {model} ===", flush=True)
        r = run(model)
        report["models"].append(r)
        print(
            f"pass={r['pass']} search={r['search_ok']} grounded={r['grounded']} "
            f"turns={r['turns']} wall={r['wall_s']}s tok/s={r['tok_s']}",
            flush=True,
        )
        if not r["pass"]:
            print(f"failure={r['search_failure']} final={r['final']!r}", flush=True)

    with open("final_e2e_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("Saved final_e2e_result.json", flush=True)


if __name__ == "__main__":
    main()
