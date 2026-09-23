import json
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11435/api/chat"

MEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": "Recherche le catalogue Seerr. include est une liste de groupes alternatifs (OR). Dans un groupe, people, genres, keywords et dates sont des contraintes cumulatives (AND). people contient des noms de personnes, genres des genres, keywords des thèmes/concepts. exclude contient des groupes à rejeter.",
        "parameters": {
            "type": "object",
            "properties": {
                "media_type": {"type": "string", "enum": ["movie", "tv"]},
                "include": {"type": "array", "items": {"type": "object", "properties": {
                    "people": {"type": "array", "items": {"type": "string"}},
                    "genres": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "dates": {"type": "array", "items": {"type": "object", "properties": {
                        "from": {"type": "integer"}, "to": {"type": "integer"}}}}
                }}},
                "exclude": {"type": "array", "items": {"type": "object", "properties": {
                    "people": {"type": "array", "items": {"type": "string"}},
                    "genres": {"type": "array", "items": {"type": "string"}},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "dates": {"type": "array", "items": {"type": "object", "properties": {
                        "from": {"type": "integer"}, "to": {"type": "integer"}}}}
                }}}
            },
            "required": ["media_type", "include"]
        }
    }
}

PLEX_TOOL = {
    "type": "function",
    "function": {
        "name": "plex_status",
        "description": "Retourne la présence dans Plex et l'état vu/non vu d'un titre précis.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    }
}

SYSTEM = """Tu es JARVIS, l'agent d'un media center. Utilise les outils disponibles quand ils sont nécessaires pour répondre à la demande. N'invente pas les données des services. Réponds dans la langue de l'utilisateur."""

MODEL_PROFILES = {"qwen3": {"think": False}, "granite3.3": {}, "phi4-mini": {}, "ministral-3": {}}

# Mock intentionally contains plausible distractors and enough candidates to force a real choice.
SEARCH_RESULT = {
    "count": 5,
    "results": [
        {"title": "Armageddon", "year": 1998, "rating": 6.83, "voteCount": 8844, "genres": ["Science Fiction", "Action"], "people": ["Bruce Willis"], "keywords": ["asteroid", "space mission"]},
        {"title": "The Fifth Element", "year": 1997, "rating": 7.55, "voteCount": 11200, "genres": ["Science Fiction", "Action"], "people": ["Bruce Willis"], "keywords": ["dystopia", "future"]},
        {"title": "12 Monkeys", "year": 1995, "rating": 7.60, "voteCount": 8500, "genres": ["Science Fiction", "Thriller"], "people": ["Bruce Willis"], "keywords": ["dystopia", "time travel"]},
        {"title": "The Jackal", "year": 1997, "rating": 6.40, "voteCount": 1800, "genres": ["Action", "Thriller"], "people": ["Bruce Willis"], "keywords": ["assassin"]},
        {"title": "Die Hard 2", "year": 1990, "rating": 7.00, "voteCount": 5800, "genres": ["Action"], "people": ["Bruce Willis"], "keywords": ["airport"]}
    ]
}

PLEX = {
    "Armageddon": {"found": True, "title": "Armageddon", "viewCount": 1, "watched": True},
    "The Fifth Element": {"found": True, "title": "The Fifth Element", "viewCount": 0, "watched": False},
    "12 Monkeys": {"found": False, "title": "12 Monkeys"},
    "The Jackal": {"found": False, "title": "The Jackal"},
    "Die Hard 2": {"found": False, "title": "Die Hard 2"},
}

CASES = [
    {
        "name": "recommend_avoid_watched",
        "prompt": "Je voudrais un film de science-fiction avec Bruce Willis des années 90 que tu me conseillerais basé sur sa note. Je ne veux pas de film dystopique et, s'il est présent dans Plex, évite ceux que j'ai déjà vus.",
        "max_turns": 6,
    },
    {
        "name": "plex_direct",
        "prompt": "Est-ce que j'ai Armageddon dans Plex et est-ce que je l'ai déjà vu ?",
        "max_turns": 4,
    },
]

def profile(model):
    for prefix, value in MODEL_PROFILES.items():
        if model.startswith(prefix):
            return value
    return {}

def call(model, messages):
    payload = {"model": model, "messages": messages, "tools": [MEDIA_TOOL, PLEX_TOOL],
               "stream": False, "keep_alive": "30m", "options": {"temperature": 0, "num_predict": 200}}
    payload.update(profile(model))
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data, time.perf_counter() - started

def arguments(call):
    raw = (call.get("function") or {}).get("arguments")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    return {}

def execute(call):
    fn = (call.get("function") or {}).get("name")
    args = arguments(call)
    if fn == "seerr_media_search":
        return SEARCH_RESULT
    if fn == "plex_status":
        return PLEX.get(args.get("title"), {"found": False, "title": args.get("title")})
    return {"error": "unknown tool"}

def tool_message(call, result):
    # Ollama accepts the generic tool role; tool_name is useful across templates.
    return {"role": "tool", "tool_name": (call.get("function") or {}).get("name"),
            "content": json.dumps(result, ensure_ascii=False)}

def run_case(model, case):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}]
    trace, total_wall, total_out, total_eval = [], 0.0, 0, 0.0
    final = ""
    for turn in range(1, case["max_turns"] + 1):
        data, wall = call(model, messages)
        total_wall += wall
        total_out += data.get("eval_count") or 0
        total_eval += (data.get("eval_duration") or 0) / 1e9
        msg = data.get("message") or {}
        calls = msg.get("tool_calls") or []
        trace.append({"turn": turn, "wall_s": round(wall, 3), "content": msg.get("content", ""), "tool_calls": calls})
        messages.append(msg)
        if not calls:
            final = msg.get("content") or ""
            break
        for tc in calls:
            result = execute(tc)
            trace[-1].setdefault("tool_results", []).append(result)
            messages.append(tool_message(tc, result))
    return {"final": final, "trace": trace, "wall_s": total_wall, "tokens": total_out,
            "tok_s": (total_out / total_eval if total_eval else 0)}

def norm(s):
    return str(s or "").casefold()

def score(case, result):
    trace = result["trace"]
    names = [(tc.get("function") or {}).get("name") for t in trace for tc in t.get("tool_calls", [])]
    final = norm(result["final"])
    if case["name"] == "plex_direct":
        ok = names and names[0] == "plex_status" and "armageddon" in final and any(x in final for x in ["vu", "visionn", "déjà"])
        return ok, {"tools": names, "final": result["final"]}
    # Search must happen, and the final answer must honor both exclusion and watched-state.
    searched = "seerr_media_search" in names
    checked_armageddon = False
    for t in trace:
        for tc in t.get("tool_calls", []):
            if (tc.get("function") or {}).get("name") == "plex_status":
                try:
                    checked_armageddon |= norm(arguments(tc).get("title")) == "armageddon"
                except Exception:
                    pass
    # Given the mock: dystopian titles are excluded; Armageddon is the sole valid SF candidate,
    # but it is watched. A correct agent should explain that no unwatched matching recommendation remains.
    acknowledges_constraint = any(x in final for x in ["aucun", "pas de", "déjà vu", "déjà regard"])
    ok = searched and checked_armageddon and acknowledges_constraint
    return ok, {"tools": names, "final": result["final"]}

def main():
    import sys
    models = sys.argv[1:] or ["qwen3:1.7b", "granite3.3:2b", "phi4-mini:3.8b", "ministral-3:3b"]
    print("model,case,ok,turns,wall_s,output_tokens,tok_s")
    for model in models:
        for case in CASES:
            try:
                result = run_case(model, case)
                ok, detail = score(case, result)
                print(f"{model},{case['name']},{int(ok)},{len(result['trace'])},{result['wall_s']:.3f},{result['tokens']},{result['tok_s']:.2f}")
                if not ok:
                    print("  FAIL:", json.dumps(detail, ensure_ascii=False))
                for t in result["trace"]:
                    calls = [((x.get("function") or {}).get("name"), arguments(x)) for x in t.get("tool_calls", [])]
                    print(f"  T{t['turn']}: tools={calls} content={t['content']!r}")
            except Exception as e:
                print(f"{model},{case['name']},0,ERROR,,,,")
                print("  ERROR:", repr(e))

if __name__ == "__main__":
    main()
