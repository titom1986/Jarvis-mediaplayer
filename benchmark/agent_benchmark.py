import json
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11435/api/chat"

MEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": "Recherche des films ou séries selon des groupes de critères. Toutes les contraintes people, genres, keywords et dates d'un même groupe sont simultanées (AND). Les groupes sont des alternatives (OR). Crée un nouveau groupe uniquement si la demande exprime une alternative. exclude porte déjà la négation : ses valeurs nomment positivement la propriété à retirer. people contient les personnes, genres les genres de catalogue, keywords les thèmes/concepts et dates les périodes. N'ajoute aucun critère non demandé.",
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

# Each case provides the result that the real deterministic tool would return
# AFTER applying the requested filters. No semantic filtering is duplicated here.
CASES = [
    {
        "name": "watched_only_match",
        "prompt": "Je voudrais un film de science-fiction avec Bruce Willis des années 90 que tu me conseillerais basé sur sa note. Je ne veux pas de film dystopique et, s'il est présent dans Plex, évite ceux que j'ai déjà vus.",
        "expected_search": {"media_type": "movie", "person": "bruce willis", "genre": "science fiction", "date": [1990, 1999], "exclude_keyword": "dystopia"},
        "expected_search": {"media_type": "movie", "person": "bruce willis", "genre": "science fiction", "date": [1990, 1999], "exclude_keyword": "dystopia"},
        "search_result": {"count": 1, "results": [
            {"title": "Armageddon", "year": 1998, "rating": 6.83, "voteCount": 8844}
        ]},
        "plex": {"Armageddon": {"found": True, "title": "Armageddon", "viewCount": 1, "watched": True}},
        "expected_plex": ["armageddon"],
        "final_mode": "no_match",
        "max_turns": 6,
    },
    {
        "name": "unwatched_match",
        "prompt": "Je voudrais un film de science-fiction avec Bruce Willis des années 90 que tu me conseillerais basé sur sa note. Je ne veux pas de film dystopique et, s'il est présent dans Plex, évite ceux que j'ai déjà vus.",
        "search_result": {"count": 1, "results": [
            {"title": "Armageddon", "year": 1998, "rating": 6.83, "voteCount": 8844}
        ]},
        "plex": {"Armageddon": {"found": True, "title": "Armageddon", "viewCount": 0, "watched": False}},
        "expected_plex": ["armageddon"],
        "final_mode": "recommend_armageddon",
        "max_turns": 6,
    },
    {
        "name": "choose_best_rating",
        "prompt": "Trouve-moi un thriller entre 2020 et 2025 et conseille-moi le mieux noté. S'il est dans Plex et déjà vu, prends le suivant.",
        "expected_search": {"media_type": "movie", "genre": "thriller", "date": [2020, 2025]},
        "search_result": {"count": 3, "results": [
            {"title": "Alpha", "year": 2023, "rating": 8.4, "voteCount": 6200},
            {"title": "Bravo", "year": 2022, "rating": 8.0, "voteCount": 9100},
            {"title": "Charlie", "year": 2021, "rating": 7.6, "voteCount": 12000}
        ]},
        "plex": {
            "Alpha": {"found": True, "title": "Alpha", "viewCount": 1, "watched": True},
            "Bravo": {"found": True, "title": "Bravo", "viewCount": 0, "watched": False},
            "Charlie": {"found": False, "title": "Charlie"}
        },
        "expected_plex": ["alpha", "bravo"],
        "final_mode": "recommend_bravo",
        "max_turns": 7,
    },
    {
        "name": "realistic_cardinality",
        "prompt": "Trouve-moi un thriller entre 2020 et 2025, conseille-moi le mieux noté, mais évite ceux que j'ai déjà vus dans Plex.",
        "expected_search": {"media_type": "movie", "genre": "thriller", "date": [2020, 2025]},
        "search_result": {"count": 12, "results": [
            {"title": "R01", "year": 2024, "rating": 8.8, "voteCount": 14000},
            {"title": "R02", "year": 2023, "rating": 8.6, "voteCount": 12000},
            {"title": "R03", "year": 2025, "rating": 8.4, "voteCount": 9000},
            {"title": "R04", "year": 2022, "rating": 8.2, "voteCount": 16000},
            {"title": "R05", "year": 2021, "rating": 8.0, "voteCount": 8000},
            {"title": "R06", "year": 2020, "rating": 7.9, "voteCount": 15000},
            {"title": "R07", "year": 2024, "rating": 7.8, "voteCount": 7000},
            {"title": "R08", "year": 2023, "rating": 7.7, "voteCount": 6000},
            {"title": "R09", "year": 2022, "rating": 7.6, "voteCount": 5000},
            {"title": "R10", "year": 2021, "rating": 7.5, "voteCount": 4000},
            {"title": "R11", "year": 2020, "rating": 7.4, "voteCount": 3000},
            {"title": "R12", "year": 2025, "rating": 7.3, "voteCount": 2000}
        ]},
        "plex": {
            "R01": {"found": True, "title": "R01", "viewCount": 1, "watched": True},
            "R02": {"found": True, "title": "R02", "viewCount": 1, "watched": True},
            "R03": {"found": True, "title": "R03", "viewCount": 0, "watched": False}
        },
        "expected_plex": ["r01", "r02", "r03"],
        "final_mode": "recommend_r03",
        "max_turns": 8,
    },
    {
        "name": "plex_direct",
        "prompt": "Est-ce que j'ai Armageddon dans Plex et est-ce que je l'ai déjà vu ?",
        "plex": {"Armageddon": {"found": True, "title": "Armageddon", "viewCount": 1, "watched": True}},
        "expected_plex": ["armageddon"],
        "final_mode": "plex_watched",
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

def _norm_list(values):
    return [norm(x) for x in (values or [])]

def validate_search_args(args, expected):
    if not expected:
        return True, ""
    if args.get("media_type") != expected.get("media_type"):
        return False, "wrong media_type"
    inc = args.get("include")
    exc = args.get("exclude") or []
    if not isinstance(inc, list) or not inc or any(not isinstance(g, dict) for g in inc):
        return False, "invalid include"
    if not isinstance(exc, list) or any(not isinstance(g, dict) for g in exc):
        return False, "invalid exclude"
    flat_people = [norm(x) for g in inc for x in g.get("people", [])]
    flat_genres = [norm(x) for g in inc for x in g.get("genres", [])]
    flat_dates = [[d.get("from"), d.get("to")] for g in inc for d in g.get("dates", []) if isinstance(d, dict)]
    flat_ex_kw = [norm(x) for g in exc for x in g.get("keywords", [])]
    if expected.get("person") and expected["person"] not in flat_people:
        return False, "missing person"
    if expected.get("genre") and expected["genre"] not in flat_genres:
        return False, "missing genre"
    if expected.get("date") and expected["date"] not in flat_dates:
        return False, "missing date"
    if expected.get("exclude_keyword") and expected["exclude_keyword"] not in flat_ex_kw:
        return False, "missing exclusion"
    return True, ""

def execute(call, case):
    fn = (call.get("function") or {}).get("name")
    args = arguments(call)
    if fn == "seerr_media_search":
        ok, why = validate_search_args(args, case.get("expected_search"))
        if not ok:
            return {"error": "invalid search arguments", "detail": why}
        return case.get("search_result", {"count": 0, "results": []})
    if fn == "plex_status":
        title = args.get("title")
        return case.get("plex", {}).get(title, {"found": False, "title": title})
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
        trace.append({"turn": turn, "wall_s": round(wall, 3), "content": msg.get("content", ""), "tool_calls": calls,
                      "done_reason": data.get("done_reason")})
        messages.append(msg)
        if not calls:
            final = msg.get("content") or ""
            break
        for tc in calls:
            result = execute(tc, case)
            trace[-1].setdefault("tool_results", []).append(result)
            messages.append(tool_message(tc, result))
    return {"final": final, "trace": trace, "wall_s": total_wall, "tokens": total_out,
            "tok_s": (total_out / total_eval if total_eval else 0)}

def norm(s):
    return str(s or "").casefold()

def score(case, result):
    trace = result["trace"]
    calls, tool_results = [], []
    for t in trace:
        tool_results.extend(t.get("tool_results", []))
        for tc in t.get("tool_calls", []):
            fn = (tc.get("function") or {}).get("name")
            try:
                args = arguments(tc)
            except Exception:
                args = {}
            calls.append((fn, args))
    names = [x[0] for x in calls]
    final = norm(result["final"])
    plex_titles = [norm(args.get("title")) for fn, args in calls if fn == "plex_status"]
    expected_plex = case.get("expected_plex", [])
    plex_ok = all(title in plex_titles for title in expected_plex)
    search_ok = True
    if case.get("expected_search"):
        search_calls = [args for fn, args in calls if fn == "seerr_media_search"]
        search_ok = bool(search_calls) and any(validate_search_args(args, case["expected_search"])[0] for args in search_calls)

    mode = case["final_mode"]
    if mode == "plex_watched":
        final_ok = "armageddon" in final and any(x in final for x in ["vu", "visionn", "déjà"])
        ok = names and names[0] == "plex_status" and plex_ok and final_ok
    elif mode == "no_match":
        final_ok = "armageddon" in final and any(x in final for x in ["aucun", "pas d'autre", "rien d'autre", "déjà vu", "déjà regard", "déjà visionn"])
        ok = search_ok and plex_ok and final_ok
    elif mode == "recommend_armageddon":
        ok = search_ok and plex_ok and "armageddon" in final
    elif mode == "recommend_bravo":
        ok = search_ok and plex_ok and "bravo" in final
    elif mode == "recommend_r03":
        ok = search_ok and plex_ok and "r03" in final
    else:
        ok = False
    return ok, {"tools": names, "plex_titles": plex_titles, "search_ok": search_ok, "final": result["final"]}

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
                    calls = []
                    for x in t.get("tool_calls", []):
                        try:
                            calls.append(((x.get("function") or {}).get("name"), arguments(x)))
                        except Exception as e:
                            calls.append(("MALFORMED", repr(e)))
                    print(f"  T{t['turn']}: done={t.get('done_reason')} tools={calls} content={t['content']!r}")
            except Exception as e:
                print(f"{model},{case['name']},0,ERROR,,,,")
                print("  ERROR:", repr(e))

if __name__ == "__main__":
    main()
