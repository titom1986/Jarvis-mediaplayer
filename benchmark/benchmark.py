import json
import time
import urllib.request

OLLAMA = "http://127.0.0.1:11435/api/chat"

MEDIA_TOOL = {
    "type": "function",
    "function": {
        "name": "seerr_media_search",
        "description": "Recherche films/séries. Contraintes d'un groupe = AND; groupes = OR. exclude porte la négation.",
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
        "description": "Vérifie si un titre est présent dans Plex et s'il a déjà été vu.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}
    }
}

SYSTEM = """Tu es l'interpréteur d'un media center. Choisis les outils nécessaires.
Toutes les contraintes cumulatives vont dans un même groupe include; plusieurs groupes seulement pour des alternatives OR.
Une personne va dans people, un genre dans genres, un concept/thème dans keywords, une période dans dates.
Une décennie est inclusive (années 90 = 1990..1999). exclude contient le concept positif à exclure.
N'invente aucun critère. Réponds dans la langue de l'utilisateur."""

CASES = [
    {
        "name": "bruce_sf_90s",
        "prompt": "Je voudrais un film de science-fiction avec Bruce Willis des années 90. Je ne veux pas de film dystopique.",
        "expect": {"tool": "seerr_media_search", "media_type": "movie", "person": "bruce willis",
                   "genre": "science fiction", "date": [1990, 1999], "exclude_keyword": "dystopia"}
    },
    {"name": "plex_simple", "prompt": "Est-ce que j'ai Interstellar dans Plex ?", "expect": {"tool": "plex_status", "title": "interstellar"}},
    {
        "name": "or_people",
        "prompt": "Trouve-moi un film avec Bruce Willis ou Brad Pitt.",
        "expect": {"tool": "seerr_media_search", "media_type": "movie", "or_people": ["bruce willis", "brad pitt"]}
    },
    {
        "name": "recent_thriller",
        "prompt": "Trouve-moi un thriller sorti entre 2020 et 2025.",
        "expect": {"tool": "seerr_media_search", "media_type": "movie", "genre": "thriller", "date": [2020, 2025]}
    }
]

def post(model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                       "tools": [MEDIA_TOOL, PLEX_TOOL], "stream": False, "keep_alive": "30m", "think": False,
                       "options": {"temperature": 0, "num_predict": 120}}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    return data, time.perf_counter() - started

def norm(x): return str(x or "").strip().casefold()

def score(expect, msg):
    calls = msg.get("tool_calls") or []
    if not calls: return False, "no tool call"
    call = calls[0].get("function") or {}
    raw_args = call.get("arguments")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            return False, "arguments are invalid JSON string: " + repr(raw_args)
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        return False, "arguments are not an object: " + repr(raw_args)
    if call.get("name") != expect["tool"]: return False, "wrong tool"
    if expect["tool"] == "plex_status":
        return norm(args.get("title")) == expect["title"], str(args)
    if args.get("media_type") != expect.get("media_type"): return False, "wrong media_type"
    inc=args.get("include") or []; exc=args.get("exclude") or []
    if not isinstance(inc, list): return False, "include is not an array: " + repr(inc)
    if not isinstance(exc, list): return False, "exclude is not an array: " + repr(exc)
    if any(not isinstance(g, dict) for g in inc): return False, "include contains non-object groups: " + repr(inc)
    if any(not isinstance(g, dict) for g in exc): return False, "exclude contains non-object groups: " + repr(exc)
    if "or_people" in expect:
        got=[norm(p) for g in inc for p in g.get("people",[])]
        ok=len(inc)>=2 and all(p in got for p in expect["or_people"])
        return ok, str(args)
    if not inc: return False, "empty include"
    g=inc[0]
    checks=[]
    if "person" in expect: checks.append(expect["person"] in [norm(x) for x in g.get("people",[])])
    if "genre" in expect: checks.append(expect["genre"] in [norm(x) for x in g.get("genres",[])])
    if "date" in expect: checks.append(expect["date"] in [[d.get("from"),d.get("to")] for d in g.get("dates",[])])
    if "exclude_keyword" in expect:
        checks.append(expect["exclude_keyword"] in [norm(x) for e in exc for x in e.get("keywords",[])])
    return all(checks), str(args)

def main():
    import sys
    models=sys.argv[1:] or [x.strip() for x in open("models.txt") if x.strip()]
    print("model,case,ok,wall_s,prompt_tokens,output_tokens,tok_s")
    for model in models:
        for case in CASES:
            try:
                data, wall=post(model,case["prompt"])
                ok, detail=score(case["expect"],data["message"])
                out=data.get("eval_count") or 0; ev=(data.get("eval_duration") or 0)/1e9
                rate=out/ev if ev else 0
                print(f"{model},{case['name']},{int(ok)},{wall:.3f},{data.get('prompt_eval_count')},{out},{rate:.2f}")
                if not ok:
                    print("  FAIL:", detail)
                    msg = data.get("message") or {}
                    print("  CONTENT:", repr(msg.get("content", "")))
                    print("  TOOL_CALLS:", json.dumps(msg.get("tool_calls") or [], ensure_ascii=False))
            except Exception as e:
                print(f"{model},{case['name']},0,ERROR,,,,")
                print("  ERROR:",repr(e))

if __name__ == "__main__":
    main()
