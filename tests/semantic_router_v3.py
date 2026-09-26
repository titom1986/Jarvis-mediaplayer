#!/usr/bin/env python3
"""V3 benchmark: semantic family routing with unseen paraphrases.

This benchmark calls only semantic_router.route(). It never executes media tools.
Cases deliberately use wording different from the v1/v2 tool-routing benchmark.
"""
import argparse
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import semantic_router


CASES = [
    # named_action — explicit target, varied register/typos/ellipsis
    ("named_action", "Tu pourrais me choper Silo ?"),
    ("named_action", "On se récupère Heat ?"),
    ("named_action", "Fais-moi venir Interstellar sur le serveur"),
    ("named_action", "J'aimerais avoir The Bear"),
    ("named_action", "Balance-moi Cars"),
    ("named_action", "Il me faudrait Oppenheimer en VF"),
    ("named_action", "Tu me mets Dark stp"),
    ("named_action", "Récupere Gladiator"),
    ("named_action", "Je voudrais bien You"),
    ("named_action", "On peut avoir From ?"),
    ("named_action", "Mets la deuxième saison de Severance"),
    ("named_action", "Choppe-moi Severance saison 2 épisode 4"),
    ("named_action", "Il me manque S02E04 de Severance"),
    ("named_action", "Relance l'épisode 3 saison 1 de Dark"),
    ("named_action", "Remets-moi Heat, je veux le retélécharger"),

    # discovery — constraints, even when eventual intent is acquisition
    ("discovery", "T'as quoi avec Bruce Willis ?"),
    ("discovery", "Je cherche quelque chose de drôle avec Tom Hanks"),
    ("discovery", "Propose-moi une série avec Pedro Pascal"),
    ("discovery", "Un truc de SF des années 90"),
    ("discovery", "J'aimerais un film qui parle de voyage dans le temps"),
    ("discovery", "Quels sont les films où joue Tomer Sisley ?"),
    ("discovery", "Trouve une série policière des années 2000"),
    ("discovery", "Je voudrais voir un film romantique avec Sandra Bullock"),
    ("discovery", "Choppe-moi un film de Noël avec Bruce Willis"),
    ("discovery", "Télécharge-moi une comédie avec Tom Hanks"),
    ("discovery", "Mets-moi un thriller des années 90"),
    ("discovery", "Je veux un film avec Lars Mikkelsen"),
    ("discovery", "T'aurais une série de science-fiction à me proposer ?"),
    ("discovery", "Quelque chose sur les voyages temporels, plutôt un film"),
    ("discovery", "Cherche des films de Noël"),

    # status — library/service presence, not transfer progress
    ("status", "J'ai Heat sur Plex ou pas ?"),
    ("status", "Love in Lapland est déjà chez Radarr ?"),
    ("status", "Severance est bien ajoutée dans Sonarr ?"),
    ("status", "Est-ce que Cars est dans ma bibliothèque Plex ?"),
    ("status", "Tu peux vérifier si Dark est présente dans Sonarr ?"),
    ("status", "Dis-moi si Oppenheimer est déjà dans Radarr"),
    ("status", "The Bear est dispo sur Plex ?"),
    ("status", "Est-ce que j'ai déjà Interstellar ?"),
    ("status", "Vérifie la présence de Silo dans Sonarr"),
    ("status", "Heat est déjà disponible chez moi ?"),

    # download_status — transfer/queue progress, deliberately no service named
    ("download_status", "Ça en est où pour le téléchargement de Heat ?"),
    ("download_status", "Severance, ça descend toujours ?"),
    ("download_status", "Il reste combien de temps pour Love in Lapland ?"),
    ("download_status", "Le téléchargement de Dark avance ?"),
    ("download_status", "Oppenheimer est encore en cours de téléchargement ?"),
    ("download_status", "Tu peux voir où en est Cars côté téléchargement ?"),
    ("download_status", "Ça a fini de télécharger Silo ?"),
    ("download_status", "The Bear est toujours dans la file de téléchargement ?"),
    ("download_status", "Le download de Gladiator en est où ?"),
    ("download_status", "Est-ce que Severance a terminé de descendre ?"),
]


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * p)))
    return ordered[index]


def load_completed(path, run_id):
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") == run_id and isinstance(row.get("index"), int):
                done[row["index"]] = row
    return done


def write_summary(path, run_id, rows, elapsed):
    passed = sum(bool(row.get("ok")) for row in rows)
    errors = sum(bool(row.get("error")) for row in rows)
    latencies = [row["wall_s"] for row in rows if isinstance(row.get("wall_s"), (int, float))]
    categories = defaultdict(lambda: {"total": 0, "passed": 0, "failed": 0})
    actual = Counter()
    for row in rows:
        bucket = categories[row["expected_family"]]
        bucket["total"] += 1
        bucket["passed" if row.get("ok") else "failed"] += 1
        actual[row.get("actual_family") or "<none>"] += 1
    summary = {
        "run_id": run_id,
        "model": semantic_router.MODEL,
        "completed": len(rows),
        "total_cases": len(CASES),
        "passed": passed,
        "failed": len(rows) - passed,
        "errors": errors,
        "accuracy_pct": round(100 * passed / len(rows), 2) if rows else 0,
        "elapsed_s": round(elapsed, 1),
        "mean_wall_s": round(statistics.mean(latencies), 3) if latencies else 0,
        "median_wall_s": round(statistics.median(latencies), 3) if latencies else 0,
        "p95_wall_s": round(percentile(latencies, .95), 3) if latencies else 0,
        "by_category": dict(categories),
        "actual_family_counts": dict(actual),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="router-v3")
    parser.add_argument("--results", default="semantic-v3-router-results.jsonl")
    parser.add_argument("--summary", default="semantic-v3-router-summary.json")
    parser.add_argument("--retries", type=int, default=1)
    args = parser.parse_args()

    completed = load_completed(args.results, args.run_id)
    started = time.perf_counter()

    for index, (expected, question) in enumerate(CASES, 1):
        if index in completed:
            continue
        actual = None
        error = None
        case_started = time.perf_counter()
        for attempt in range(args.retries + 1):
            try:
                actual = semantic_router.route(question)
                error = None
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(1)
        row = {
            "run_id": args.run_id,
            "index": index,
            "expected_family": expected,
            "actual_family": actual,
            "ok": actual == expected and error is None,
            "question": question,
            "error": error,
            "wall_s": round(time.perf_counter() - case_started, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(args.results, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        completed[index] = row
        print(f"[{index:02d}/{len(CASES)}] {'OK' if row['ok'] else 'FAIL'} {expected} <- {actual} | {question}", flush=True)

    rows = [completed[i] for i in sorted(completed) if 1 <= i <= len(CASES)]
    summary = write_summary(args.summary, args.run_id, rows, time.perf_counter() - started)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
