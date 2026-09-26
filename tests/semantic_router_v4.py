#!/usr/bin/env python3
"""V4 hold-out benchmark for the hierarchical semantic router.

All prompts are new relative to v3. Only semantic_router.route() is called;
no Plex/Radarr/Sonarr/catalogue operation is ever executed.
"""
import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import semantic_router


CASES = [
    # named_action
    ("named_action", "Tu peux me ramener Dune ?"),
    ("named_action", "Je veux Arrival sur Plex"),
    ("named_action", "Ajoute-moi Shogun"),
    ("named_action", "Faut que je récupère Her"),
    ("named_action", "Envoie The Holiday dans Radarr"),
    ("named_action", "Mets Andor à télécharger"),
    ("named_action", "Va me chercher Blade Runner 2049"),
    ("named_action", "J'aimerais récupérer Ça"),
    ("named_action", "Tu peux ajouter Fallout ?"),
    ("named_action", "Télécharge-moi Lost"),
    ("named_action", "Prends l'épisode 6 de la saison 1 de Fallout"),
    ("named_action", "Je veux la saison 3 de The White Lotus"),
    ("named_action", "Récupère-moi l'épisode 2 saison 1 d'Andor"),
    ("named_action", "Refais partir le téléchargement de Dune"),
    ("named_action", "Il me faut l'épisode suivant de Shogun"),

    # discovery
    ("discovery", "Une série avec un détective mais pas trop sombre"),
    ("discovery", "Quels films a fait Rebecca Ferguson ?"),
    ("discovery", "Trouve quelque chose avec des dinosaures"),
    ("discovery", "Je cherche une comédie sortie récemment"),
    ("discovery", "Un film des années 80 avec Harrison Ford"),
    ("discovery", "Propose une série historique"),
    ("discovery", "Tu as des films sur l'intelligence artificielle ?"),
    ("discovery", "Cherche-moi un thriller avec Jake Gyllenhaal"),
    ("discovery", "Je veux regarder un truc de guerre des années 2000"),
    ("discovery", "Qu'est-ce que tu proposes comme film familial ?"),
    ("discovery", "Trouve une série avec des zombies et ajoute-la"),
    ("discovery", "Récupère un bon film avec Amy Adams"),
    ("discovery", "Mets-moi une comédie romantique récente"),
    ("discovery", "Je voudrais un film qui se passe dans l'espace"),
    ("discovery", "Des séries nordiques policières, tu as quoi ?"),

    # status
    ("status", "Dune est déjà dans Plex ?"),
    ("status", "Est-ce que j'ai Arrival ?"),
    ("status", "Andor est connue de Sonarr ?"),
    ("status", "Vérifie si The Holiday est dans Radarr"),
    ("status", "Fallout est disponible chez moi ?"),
    ("status", "J'ai déjà Her dans ma médiathèque ?"),
    ("status", "Lost est présente dans Sonarr ou non ?"),
    ("status", "Tu vois Shogun dans Plex ?"),
    ("status", "Blade Runner 2049 est déjà ajouté ?"),
    ("status", "Est-ce que The White Lotus est chez moi ?"),

    # download_status
    ("download_status", "Dune est à combien de pourcent ?"),
    ("download_status", "Arrival a fini de se télécharger ?"),
    ("download_status", "Ça télécharge toujours Andor ?"),
    ("download_status", "The Holiday est encore dans la queue ?"),
    ("download_status", "Le transfert de Fallout est terminé ?"),
    ("download_status", "Où en est le download de Her ?"),
    ("download_status", "Lost est toujours en train de descendre ?"),
    ("download_status", "Il reste du téléchargement sur Shogun ?"),
    ("download_status", "Blade Runner 2049 est encore en cours ?"),
    ("download_status", "The White Lotus, le téléchargement est fini ?"),
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
    parser.add_argument("--run-id", default="router-v4")
    parser.add_argument("--results", default="semantic-v4-router-results.jsonl")
    parser.add_argument("--summary", default="semantic-v4-router-summary.json")
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
