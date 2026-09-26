"""Live semantic-routing benchmark for the local Jarvis model.

Safe by design: it asks Ollama for the FIRST tool choice only and never executes
that tool, so request/download tools cannot mutate Radarr or Sonarr.

The cases test semantic invariants, not memorized phrases.  Run on Seedhost:
    python3 tests/semantic_routing_batch.py
"""
import sys
import os
import json
import time
import math
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from config import OLLAMA_URL, MODEL
from agent import TOOLS, _model_config

# expected is a set because equivalent first steps can be valid for discovery.
CASES = [
    # Explicit named movie target -> resolve title, never decompose title words.
    ("Télécharge Love in Lapland en français", {"catalog_title"}),
    ("Télécharge Love in Lapland", {"catalog_title"}),
    ("Ajoute Love in Lapland", {"catalog_title"}),
    ("Récupère Love in Lapland en VF", {"catalog_title"}),
    ("Je veux Love in Lapland en français", {"catalog_title"}),
    ("Tu peux me télécharger Love in Lapland ?", {"catalog_title"}),
    ("Mets-moi Love in Lapland", {"catalog_title"}),
    ("Je voudrais le film Love in Lapland", {"catalog_title"}),
    ("Télécharge Heat", {"catalog_title"}),
    ("Ajoute Cars", {"catalog_title"}),
    ("Je veux Her", {"catalog_title"}),
    ("Récupère The Holiday", {"catalog_title"}),
    ("Télécharge Ça", {"catalog_title"}),
    ("Mets Die Hard sur le serveur", {"catalog_title"}),
    ("Ajoute Dune Part Two", {"catalog_title"}),
    ("telecharge Interstellar", {"catalog_title"}),
    ("téléchage Gladiator", {"catalog_title"}),
    ("peux tu recuperer Oppenheimer", {"catalog_title"}),

    # Explicit named TV target -> same invariant.
    ("Télécharge la série Severance", {"catalog_title"}),
    ("Ajoute toute la série Dark", {"catalog_title"}),
    ("Je veux la saison 2 de Severance", {"catalog_title"}),
    ("Télécharge la saison 3 de The Bear", {"catalog_title"}),
    ("Récupère l'épisode 4 de la saison 2 de Severance", {"catalog_title"}),
    ("Je veux Severance S02E04", {"catalog_title"}),
    ("Retélécharge Severance S02E04", {"catalog_title"}),
    ("Redemande l'épisode 3 de la saison 1 de Dark", {"catalog_title"}),

    # Discovery: no named target. Constraints are appropriate.
    ("Trouve-moi un film avec Bruce Willis", {"catalog_person"}),
    ("Trouve-moi un film de Noël", {"catalog_keyword", "catalog_keyword_vocabulary"}),
    ("Je cherche un film de science-fiction", {"catalog_genre"}),
    ("Un thriller avec Tom Hanks", {"catalog_genre", "catalog_person"}),
    ("Un film romantique des années 90", {"catalog_genre", "catalog_years"}),
    ("Trouve une série avec Bryan Cranston", {"catalog_person"}),
    ("Je cherche une série de science-fiction", {"catalog_genre"}),
    ("Un film sur le voyage dans le temps", {"catalog_keyword", "catalog_keyword_vocabulary"}),
    ("Des films avec Tomer Sisley", {"catalog_person"}),
    ("Quels films avec Lars Mikkelsen ?", {"catalog_person"}),

    # Discovery + eventual action: discovery must happen before the write.
    ("Télécharge-moi un film avec Bruce Willis", {"catalog_person"}),
    ("Ajoute un film de Noël", {"catalog_keyword", "catalog_keyword_vocabulary"}),
    ("Télécharge une comédie avec Tom Hanks", {"catalog_genre", "catalog_person"}),
    ("Je veux un thriller des années 2000", {"catalog_genre", "catalog_years"}),
    ("Télécharge une série avec Pedro Pascal", {"catalog_person"}),

    # Status/queue on a named target: direct service lookup is valid; no semantic
    # catalogue decomposition should be necessary.
    ("Est-ce que j'ai Heat dans Plex ?", {"plex_status"}),
    ("Love in Lapland est dans Radarr ?", {"radarr_status"}),
    ("Où en est le téléchargement de Love in Lapland ?", {"radarr_queue_status"}),
    ("Severance est dans Sonarr ?", {"sonarr_status"}),
    ("Où en est le téléchargement de Severance ?", {"sonarr_queue_status"}),
    ("Est-ce que Cars est disponible sur Plex ?", {"plex_status"}),

    # Titles made of ordinary semantic words: regression against title decomposition.
    ("Télécharge Love Story", {"catalog_title"}),
    ("Télécharge A Christmas Story", {"catalog_title"}),
    ("Ajoute The Accountant", {"catalog_title"}),
    ("Télécharge The Tourist", {"catalog_title"}),
    ("Je veux The Bear", {"catalog_title"}),
    ("Ajoute You", {"catalog_title"}),
    ("Télécharge From", {"catalog_title"}),
    ("Je veux Dark", {"catalog_title"}),
    ("Télécharge Wednesday", {"catalog_title"}),
    ("Ajoute Foundation", {"catalog_title"}),
    ("Télécharge Lost", {"catalog_title"}),
    ("Je veux Silo", {"catalog_title"}),
]


def classify_case(index):
    if index <= 18:
        return "named_movie"
    if index <= 26:
        return "named_tv"
    if index <= 36:
        return "discovery"
    if index <= 41:
        return "discovery_action"
    if index <= 47:
        return "status"
    return "opaque_title"


def probe_first_tool(question):
    """Return full first-turn telemetry; never execute a tool."""
    system, extra = _model_config()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0, "num_predict": 120},
    }
    payload.update(extra)

    started = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=None)
    response.raise_for_status()
    data = response.json()
    wall_s = time.perf_counter() - started
    message = data.get("message") or {}
    calls = message.get("tool_calls") or []
    first = calls[0].get("function", {}) if calls else {}
    return {
        "tool": first.get("name"),
        "arguments": first.get("arguments") or {},
        "tool_call_count": len(calls),
        "content": message.get("content", ""),
        "wall_s": round(wall_s, 3),
        "load_ms": round((data.get("load_duration") or 0) / 1_000_000, 1),
        "prompt_eval_ms": round((data.get("prompt_eval_duration") or 0) / 1_000_000, 1),
        "eval_ms": round((data.get("eval_duration") or 0) / 1_000_000, 1),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }


def _append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_completed(path):
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("run_id") and item.get("index"):
                completed[(item["run_id"], int(item["index"]))] = item
    return completed


def _write_summary(path, run_id, records, started_at):
    records = sorted(records, key=lambda x: x["index"])
    passed = sum(1 for x in records if x.get("ok"))
    failed = sum(1 for x in records if not x.get("ok") and not x.get("error"))
    errors = sum(1 for x in records if x.get("error"))
    walls = [x["metrics"]["wall_s"] for x in records if x.get("metrics")]
    prompt_tokens = [x["metrics"].get("prompt_tokens") or 0 for x in records if x.get("metrics")]
    output_tokens = [x["metrics"].get("output_tokens") or 0 for x in records if x.get("metrics")]
    by_category = {}
    confusion = {}
    for item in records:
        cat = item["category"]
        bucket = by_category.setdefault(cat, {"total": 0, "passed": 0, "failed": 0, "errors": 0})
        bucket["total"] += 1
        if item.get("ok"):
            bucket["passed"] += 1
        elif item.get("error"):
            bucket["errors"] += 1
        else:
            bucket["failed"] += 1
        actual = item.get("actual_tool") or "<none>"
        confusion[actual] = confusion.get(actual, 0) + 1

    elapsed = time.time() - started_at
    summary = {
        "run_id": run_id,
        "model": MODEL,
        "completed": len(records),
        "total_cases": len(CASES),
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "accuracy_pct": round(100 * passed / len(records), 2) if records else 0,
        "elapsed_s": round(elapsed, 1),
        "mean_wall_s": round(statistics.mean(walls), 3) if walls else None,
        "median_wall_s": round(statistics.median(walls), 3) if walls else None,
        "p95_wall_s": round(sorted(walls)[max(0, math.ceil(.95 * len(walls)) - 1)], 3) if walls else None,
        "total_prompt_tokens": sum(prompt_tokens),
        "total_output_tokens": sum(output_tokens),
        "by_category": by_category,
        "actual_tool_counts": confusion,
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", help="Stable id used for resume. Defaults to a new timestamp.")
    parser.add_argument("--results", default="semantic-results.jsonl")
    parser.add_argument("--summary", default="semantic-summary.json")
    parser.add_argument("--retries", type=int, default=2, help="Retries for transport/server errors only.")
    parser.add_argument("--delay", type=float, default=0.0, help="Optional delay between cases.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    results_path = Path(args.results)
    summary_path = Path(args.summary)
    completed = _load_completed(results_path)
    records = [v for (rid, _), v in completed.items() if rid == run_id]
    done_indices = {x["index"] for x in records}
    started_at = time.time()

    print(f"Semantic benchmark | model={MODEL} | run_id={run_id} | cases={len(CASES)}")
    if done_indices:
        print(f"Resume: {len(done_indices)} case(s) already persisted in {results_path}")

    try:
        for index, (question, expected) in enumerate(CASES, 1):
            if index in done_indices:
                continue
            category = classify_case(index)
            telemetry = None
            error = None
            for attempt in range(1, max(1, args.retries) + 1):
                try:
                    telemetry = probe_first_tool(question)
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print(f"RETRY {index:02d} attempt={attempt}/{args.retries} {error}", flush=True)
                    if attempt < args.retries:
                        time.sleep(min(2 ** (attempt - 1), 5))

            actual = telemetry.get("tool") if telemetry else None
            ok = error is None and actual in expected
            record = {
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "index": index,
                "category": category,
                "question": question,
                "expected_tools": sorted(expected),
                "actual_tool": actual,
                "actual_arguments": telemetry.get("arguments") if telemetry else None,
                "tool_call_count": telemetry.get("tool_call_count") if telemetry else None,
                "ok": ok,
                "error": error if telemetry is None else None,
                "metrics": ({k: telemetry.get(k) for k in (
                    "wall_s", "load_ms", "prompt_eval_ms", "eval_ms",
                    "prompt_tokens", "output_tokens"
                )} if telemetry else None),
            }
            _append_jsonl(results_path, record)
            records.append(record)
            summary = _write_summary(summary_path, run_id, records, started_at)

            metric = record["metrics"] or {}
            status = "PASS" if ok else ("ERROR" if record["error"] else "FAIL")
            print(
                f"{status} {index:02d}/{len(CASES)} [{category}] "
                f"tool={actual!s} expected={','.join(sorted(expected))} "
                f"wall={metric.get('wall_s', '-')}s "
                f"tokens={metric.get('prompt_tokens', '-')}/{metric.get('output_tokens', '-')} "
                f"| {question}",
                flush=True,
            )
            if args.delay:
                time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\nInterrupted: completed cases are already persisted; rerun with the same --run-id to resume.")
        _write_summary(summary_path, run_id, records, started_at)
        raise SystemExit(130)

    summary = _write_summary(summary_path, run_id, records, started_at)
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [x for x in records if not x.get("ok")]
    if failures:
        print("\n=== FAILURES / ERRORS ===")
        for item in failures:
            print(
                f"{item['index']:02d} [{item['category']}] {item['question']!r} "
                f"expected={item['expected_tools']} actual={item.get('actual_tool')!r} "
                f"args={item.get('actual_arguments')!r} error={item.get('error')!r}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
