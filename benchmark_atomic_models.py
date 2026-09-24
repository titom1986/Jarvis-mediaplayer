#!/usr/bin/env python3
"""Run the same non-held-out catalogue prompts against all candidate Ollama models.

The runner launches agent.py in a fresh process per case so config.MODEL is
reloaded cleanly. It records raw agent traces plus host/Ollama resource snapshots.
No service responses are cached between cases.
"""
import json
import os
import subprocess
import time
from datetime import datetime

MODELS = [
    "qwen3:4b-instruct",
    "qwen3:1.7b",
    "granite3.3:2b",
    "phi4-mini:3.8b",
    "ministral-3:3b",
]

# Representative but deliberately NOT the reserved final E2E prompt.
CASES = [
    ("semantic_grounding", "Je cherche un film de science-fiction avec Tom Cruise sorti entre 2000 et 2015 avec des extraterrestres"),
    ("plain_constraints", "Je cherche un film d'action avec Keanu Reeves sorti entre 1990 et 2010"),
    ("explicit_exclusion", "Je cherche un film de science-fiction des années 2000 mais pas sorti en 2005"),
]

REPORT = "benchmark-atomic-models.txt"


def run(cmd, env=None):
    try:
        return subprocess.run(
            cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, check=False,
        )
    except Exception as exc:
        return subprocess.CompletedProcess(cmd, 127, stdout=f"[diagnostic error] {type(exc).__name__}: {exc}\\n")


def snapshot():
    parts = []
    for cmd in (["uptime"], ["free", "-h"], ["docker", "stats", "--no-stream", "ollama"]):
        result = run(cmd)
        parts.append("$ " + " ".join(cmd) + "\n" + result.stdout.rstrip())
    return "\n".join(parts)


def ollama_models():
    result = run(["docker", "exec", "ollama", "ollama", "list"])
    return result.returncode, result.stdout.rstrip()


with open(REPORT, "w", encoding="utf-8") as out:
    def emit(value=""):
        print(value, file=out, flush=True)

    emit("=== ATOMIC MODEL BENCHMARK ===")
    emit("started=" + datetime.now().astimezone().isoformat())
    emit(run(["git", "log", "-1", "--oneline"]).stdout.rstrip())
    emit()
    emit("=== OLLAMA MODELS ===")
    rc, listing = ollama_models()
    emit(listing)
    emit(f"ollama_list_exit={rc}")
    emit()
    emit("=== SYSTEM BEFORE ===")
    emit(snapshot())

    summary = []
    for model in MODELS:
        for case_name, prompt in CASES:
            emit()
            emit("=" * 100)
            emit(f"MODEL={model}")
            emit(f"CASE={case_name}")
            emit("PROMPT=" + prompt)
            emit("=" * 100)
            env = os.environ.copy()
            env["OLLAMA_MODEL"] = model
            before = snapshot()
            emit("--- RESOURCE BEFORE ---")
            emit(before)
            started = time.perf_counter()
            proc = run(["python3", "agent.py", prompt], env=env)
            elapsed = time.perf_counter() - started
            emit("--- RAW AGENT TRACE ---")
            emit(proc.stdout.rstrip())
            emit("--- RESOURCE AFTER ---")
            emit(snapshot())
            emit(f"PROCESS_EXIT={proc.returncode}")
            emit(f"PROCESS_WALL_S={elapsed:.3f}")
            summary.append({
                "model": model,
                "case": case_name,
                "exit": proc.returncode,
                "wall_s": round(elapsed, 3),
            })

    emit()
    emit("=== MACHINE SUMMARY ===")
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    emit()
    emit("=== SYSTEM AFTER ALL ===")
    emit(snapshot())
    emit("finished=" + datetime.now().astimezone().isoformat())
