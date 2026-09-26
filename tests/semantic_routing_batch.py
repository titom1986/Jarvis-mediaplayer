"""Live semantic-routing benchmark for the local Jarvis model.

Safe by design: it asks Ollama for the FIRST tool choice only and never executes
that tool, so request/download tools cannot mutate Radarr or Sonarr.

The cases test semantic invariants, not memorized phrases.  Run on Seedhost:
    python3 tests/semantic_routing_batch.py
"""
import sys
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


def first_tool(question):
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
    response = requests.post(OLLAMA_URL, json=payload, timeout=None)
    response.raise_for_status()
    calls = response.json()["message"].get("tool_calls", [])
    return calls[0]["function"]["name"] if calls else None


def main():
    failures = []
    for index, (question, expected) in enumerate(CASES, 1):
        actual = first_tool(question)
        ok = actual in expected
        print(f"{'PASS' if ok else 'FAIL'} {index:02d}  {actual!s:28} {question}")
        if not ok:
            failures.append((question, expected, actual))
    print(f"\n{len(CASES) - len(failures)}/{len(CASES)} semantic routes passed")
    if failures:
        print("\nFailures:")
        for question, expected, actual in failures:
            print(f"- {question!r}: expected {sorted(expected)}, got {actual!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
