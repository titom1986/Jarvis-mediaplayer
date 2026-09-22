import json
import requests

from config import OLLAMA_URL, MODEL


CRITERIA_SCHEMA = {
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
                    "from": {"type": ["integer", "null"]},
                    "to": {"type": ["integer", "null"]}
                },
                "required": ["from", "to"]
            }
        }
    },
    "required": ["people", "genres", "keywords", "dates"]
}

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "media_type": {"type": "string", "enum": ["movie", "tv", "unknown"]},
        "search": {"type": "boolean"},
        "required": CRITERIA_SCHEMA,
        "forbidden": CRITERIA_SCHEMA,
        "alternatives": {"type": "array", "items": CRITERIA_SCHEMA},
        "recommend_by_rating": {"type": "boolean"},
        "avoid_watched": {"type": "boolean"},
        "download": {"type": "boolean"},
        "french_download": {"type": "boolean"}
    },
    "required": [
        "media_type", "search", "required", "forbidden", "alternatives",
        "recommend_by_rating", "avoid_watched", "download", "french_download"
    ]
}

EXTRACT_SYSTEM = """Tu extrais littéralement une intention de recherche multimédia.
Ne planifie aucun appel d'outil et n'invente aucun critère.
Chaque contrainte sémantique de la demande doit apparaître UNE SEULE FOIS dans UNE SEULE catégorie,
la plus spécifique. Les catégories sont mutuellement exclusives : une personne va uniquement dans
people, un genre uniquement dans genres, un thème/concept uniquement dans keywords et une période
uniquement dans dates. Ne duplique jamais une même information entre people, genres et keywords.
Exemple : « science-fiction avec Bruce Willis » => genres=["Science Fiction"],
people=["Bruce Willis"], keywords=[].
required contient TOUS les critères cumulatifs demandés.
forbidden contient les propriétés que l'utilisateur refuse. Les valeurs y sont POSITIVES :
« pas dystopique » => keyword « dystopia », jamais « not dystopian ».
alternatives est utilisé UNIQUEMENT si l'utilisateur formule explicitement des branches avec
« soit/ou/or ». Chaque branche est un ensemble cumulatif de critères.
people = personnes. genres = genres de catalogue en anglais. keywords = thèmes/concepts de
métadonnées en anglais. dates = bornes inclusives ; années 90 => 1990..1999.
recommend_by_rating n'est vrai que si la note doit servir à choisir/classer.
avoid_watched n'est vrai que si les éléments déjà vus doivent être évités.
download n'est vrai que sur demande explicite d'ajout/téléchargement.
french_download n'est vrai que si une version française est explicitement demandée.
Retourne uniquement le JSON conforme au schéma."""


def extract_intent(question, post=requests.post):
    response = post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": question},
            ],
            "format": INTENT_SCHEMA,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": 0, "num_predict": 220},
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    intent = json.loads(content)
    validate_intent(intent)
    return intent


def _criteria_values(criteria):
    for category in ("people", "genres", "keywords"):
        for value in criteria.get(category, []):
            yield category, str(value).strip().casefold()


def validate_intent(intent):
    """Valide uniquement les invariants structurels, sans connaissance métier."""
    for scope in ("required", "forbidden"):
        seen = {}
        for category, value in _criteria_values(intent.get(scope, {})):
            if not value:
                continue
            previous = seen.get(value)
            if previous and previous != category:
                raise ValueError(
                    f"Extraction ambiguë : {value!r} apparaît dans {previous} et {category}"
                )
            seen[value] = category

    return intent


def compile_media_search(intent):
    if not intent.get("search") or intent.get("media_type") not in ("movie", "tv"):
        return None

    required = intent["required"]
    alternatives = intent.get("alternatives") or []

    # Sans alternative explicite, tous les critères demandés restent dans UN groupe AND.
    include = alternatives if alternatives else [required]

    # Une propriété interdite est un groupe d'exclusion cumulatif unique.
    forbidden = intent["forbidden"]
    has_forbidden = any(forbidden.get(key) for key in ("people", "genres", "keywords", "dates"))
    exclude = [forbidden] if has_forbidden else []

    return {
        "media_type": intent["media_type"],
        "include": include,
        "exclude": exclude,
    }
