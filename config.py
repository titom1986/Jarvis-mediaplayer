import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434/api/chat"
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b-instruct"
)


SERVICES = {
    "radarr": {
        "url": os.getenv(
            "RADARR_URL",
            "http://127.0.0.1:7878/radarr"
        ),
        "api_key": os.getenv("RADARR_API_KEY")
    },

    "sonarr": {
        "url": os.getenv("SONARR_URL"),
        "api_key": os.getenv("SONARR_API_KEY")
    },

    "seerr": {
        "url": os.getenv("SEERR_URL"),
        "api_key": os.getenv("SEERR_API_KEY")
    },

    "plex": {
        "url": os.getenv("PLEX_URL"),
        "token": os.getenv("PLEX_TOKEN")
    }
}
