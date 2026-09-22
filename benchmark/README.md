# JARVIS model benchmark

Isolated Ollama instance for comparing small local models without touching the existing JARVIS/Ollama installation.

## Isolation

- separate Compose project: `jarvis-model-benchmark`
- separate container: `jarvis-model-benchmark-ollama`
- separate named model volume: `jarvis-model-benchmark-ollama`
- loopback-only host port: `127.0.0.1:11435`
- no mounts from Plex, Seerr, Radarr, Sonarr or the media library
- no `network_mode: host`
- no `privileged`
- no Docker socket mount
- `restart: "no"`

The existing Ollama on port 11434 is therefore left untouched.

## Start

```bash
cd ~/seed-agent/benchmark
docker compose up -d
```

Check:

```bash
curl http://127.0.0.1:11435/api/version
```

## Pull benchmark models

```bash
while read -r model; do
  docker compose exec ollama-bench ollama pull "$model"
done < models.txt
```

## Stop

```bash
docker compose down
```

Models remain in the dedicated benchmark volume. To remove that volume too:

```bash
docker compose down -v
```

Do not run `down -v` unless the downloaded benchmark models should be deleted.
