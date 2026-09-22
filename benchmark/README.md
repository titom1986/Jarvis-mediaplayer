# JARVIS model benchmark

Instance Ollama isolée pour comparer de petits modèles sans toucher au JARVIS/Ollama existant.

- port benchmark : 127.0.0.1:11435
- volume dédié
- aucun montage Plex/Seerr/Radarr/Sonarr
- aucun accès au Docker socket
- aucun host networking

## Démarrage (Seedhost legacy Compose)

```bash
cd ~/seed-agent/benchmark
docker-compose up -d
```

## Télécharger les modèles

```bash
while read -r model; do docker-compose exec ollama-bench ollama pull "$model"; done < models.txt
```

## Benchmark

Le harness n'appelle aucun vrai service média. Il teste uniquement la compréhension et le tool calling avec des schémas déterministes.

```bash
python3 benchmark.py
```

Colonnes : modèle, scénario, succès (0/1), temps total, tokens prompt, tokens générés, tokens/s.

## Arrêt

```bash
docker-compose down
```

Le volume des modèles est conservé. `docker-compose down -v` le supprime aussi.
