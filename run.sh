#!/usr/bin/env bash
set -e

# Charger les variables d'environnement si un fichier .env existe
[ -f .env ] && source .env

# Si on est en local (présence de .venv), activer l'environnement
if [ -d ".venv" ]; then
    echo "Running locally with .venv"
    source .venv/bin/activate
    exec python -m uvicorn src.api.main:app --reload --port 5001
else
    echo "Running in container (using uv)"
    # Ici on utilise uv run car dans ton Dockerfile tu as installé avec uv
    exec uv run --no-dev --no-sync python -m uvicorn src.api.main:app --host 0.0.0.0 --port 5001
fi