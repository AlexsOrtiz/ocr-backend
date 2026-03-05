#!/usr/bin/env bash
# Levanta la API en modo desarrollo sin reiniciar por cambios en venv
cd "$(dirname "$0")"
uvicorn app.main:app --reload --reload-exclude 'venv/*' "$@"
