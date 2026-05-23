#!/bin/bash
set -e

cd "$(dirname "$0")"

if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example — edit it to configure your vLLM URL"
fi

python -m app.main
