#!/bin/bash

echo "Starting RAG Project API..."

if [ ! -f .env ]; then
    echo "Creating .env file from example..."
    cp .env.example .env
fi

pip install -e .

python main.py