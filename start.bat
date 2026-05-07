@echo off

echo Starting RAG Project API...

if not exist .env (
    echo Creating .env file from example...
    copy .env.example .env
)

pip install -e .

python main.py