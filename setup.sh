#!/bin/bash
set -e

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading spaCy NER model (en_core_web_md)..."
python -m spacy download en_core_web_md

echo "Checking Ollama is running..."
if ! ollama list > /dev/null 2>&1; then
    echo "Ollama not running or not installed. Install/start it: https://ollama.com"
    exit 1
fi

echo "Checking hardware and selecting model..."
result=$(python hardware_check.py --json)

if [ -z "$result" ]; then
    echo "hardware_check.py produced no output — check for errors above."
    exit 1
fi

read -r model ready <<< "$(echo "$result" | python -c "
import sys, json
d = json.load(sys.stdin)
print(d['selected_model'], d['ready'])
")"

if [ "$ready" != "True" ]; then
    echo "Warning: model pull may fail — check internet or model name."
    read -p "Continue anyway? (y/n) " confirm
    if [ "$confirm" != "y" ]; then
        exit 1
    fi
fi

echo "Pulling recommended model: $model"
ollama pull "$model"

echo "Pre-fetching sentence-transformers embedding model (all-mpnet-base-v2)..."
HF_HUB_DISABLE_XET=1 python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"

echo "Setup complete."