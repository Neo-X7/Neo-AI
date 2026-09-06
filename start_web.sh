#!/bin/bash
cd "$(dirname "$0")"
source /home/yorichii/mlenv/bin/activate

if [ ! -f cert.pem ] || [ ! -f key.pem ]; then
    echo "Generating self-signed HTTPS cert (first run only)..."
    openssl req -x509 -newkey rsa:2048 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"
fi

echo "Starting Neo web UI at https://127.0.0.1:8000"
python3 -m uvicorn server:app --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
