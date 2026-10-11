#!/bin/sh
set -e

uvicorn api.main:app --host 127.0.0.1 --port 8000 &

python - <<'PY'
import time
import urllib.request

for _ in range(60):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/divisions", timeout=2)
        break
    except Exception:
        time.sleep(1)
PY

export UFC_ELO_API_URL=http://127.0.0.1:8000
exec python -m frontend.gradio
