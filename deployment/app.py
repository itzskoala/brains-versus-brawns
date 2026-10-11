"""Entry point for the Gradio-SDK Hugging Face Space. Runs the FastAPI
backend (api/main.py) in a background thread on 127.0.0.1:8000, waits for
it to finish loading data/models, then launches the Gradio UI
(frontend/gradio.py) in the main thread - Space SDKs run a single
container/process, so both halves of this normally-two-process app
(see frontend/gradio.py's own docstring) have to share one here.
"""
import os
import threading
import time
import urllib.request

os.environ.setdefault("UFC_ELO_API_URL", "http://127.0.0.1:8000")

import uvicorn

from api.main import app as fastapi_app


def _run_api():
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")


threading.Thread(target=_run_api, daemon=True).start()

for _ in range(120):
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/divisions", timeout=2)
        break
    except Exception:
        time.sleep(1)

from frontend.gradio import demo

if __name__ == "__main__":
    demo.launch()
