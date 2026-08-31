"""Run the local interface: `python -m src.web`."""

import logging

import uvicorn

from src.web.app import build_default_app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    print("\n  Atas do Copom -> http://localhost:8000\n")
    uvicorn.run(build_default_app(), host="127.0.0.1", port=8000, log_level="warning")
