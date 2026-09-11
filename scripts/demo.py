"""Start the local demo: python scripts/demo.py [--prepare]."""

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Ingerir, transformar e indexar antes de abrir a interface",
    )
    args = parser.parse_args()
    python = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")

    def run(*command):
        subprocess.run([str(c) for c in command], cwd=ROOT, check=True)

    if not python.exists():
        run(sys.executable, "-m", "venv", ROOT / ".venv")
    run(python, "-m", "pip", "install", "-e", ".[web,openai]")
    run("docker", "compose", "up", "-d", "qdrant", "phoenix")
    for attempt in range(60):
        try:
            with urllib.request.urlopen("http://localhost:6333/readyz", timeout=2):
                break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 59:
                raise SystemExit("Qdrant não ficou pronto. Confira o Docker Desktop.") from None
            time.sleep(1)
    if args.prepare:
        run(python, "-m", "src.pipeline", "run-all")
    run(python, "-m", "src.web")


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise SystemExit(f"Não foi possível iniciar a demonstração: {error}") from error
