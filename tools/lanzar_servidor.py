#!/usr/bin/env python3
"""Arranca Uvicorn desligado de la terminal (doble fork)."""

import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
LOG = Path("/tmp/opencode/uvicorn.log")


def main() -> None:
    puerto = sys.argv[1] if len(sys.argv) > 1 else "8080"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(f"lanzado (pid {pid}) -> http://localhost:{puerto}")
        return
    os.setsid()
    pid2 = os.fork()
    if pid2:
        os._exit(0)
    log = os.open(LOG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(log, 1)
    os.dup2(log, 2)
    os.chdir(RAIZ)
    exe = str(RAIZ / ".venv" / "bin" / "uvicorn")
    os.execv(exe, [exe, "src.app:app", "--host", "127.0.0.1", "--port", puerto])


if __name__ == "__main__":
    main()
