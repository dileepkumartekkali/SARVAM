"""Tiny stdlib-only .env loader for the two local dev-launch scripts. Not a
new dependency (python-dotenv) for something this small — just KEY=VALUE
line parsing. Real exported shell env vars still win (setdefault, not
overwrite).
"""

from __future__ import annotations

import os
from pathlib import Path

# backend/scripts/_dotenv.py -> repo root is 2 parents up.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv() -> None:
    if not _ENV_FILE.exists():
        return
    for line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
