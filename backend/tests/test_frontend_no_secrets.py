"""CI check: grep the frontend source for key/secret patterns (S2S plan §6 —
Sarvam/LLM keys must live only in backend/Speech Gateway env, zero client
exposure). Scans source now; once a real build step exists, point this at
the built bundle too — a leaked env var can appear in *either* place.
"""

import re
from pathlib import Path

# Env var names that must never be read by frontend code, plus generic
# provider key-shape patterns (OpenAI/Anthropic/Google/Grok-style prefixes).
_FORBIDDEN_PATTERNS = [
    re.compile(r"SARVAM_API_KEY"),
    re.compile(r"AZURE_SPEECH_KEY"),
    re.compile(r"JWT_SIGNING_SECRET"),
    re.compile(r"GROK_API_KEY"),
    re.compile(r"OPENAI_API_KEY"),
    re.compile(r"ANTHROPIC_API_KEY"),
    re.compile(r"GEMINI_API_KEY"),
    re.compile(r"\bsk-[A-Za-z0-9]{16,}"),  # OpenAI/Anthropic-shaped key literal
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),  # Google API key literal
]

_FRONTEND_ROOT = Path(__file__).resolve().parents[2] / "frontend"
_SCANNED_SUFFIXES = {".js", ".jsx", ".ts", ".tsx", ".json", ".html"}


def _frontend_files():
    if not _FRONTEND_ROOT.exists():
        return []
    return [
        p
        for p in _FRONTEND_ROOT.rglob("*")
        if p.is_file() and p.suffix in _SCANNED_SUFFIXES and "node_modules" not in p.parts
    ]


def test_frontend_source_contains_no_secret_patterns():
    offenders = []
    for path in _frontend_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(_FRONTEND_ROOT)}: matched {pattern.pattern}")

    assert offenders == [], "secret-shaped pattern found in frontend source:\n" + "\n".join(offenders)


def test_frontend_root_actually_scanned():
    """Guards against the check silently scanning zero files (e.g. a path
    typo) and passing trivially."""
    assert _FRONTEND_ROOT.exists()
    assert len(_frontend_files()) > 0
