"""Loads versioned system-prompt template files from backend/prompts/.

Templates are files, not inline Python strings, so they can be edited/A-B
tested without a code deploy (agent_system_prompt.md §4). Only the version
identifier (e.g. "text_mode_system.v1") is logged into turn_trace — never the
full prompt text.

`prompts/` lives INSIDE `backend/` (a sibling of `agent_core/`), not at the
monorepo root — a real production bug, found live: the original layout had
it one level above `backend/`, and Render's Docker build (root directory set
to `backend/`) can only COPY files inside its own build context. The
Dockerfile never had `prompts/` available to copy at all, so every /chat
call crashed with FileNotFoundError. Keeping it inside `backend/` means any
Docker build rooted at `backend/` — Render's or otherwise — always has it.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from ..supervisor.state import Mode

# backend/agent_core/agents/prompt_templates.py -> backend/ is 2 parents up.
_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

_MODE_TO_STEM = {
    Mode.TEXT_TO_TEXT: "text_mode_system",
    Mode.SPEECH_TO_TEXT: "text_mode_system",
    Mode.TEXT_TO_SPEECH: "voice_mode_system",
    Mode.SPEECH_TO_SPEECH: "voice_mode_system",
}


def _prompts_dir() -> Path:
    override = os.environ.get("PROMPTS_DIR")
    return Path(override) if override else _DEFAULT_PROMPTS_DIR


@lru_cache(maxsize=None)
def _read_template(stem: str, version: str) -> str:
    path = _prompts_dir() / f"{stem}.{version}.txt"
    return path.read_text(encoding="utf-8")


def load_template(mode: Mode, *, version: str = "v1") -> tuple[str, str]:
    """Returns (raw_template_text, version_id). Substitution happens in task_agent."""
    stem = _MODE_TO_STEM[mode]
    return _read_template(stem, version), f"{stem}.{version}"
