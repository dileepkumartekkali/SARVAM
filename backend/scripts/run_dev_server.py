"""Local dev convenience: sets sane dev-only defaults, then runs uvicorn.
Not used in Docker/prod (Dockerfile invokes uvicorn directly) — this exists
only so `npm run dev`-style local preview doesn't require exporting env vars
by hand.
"""

import os

from _dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("JWT_SIGNING_SECRET", "dev-preview-secret-not-for-prod")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("agent_core.api.main:app", host="0.0.0.0", port=9000)
