"""Local dev convenience for the Speech Gateway — mirrors run_dev_server.py.
Not used in Docker/prod (Dockerfile.gateway invokes uvicorn directly).

No SARVAM_API_KEY/AZURE_SPEECH_KEY set here on purpose: the gateway boots
fine without them (keys are only read when a call is actually made) — real
STT/TTS calls will fail until real keys are exported in this shell first.
"""

import os

from _dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("CORS_ALLOWED_ORIGINS", "http://localhost:3000")

import uvicorn  # noqa: E402

if __name__ == "__main__":
    uvicorn.run("agent_core.speech_gateway.main:gateway_app", host="0.0.0.0", port=9100)
