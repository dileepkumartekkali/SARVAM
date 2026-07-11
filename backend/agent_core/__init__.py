"""MAAV / Vaani agent core.

The package is layered so that provider- and transport-specific concerns never
leak into the reasoning core:

    llm_adapter/  provider-agnostic LLM access (router + interface)
    tools/        permission-gated external actions
    agents/       task + language reasoning units
    supervisor/   LangGraph wiring, session state, checkpointer selection
    speech/       Sarvam STT/TTS proxy + barge-in state machine (Phase 3)
    api/          FastAPI + WebSocket transport

See docs/agent_system_prompt.md and docs/speech_to_speech_implementation_plan.md
for the binding spec these modules implement.
"""
