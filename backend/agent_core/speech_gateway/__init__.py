"""Speech Gateway — the only service that talks to Sarvam (and the Azure
fallback). See docs/speech_to_speech_implementation_plan.md §2 for the three
reasons this is a separate service from the main backend: key custody, a
validation chokepoint, and (Phase 5) barge-in coordination.

Phase 4 scope: Speech→Text and Text→Speech as independent capabilities.
Full-duplex Speech→Speech with barge-in is Phase 5 — not built here.
"""
