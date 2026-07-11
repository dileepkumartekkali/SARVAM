"""LangGraph supervisor: session state, graph wiring, checkpointer selection.

The graph itself and checkpointer selection are later phases. This module
exports only the typed session state the graph threads through its nodes.
"""

from .state import Mode, SessionState

__all__ = ["Mode", "SessionState"]
