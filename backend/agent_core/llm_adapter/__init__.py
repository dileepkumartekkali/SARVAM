"""Provider-agnostic LLM access: the adapter interface, the fallback router,
and env-driven provider assembly.

Business logic should only ever import `build_router_from_env` and call
`LLMRouter.complete_with_fallback` / `stream_with_fallback` — never a
concrete provider class, so it never branches on which provider is active.
"""

from .base import LLMProvider, LLMProviderError, LLMRouter
from .config import build_router_from_env, provider_order

__all__ = [
    "LLMProvider",
    "LLMProviderError",
    "LLMRouter",
    "build_router_from_env",
    "provider_order",
]
