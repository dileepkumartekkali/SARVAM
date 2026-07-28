"""Shared HTTP status → retriable mapping, used by every provider adapter.

429 (rate limit) and 5xx (provider-side fault) are retriable — the router may
fall through to the next provider. Everything else (401/403 auth, 400/404
malformed request) is a fault the caller must see, not paper over.
"""


def status_retriable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


# Live-confirmed real bug: Groq returned 400 "Failed to call a function.
# Please adjust your prompt." with error.code == "tool_use_failed" when
# Llama-3.3-70b generated malformed native function-call syntax
# (`<function=search_company_knowledge {...}</function>` instead of a real
# tool_calls field) for a RAG-triggering query. A plain 400 is correctly
# treated as "the caller's own fault, don't paper over it" for a genuinely
# malformed request -- but this is the MODEL's own generation failing, not
# anything wrong with what was sent, the same class of transient fault a
# 5xx represents. Without this, the whole turn died with "I'm having
# trouble getting an answer right now" instead of falling back to the next
# configured provider.
_RETRIABLE_400_ERROR_CODES = frozenset({"tool_use_failed"})


def tool_call_error_retriable(status_code: int, response_text: str) -> bool:
    if status_retriable(status_code):
        return True
    if status_code != 400:
        return False
    return any(code in response_text for code in _RETRIABLE_400_ERROR_CODES)
