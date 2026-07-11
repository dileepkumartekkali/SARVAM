"""Shared HTTP status → retriable mapping, used by every provider adapter.

429 (rate limit) and 5xx (provider-side fault) are retriable — the router may
fall through to the next provider. Everything else (401/403 auth, 400/404
malformed request) is a fault the caller must see, not paper over.
"""


def status_retriable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500
