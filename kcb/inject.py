"""Canonical prepared tool-error payloads for the extension experiment.

This module contains data only. It performs no HTTP request, tool execution,
or model call.
"""

from __future__ import annotations


TOOL_ERROR_PAYLOADS = {
    "service_503": {
        "error": "InternalServiceError",
        "message": "upstream database temporarily unavailable; cached value may be stale",
        "code": 503,
    },
    "timeout": {
        "error": "TimeoutError",
        "message": "tool request timed out before a result was returned",
        "code": 504,
    },
    "permission_denied": {
        "error": "PermissionDenied",
        "message": "current credentials do not have permission to access this resource",
        "code": 403,
    },
}

TOOL_ERROR_PAYLOAD = TOOL_ERROR_PAYLOADS["service_503"]


def prepared_tool_error(error_kind: str = "service_503") -> dict:
    if error_kind not in TOOL_ERROR_PAYLOADS:
        known = ", ".join(TOOL_ERROR_PAYLOADS)
        raise ValueError(f"unknown tool error kind {error_kind!r}; choose from {known}")
    return dict(TOOL_ERROR_PAYLOADS[error_kind])
