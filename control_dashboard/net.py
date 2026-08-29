"""HTTP and TCP probes used by health checks."""

from __future__ import annotations
import json
import socket
import urllib.error
import urllib.request
from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class HttpResult:
    """Outcome of one HTTP probe."""

    ok: bool
    status_code: int | None
    body: str
    latency_ms: float
    error: str | None


def host_resolves(host: str) -> bool:
    """Return whether a hostname can be resolved."""
    try:
        socket.getaddrinfo(host, None)
    except OSError:
        return False
    return True


def port_open(host: str, port: int, timeout_s: float = 1.0) -> bool:
    """Return whether a TCP port accepts a connection."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def http_request(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout_s: float = 4.0,
) -> HttpResult:
    """Perform one HTTP request and return status, body, and latency."""
    import time

    data = None
    request_headers = dict(headers or {})
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, method=method, headers=request_headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = response.read().decode("utf-8", errors="replace")
            latency_ms = (time.perf_counter() - started) * 1000.0
            return HttpResult(True, int(response.status), body, latency_ms, None)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HttpResult(False, int(exc.code), body, latency_ms, f"HTTP {exc.code}")
    except urllib.error.URLError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HttpResult(False, None, "", latency_ms, str(exc.reason) if exc.reason else str(exc))
    except TimeoutError:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HttpResult(False, None, "", latency_ms, "timed out")
    except OSError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return HttpResult(False, None, "", latency_ms, str(exc))


def json_payload(result: HttpResult) -> dict[str, Any] | None:
    """Parse a JSON object body, or return None."""
    if not result.body.strip():
        return None
    try:
        payload: object = json.loads(result.body)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
