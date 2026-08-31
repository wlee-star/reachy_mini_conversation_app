"""Strip secrets from dashboard output."""

from __future__ import annotations
import re


_SECRET_KEY_RE = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|authorization|bearer|credential)",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>(?:token|secret|password|passwd|api[_-]?key|authorization|bearer)"
    r"\s*[=:]\s*)(?P<value>\S+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(Bearer\s+)\S+", re.IGNORECASE)
_ENV_LINE_RE = re.compile(
    r"^(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$",
)


def is_secret_key(name: str) -> bool:
    """Return whether an environment or field name should be masked."""
    return _SECRET_KEY_RE.search(name or "") is not None


def mask_secret(value: str | None) -> str:
    """Return a non-reversible configured/empty marker for a secret."""
    if value is None or not str(value).strip():
        return ""
    return "********"


def redact_text(text: str) -> str:
    """Remove credential-like values from a log or command line."""
    redacted = _BEARER_RE.sub(r"\1********", text)
    return _ASSIGNMENT_RE.sub(r"\g<prefix>********", redacted)


def public_env_map(raw: dict[str, str]) -> dict[str, str]:
    """Return env values safe to show, with secrets masked."""
    visible: dict[str, str] = {}
    for key, value in raw.items():
        if is_secret_key(key):
            visible[key] = "configured" if value.strip() else "not configured"
        else:
            visible[key] = value
    return visible


def parse_env_file(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines from a dotenv file, ignoring comments."""
    parsed: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _ENV_LINE_RE.match(line)
        if match is None:
            continue
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        parsed[match.group("key")] = value
    return parsed
