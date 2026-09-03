#!/usr/bin/env python
"""Fast-path reef tank status queries from Apex /status or the local cache."""

import os
import json
import logging
from typing import Any

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.hermes_client import reef_cache_age_seconds, reef_cache_status_for_age
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


CACHE_PATH = os.path.expanduser("~/reef-monitor/reef_cache.json")
logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 5.0
_FOCUS_PROBE_NAMES = ("Tmp", "pH", "ORP", "FS100", "LLSATO")


def _status_url() -> str | None:
    raw = (config.APEX_STATUS_URL or "").strip()
    return raw or None


def _normalize_probes(raw: object) -> dict[str, Any]:
    if isinstance(raw, dict):
        probes: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                probes[str(key)] = value
            else:
                probes[str(key)] = {"value": value, "type": None, "status": "ok", "note": ""}
        return probes
    if not isinstance(raw, list):
        return {}
    probes = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        probes[name.strip()] = {
            "value": item.get("value"),
            "type": item.get("type"),
            "status": item.get("status") or "ok",
            "note": item.get("note") or "",
        }
    return probes


def _controller_name(raw: object) -> object:
    if isinstance(raw, dict):
        hostname = raw.get("hostname")
        return hostname if hostname is not None else raw
    return raw


def focus_probe_values(probes: dict[str, Any]) -> dict[str, object]:
    """Return the raw Tmp/pH/ORP/FS100/LLSATO values, or None when a probe is absent."""
    by_lower = {key.lower(): value for key, value in probes.items()}
    summary: dict[str, object] = {}
    for name in _FOCUS_PROBE_NAMES:
        item = by_lower.get(name.lower())
        if isinstance(item, dict):
            summary[name] = item.get("value")
        else:
            summary[name] = item
    return summary


def _log_reef_snapshot(
    snapshot: dict[str, Any],
    *,
    url: str | None = None,
    http_status: int | None = None,
    invoked_by: str,
) -> None:
    probes_raw = snapshot.get("probes")
    probes = probes_raw if isinstance(probes_raw, dict) else {}
    logger.info(
        "[APEX] %s source=%s url=%s http_status=%s cached_at=%s focus_probes=%s ato=%s",
        invoked_by,
        snapshot.get("source"),
        url,
        http_status,
        snapshot.get("cached_at"),
        focus_probe_values(probes),
        snapshot.get("ato"),
    )


def _ato_payload(payload: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    ato = payload.get("ato")
    if isinstance(ato, dict):
        return {str(key): value for key, value in ato.items()}
    llsato = probes.get("LLSATO")
    if isinstance(llsato, dict):
        return {"llsato": llsato.get("value")}
    return {}


def _reading_timestamp(payload: dict[str, Any], controller: object) -> str | None:
    if isinstance(controller, dict):
        controller_date = controller.get("date")
        if isinstance(controller_date, str) and controller_date.strip():
            return controller_date.strip()
    fetched_at = payload.get("fetched_at")
    if isinstance(fetched_at, str) and fetched_at.strip():
        return fetched_at.strip()
    return None


def _snapshot_from_live_payload(payload: dict[str, Any]) -> dict[str, Any]:
    probes = _normalize_probes(payload.get("probes"))
    controller = payload.get("controller")
    outlets = payload.get("outlets")
    return {
        "probes": probes,
        "ato": _ato_payload(payload, probes),
        "alarms": payload.get("alarms") if isinstance(payload.get("alarms"), dict) else {},
        "alerts": payload.get("alerts") if isinstance(payload.get("alerts"), list) else [],
        "outlets": outlets if isinstance(outlets, list) else [],
        "controller": _controller_name(controller),
        "cached_at": _reading_timestamp(payload, controller),
        "age_seconds": 0,
        "cache_age_seconds": 0,
        "stale": False,
        "source": "live",
        "status": "success",
    }


async def _fetch_live_status(url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as http_client:
            response = await http_client.get(url)
            logger.info("[APEX] GET %s -> HTTP %s", url, response.status_code)
            response.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("[APEX] GET %s timed out", url)
        return None, "Apex status is currently unavailable."
    except httpx.HTTPStatusError as exc:
        logger.warning("[APEX] GET %s rejected HTTP %s", url, exc.response.status_code)
        return None, "Apex status rejected the request."
    except httpx.RequestError as exc:
        logger.warning("[APEX] GET %s failed: %s", url, exc)
        return None, "Apex status is currently unavailable."

    try:
        payload: object = response.json()
    except ValueError:
        logger.warning("[APEX] GET %s returned malformed JSON", url)
        return None, "Apex status returned an unexpected response."
    if not isinstance(payload, dict):
        logger.warning("[APEX] GET %s payload must be a JSON object", url)
        return None, "Apex status returned an unexpected response."
    snapshot = _snapshot_from_live_payload(payload)
    _log_reef_snapshot(snapshot, url=url, http_status=response.status_code, invoked_by="live_status")
    return snapshot, None


def _load_cache() -> tuple[dict[str, Any] | None, str | None]:
    if not os.path.exists(CACHE_PATH):
        return None, "Reef cache not found. Ensure reef_cache.py cron is running."
    try:
        with open(CACHE_PATH, encoding="utf-8") as cache_file:
            raw: object = json.load(cache_file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read reef cache %s: %s", CACHE_PATH, exc)
        return None, "Apex reef cache could not be read."
    if not isinstance(raw, dict):
        logger.warning("Reef cache %s must contain a JSON object", CACHE_PATH)
        return None, "Apex reef cache has an unexpected format."
    snapshot = {str(key): value for key, value in raw.items()}
    if "probes" in snapshot:
        snapshot["probes"] = _normalize_probes(snapshot.get("probes"))
    age_seconds = reef_cache_age_seconds(
        generated_at=snapshot.get("cached_at") or snapshot.get("fetched_at"),
        data_timestamp=snapshot.get("cached_at"),
        path=CACHE_PATH,
    )
    snapshot["source"] = "cache"
    snapshot["stale"] = True
    snapshot["age_seconds"] = age_seconds
    snapshot["cache_age_seconds"] = age_seconds
    snapshot["status"] = reef_cache_status_for_age(age_seconds)
    logger.info(
        "[REEF] cached Apex snapshot found path=%s age_seconds=%s status=%s",
        CACHE_PATH,
        age_seconds,
        snapshot["status"],
    )
    logger.info("[REEF] marking response stale=true source=cache")
    _log_reef_snapshot(snapshot, invoked_by="reef_cache")
    return snapshot, None


def snapshot_provenance(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return status/stale/source/cache_age_seconds for a Reef snapshot."""
    stale = bool(snapshot.get("stale", False))
    source_raw = snapshot.get("source")
    source = source_raw if source_raw in {"live", "cache", "none"} else ("cache" if stale else "live")
    age_raw = snapshot.get("cache_age_seconds", snapshot.get("age_seconds"))
    age = float(age_raw) if isinstance(age_raw, (int, float)) else None
    status_raw = snapshot.get("status")
    if status_raw in {"success", "degraded", "stale", "error"}:
        status = status_raw
    elif source == "live" and not stale:
        status = "success"
    else:
        status = reef_cache_status_for_age(age)
    live = source == "live" and not stale and status == "success"
    return {
        "status": status,
        "stale": stale,
        "source": source,
        "cache_age_seconds": age,
        "live": live,
        "degraded": not live,
    }


async def _load_reef_snapshot() -> tuple[dict[str, Any] | None, str | None]:
    live_error: str | None = None
    url = _status_url()
    if url is not None:
        snapshot, live_error = await _fetch_live_status(url)
        if snapshot is not None:
            logger.info("[REEF] Apex live request succeeded source=live")
            return snapshot, None
        logger.warning("[REEF] live Apex request failed (%s); checking Reef cache", live_error)
    else:
        logger.info("[REEF] APEX_STATUS_URL unset; reading Reef cache")
    cache, cache_error = _load_cache()
    if cache is None:
        logger.warning("[REEF] no usable cache; returning error")
        return None, live_error or cache_error
    return cache, None


class ReefStatus(Tool):
    """Fast-path reef tank status from Apex /status or the local cache."""

    name = "reef_status"
    description = (
        "Get current reef tank snapshot (temperature, pH, ORP, ATO, outlets, alarms) "
        "from the local Apex /status URL when APEX_STATUS_URL is set, otherwise reef_cache.json. "
        "Use for live current numbers. Not for historical trends, threading, or ATO history "
        "— use ask_hermes for those. Returns structured data. "
        "source=live and stale=false is current Apex data. source=cache and stale=true is cached: "
        "use the numbers but tell the user they are cached/stale, not live."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of specific metrics to include. Defaults to all.",
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
        """Return the current reef snapshot in the legacy response shape."""
        include = kwargs.get("include")
        if include is not None and not isinstance(include, list):
            return {"error": "include must be a list of metric names"}

        logger.info("[APEX] reef_status tool invoked")
        cache, error = await _load_reef_snapshot()
        if cache is None:
            logger.warning("[REEF] returning error source=none")
            return {
                "error": error or "Reef cache not found. Ensure reef_cache.py cron is running.",
                "status": "error",
                "stale": True,
                "source": "none",
                "live": False,
                "degraded": True,
            }

        probes_raw = cache.get("probes", {})
        probes = probes_raw if isinstance(probes_raw, dict) else {}
        if include:
            probes = {key: value for key, value in probes.items() if key in include}

        results = {}
        for metric, data in probes.items():
            if not isinstance(data, dict):
                continue
            results[metric] = {
                "value": data.get("value"),
                "type": data.get("type"),
                "status": data.get("status"),
                "note": data.get("note"),
            }

        provenance = snapshot_provenance(cache)
        result = {
            "reef_status": {
                "probes": results,
                "ato": cache.get("ato", {}),
                "alarms": cache.get("alarms", {}),
                "alerts": cache.get("alerts", []),
                "outlets": cache.get("outlets", []),
                "controller": cache.get("controller"),
                "cached_at": cache.get("cached_at"),
                "age_seconds": cache.get("age_seconds"),
                "stale": provenance["stale"],
            },
            **provenance,
        }
        logger.info("[APEX] reef_status tool result=%s", result)
        return result
