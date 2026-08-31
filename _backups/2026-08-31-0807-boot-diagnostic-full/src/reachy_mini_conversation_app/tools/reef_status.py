#!/usr/bin/env python
"""Fast-path reef tank status queries from Apex /status or the local cache."""

import os
import json
import logging
from typing import Any

import httpx

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


CACHE_PATH = os.path.expanduser("~/reef-monitor/reef_cache.json")
logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 5.0


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


def _ato_payload(payload: dict[str, Any], probes: dict[str, Any]) -> dict[str, Any]:
    ato = payload.get("ato")
    if isinstance(ato, dict):
        return {str(key): value for key, value in ato.items()}
    llsato = probes.get("LLSATO")
    if isinstance(llsato, dict):
        return {"llsato": llsato.get("value")}
    return {}


def _snapshot_from_live_payload(payload: dict[str, Any]) -> dict[str, Any]:
    probes = _normalize_probes(payload.get("probes"))
    controller = payload.get("controller")
    cached_at = controller.get("date") if isinstance(controller, dict) else None
    outlets = payload.get("outlets")
    return {
        "probes": probes,
        "ato": _ato_payload(payload, probes),
        "alarms": payload.get("alarms") if isinstance(payload.get("alarms"), dict) else {},
        "alerts": payload.get("alerts") if isinstance(payload.get("alerts"), list) else [],
        "outlets": outlets if isinstance(outlets, list) else [],
        "controller": _controller_name(controller),
        "cached_at": cached_at,
        "age_seconds": 0,
        "stale": False,
        "source": "apex_status_http",
    }


async def _fetch_live_status(url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S) as http_client:
            response = await http_client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException:
        logger.warning("Apex status request timed out")
        return None, "Apex status is currently unavailable."
    except httpx.HTTPStatusError as exc:
        logger.warning("Apex status HTTP %s for GET %s", exc.response.status_code, url)
        return None, "Apex status rejected the request."
    except httpx.RequestError as exc:
        logger.warning("Apex status request failed: %s", exc)
        return None, "Apex status is currently unavailable."

    try:
        payload: object = response.json()
    except ValueError:
        logger.warning("Apex status returned malformed JSON")
        return None, "Apex status returned an unexpected response."
    if not isinstance(payload, dict):
        logger.warning("Apex status payload must be a JSON object")
        return None, "Apex status returned an unexpected response."
    logger.info("[APEX] live status read succeeded")
    return _snapshot_from_live_payload(payload), None


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
    snapshot.setdefault("source", "reef_cache_direct")
    if "probes" in snapshot:
        snapshot["probes"] = _normalize_probes(snapshot.get("probes"))
    return snapshot, None


async def _load_reef_snapshot() -> tuple[dict[str, Any] | None, str | None]:
    url = _status_url()
    if url is not None:
        snapshot, error = await _fetch_live_status(url)
        if snapshot is not None:
            return snapshot, None
        logger.warning("Apex live status failed (%s); trying reef cache", error)
    return _load_cache()


class ReefStatus(Tool):
    """Fast-path reef tank status from Apex /status or the local cache."""

    name = "reef_status"
    description = (
        "Get current reef tank snapshot (temperature, pH, ORP, ATO, outlets, alarms) "
        "from the local Apex /status URL when APEX_STATUS_URL is set, otherwise reef_cache.json. "
        "Use for live current numbers. Not for historical trends, threading, or ATO history "
        "— use ask_hermes for those. Returns structured data."
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

        cache, error = await _load_reef_snapshot()
        if cache is None:
            return {"error": error or "Reef cache not found. Ensure reef_cache.py cron is running."}

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

        source = cache.get("source")
        return {
            "reef_status": {
                "probes": results,
                "ato": cache.get("ato", {}),
                "alarms": cache.get("alarms", {}),
                "alerts": cache.get("alerts", []),
                "outlets": cache.get("outlets", []),
                "controller": cache.get("controller"),
                "cached_at": cache.get("cached_at"),
                "age_seconds": cache.get("age_seconds"),
                "stale": cache.get("stale", False),
            },
            "source": source if isinstance(source, str) else "reef_cache_direct",
        }
