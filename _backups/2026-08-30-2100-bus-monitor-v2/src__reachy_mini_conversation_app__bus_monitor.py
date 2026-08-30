"""Background Route 311 monitoring against the existing Home Assistant sensor.

The live Home Assistant 311 entity remains the source of truth. This module
does not add a transport API. Recurring polls are deterministic and do not
go through the LLM.
"""

import os
import re
import json
import time
import uuid
import asyncio
import logging
from typing import Any, Callable, Awaitable
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, dataclass
from urllib.parse import quote
from collections.abc import Mapping

import httpx

from reachy_mini_conversation_app.tools.home_assistant import (
    _REQUEST_TIMEOUT_S,
    HomeAssistantConfigError,
    _first_string,
    _request_json,
    _bus_entity_id,
    _arrival_details,
    _minutes_from_value,
    _extract_bus_arrival,
    _home_assistant_config,
)


logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_ROUTE = "311"
DEFAULT_PREPARATION_MINUTES = 15
DEFAULT_URGENT_MINUTES = 5
STALE_AFTER_S = 180.0
QUERY_CACHE_TTL_S = 15.0
DEFAULT_POLL_S = 45.0
CLOSE_POLL_S = 15.0
CLOSE_POLL_WITHIN_MINUTES = 8
MAX_MONITOR_AGE_S = 2 * 60 * 60
MAX_HA_FAILURE_BACKOFF_S = 60.0
MONITOR_FILENAME = "bus_monitors.v1.json"
_BUS_QUERY_RE = re.compile(
    r"\b(?:311|three[\s-]?eleven|three[\s-]?one[\s-]?one|next bus|the bus|"
    r"bus (?:is |coming|arrival|arriving)|when(?:'s| is) the bus|"
    r"let me know when.{0,40}bus|monitor (?:the |that )?bus|watch (?:the |that )?bus|"
    r"bus reminder)\b",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"\b(?:where(?:s| is) (?:it|that|my bus)|whats? the status|status of (?:it|the bus)|"
    r"how far (?:is it|away))\b",
    re.IGNORECASE,
)
_TEN_MINUTE_RE = re.compile(r"\b10[ -]?minute", re.IGNORECASE)
_CANCEL_RE = re.compile(
    r"\b(?:cancel(?: the)?(?: bus)?(?: reminder| monitor)?|stop watching(?: the bus)?|"
    r"don'?t monitor|do not monitor|not anymore)\b",
    re.IGNORECASE,
)
_CANCEL_THAT_RE = re.compile(r"\b(?:cancel that|never mind|forget (?:it|that)|stop that)\b", re.IGNORECASE)
_CONFIRM_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|please|ok|okay|go ahead|do it)\b", re.IGNORECASE)
_QUESTION_PREFIX = re.compile(r"^(?:is|are|was|whats|what is|did you|have you)\b", re.IGNORECASE)

NotifyFn = Callable[[str], Awaitable[None]]


@dataclass(frozen=True)
class BusArrival:
    """One upcoming service from the Home Assistant 311 sensor."""

    minutes: int
    entity_id: str
    route: str | None = None
    destination: str | None = None
    eta_display: str | None = None
    realtime: bool = False
    service_id: str | None = None
    stop: str | None = None


@dataclass
class LiveBusSnapshot:
    """Latest live arrivals plus query timing."""

    arrivals: list[BusArrival]
    entity_id: str
    last_updated_s: float | None
    data_age_s: float | None
    stale: bool
    ha_query_latency_s: float
    error: str | None = None
    fetched_at: float = 0.0

    @property
    def next_arrival(self) -> BusArrival | None:
        """Return the earliest upcoming service, if any."""
        return self.arrivals[0] if self.arrivals else None

    @property
    def following_arrival(self) -> BusArrival | None:
        """Return the second upcoming service, if any."""
        return self.arrivals[1] if len(self.arrivals) > 1 else None


@dataclass
class BusMonitorState:
    """Persisted state for one active Route 311 watch."""

    monitor_id: str
    entity_id: str
    route: str | None
    stop: str | None
    service_id: str | None
    created_at: float
    latest_live_eta: int | None
    preparation_threshold: int
    urgent_threshold: int
    status: str
    last_check: float | None
    preparation_alert_sent: bool
    urgent_alert_sent: bool
    arrival_alert_sent: bool = False
    destination: str | None = None


@dataclass(frozen=True)
class BusUtterance:
    """A parsed spoken request related to Route 311 monitoring."""

    action: str
    preparation_threshold: int = DEFAULT_PREPARATION_MINUTES


def monitor_path_for_instance(instance_path: str | Path | None = None) -> Path:
    """Return the bus-monitor JSON path for this app instance."""
    if instance_path is not None:
        return Path(instance_path).expanduser() / MONITOR_FILENAME

    data_home = os.getenv("XDG_DATA_HOME")
    data_root = Path(data_home).expanduser() if data_home else Path.home() / ".local" / "share"
    return data_root / "reachy_mini_conversation_app" / MONITOR_FILENAME


def _service_id_from_source(source: Mapping[str, Any]) -> str | None:
    return _first_string(
        dict(source),
        ("trip_id", "tripId", "vehicle_id", "vehicleId", "departure_id", "id"),
    )


def _parse_ha_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _arrival_from_details(
    details: dict[str, Any],
    source: Mapping[str, Any],
    attributes: Mapping[str, Any],
) -> BusArrival | None:
    minutes = details.get("minutes")
    if not isinstance(minutes, int):
        # Home Assistant's `or` chain skips integer 0; recover it as a valid ETA.
        minutes = _minutes_from_value(source.get("minutes"))
        if minutes is None:
            minutes = _minutes_from_value(source.get("due_in") or source.get("due_in_minutes"))
        if minutes is None:
            return None
    entity_id = details.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id:
        return None
    route = details.get("route") if isinstance(details.get("route"), str) else None
    destination = details.get("destination") if isinstance(details.get("destination"), str) else None
    eta_display = details.get("eta_display") if isinstance(details.get("eta_display"), str) else None
    stop = _first_string(dict(attributes), ("stop", "stop_name", "friendly_name"))
    if stop is None:
        stop = _first_string(dict(source), ("stop", "stop_name"))
    return BusArrival(
        minutes=minutes,
        entity_id=entity_id,
        route=route or DEFAULT_ROUTE,
        destination=destination,
        eta_display=eta_display,
        realtime=bool(details.get("realtime")),
        service_id=_service_id_from_source(source),
        stop=stop,
    )


def extract_arrivals(payload: object, entity_id: str) -> list[BusArrival]:
    """Parse every upcoming service from a Home Assistant state payload."""
    if not isinstance(payload, dict):
        return []
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}

    arrivals: list[BusArrival] = []
    seen: set[tuple[object, ...]] = set()
    schedule_list_seen = False
    for key in ("arrivals", "next_arrivals", "departures", "next_departures", "services", "buses"):
        candidates = attributes.get(key)
        if not isinstance(candidates, list):
            continue
        schedule_list_seen = True
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            details = _arrival_details(candidate, entity_id, attributes)
            if details is None:
                continue
            arrival = _arrival_from_details(details, candidate, attributes)
            if arrival is None:
                continue
            identity = (arrival.service_id, arrival.minutes, arrival.destination, arrival.eta_display)
            if identity in seen:
                continue
            seen.add(identity)
            arrivals.append(arrival)

    if arrivals:
        arrivals.sort(key=lambda item: item.minutes)
        return arrivals
    if schedule_list_seen:
        return []

    details = _extract_bus_arrival(payload, entity_id)
    if details is None:
        return []
    arrival = _arrival_from_details(details, attributes, attributes)
    return [arrival] if arrival is not None else []


def route_label(arrival: BusArrival | None) -> str:
    """Return a short spoken route name."""
    if arrival is not None and arrival.route:
        return arrival.route
    return DEFAULT_ROUTE


def format_initial_spoken(
    snapshot: LiveBusSnapshot,
    *,
    preparation_threshold: int = DEFAULT_PREPARATION_MINUTES,
    urgent_threshold: int = DEFAULT_URGENT_MINUTES,
) -> str:
    """Build the first live-arrival line, including a monitor offer when useful."""
    if snapshot.error:
        return "I could not read the live 311 arrival from Home Assistant right now."
    arrival = snapshot.next_arrival
    if arrival is None:
        return "I could not find an upcoming 311 in Home Assistant right now."

    route = route_label(arrival)
    if arrival.minutes <= 0:
        return f"The next {route} is due now. Please leave now."
    if arrival.minutes <= 1:
        return f"The next {route} is arriving now. Please leave now."
    if arrival.minutes <= urgent_threshold:
        return f"The next {route} is arriving in about {arrival.minutes} minutes. Please leave now."

    parts = [f"The next {route} is currently about {arrival.minutes} minutes away"]
    if arrival.eta_display:
        parts[0] += f", expected at {arrival.eta_display}"
    following = snapshot.following_arrival
    if following is not None:
        parts.append(f"The following {route_label(following)} is about {following.minutes} minutes away")
    spoken = ". ".join(parts) + "."
    if arrival.minutes <= preparation_threshold:
        spoken = spoken.rstrip(".")
        spoken += f". Would you like me to keep monitoring it and let you know when it's about {urgent_threshold} minutes away?"
        return spoken
    spoken = spoken.rstrip(".")
    spoken += f". Would you like me to monitor it and alert you when it's within {preparation_threshold} minutes?"
    return spoken


def format_preparation_alert(arrival: BusArrival) -> str:
    """Build the one-shot preparation warning."""
    return f"The {route_label(arrival)} is now about {arrival.minutes} minutes away. You have time to get ready."


def format_urgent_alert(arrival: BusArrival) -> str:
    """Build the one-shot leave-now warning."""
    if arrival.minutes <= 1:
        return f"The {route_label(arrival)} is arriving now. Please leave now."
    return f"The {route_label(arrival)} is now about {arrival.minutes} minutes away. Please leave now."


def format_arrival_alert(arrival: BusArrival | None) -> str:
    """Build the one-shot terminal notice for the watched service."""
    route = route_label(arrival)
    if arrival is not None and arrival.minutes <= 0:
        return f"The {route} is due now. Please leave now if you haven't already."
    return f"The {route} has arrived. Please leave now if you haven't already."


def offer_kind(
    minutes: int,
    *,
    preparation_threshold: int = DEFAULT_PREPARATION_MINUTES,
    urgent_threshold: int = DEFAULT_URGENT_MINUTES,
) -> str:
    """Return which confirmation, if any, should be offered after the live report."""
    if minutes <= urgent_threshold:
        return "leave_now"
    if minutes <= preparation_threshold:
        return "offer_urgent"
    return "offer_prepare"


def match_bus_intent(transcript: str, *, pending_offer: bool, monitor_active: bool) -> BusUtterance | None:
    """Parse a spoken bus-monitor request without using the LLM."""
    text = transcript.lower().strip()
    text = text.replace("'", "")
    text = re.sub(r"[.!?,;:]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return None
    threshold = 10 if _TEN_MINUTE_RE.search(text) else DEFAULT_PREPARATION_MINUTES
    if pending_offer or monitor_active:
        if _CANCEL_RE.search(text) or _CANCEL_THAT_RE.search(text):
            return BusUtterance(action="cancel", preparation_threshold=threshold)
    elif _CANCEL_RE.search(text) and _BUS_QUERY_RE.search(text):
        return BusUtterance(action="cancel", preparation_threshold=threshold)
    if pending_offer and _CONFIRM_RE.search(text) and not _QUESTION_PREFIX.search(text):
        return BusUtterance(action="confirm", preparation_threshold=threshold)
    if (pending_offer or monitor_active) and _STATUS_RE.search(text):
        return BusUtterance(action="query", preparation_threshold=threshold)
    if _BUS_QUERY_RE.search(text):
        return BusUtterance(action="query", preparation_threshold=threshold)
    return None


def same_service(monitor: BusMonitorState, arrival: BusArrival) -> bool:
    """Return whether this live arrival is still the monitored service."""
    if monitor.service_id and arrival.service_id:
        return monitor.service_id == arrival.service_id
    if monitor.route and arrival.route and monitor.route != arrival.route:
        return False
    if monitor.destination and arrival.destination and monitor.destination != arrival.destination:
        return False
    if monitor.latest_live_eta is not None and arrival.minutes > monitor.latest_live_eta + 8:
        return False
    return True


def find_monitored_arrival(monitor: BusMonitorState, snapshot: LiveBusSnapshot) -> BusArrival | None:
    """Return the watched service from the live snapshot, if it is still present."""
    for arrival in snapshot.arrivals:
        if same_service(monitor, arrival):
            return arrival
    return None


def evaluate_alerts(monitor: BusMonitorState, arrival: BusArrival) -> str | None:
    """Return the next one-shot alert kind, or None if nothing should be spoken."""
    if arrival.minutes <= 0:
        if not monitor.arrival_alert_sent:
            return "arrival"
        return None
    if not monitor.urgent_alert_sent and arrival.minutes <= monitor.urgent_threshold:
        return "urgent"
    if not monitor.preparation_alert_sent and arrival.minutes <= monitor.preparation_threshold:
        return "preparation"
    return None


async def fetch_live_snapshot() -> LiveBusSnapshot:
    """Read the configured Home Assistant 311 sensor once."""
    started = time.perf_counter()
    try:
        base_url, token = _home_assistant_config()
    except HomeAssistantConfigError:
        return LiveBusSnapshot(
            arrivals=[],
            entity_id=_bus_entity_id({}),
            last_updated_s=None,
            data_age_s=None,
            stale=False,
            ha_query_latency_s=time.perf_counter() - started,
            error="Home Assistant is not configured",
            fetched_at=time.time(),
        )

    entity_id = _bus_entity_id({})
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    url = f"{base_url}/api/states/{quote(entity_id, safe='')}"

    async def _read_state() -> tuple[dict[str, Any] | None, str | None]:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_S, headers=headers) as http_client:
            raw, request_error = await _request_json(http_client, "GET", url)
        if request_error is not None:
            return None, request_error
        if not isinstance(raw, dict):
            return None, "Bus arrival data unavailable"
        return raw, None

    payload, error = await _read_state()
    if error is not None:
        logger.warning("Home Assistant bus query failed; retrying once: %s", error)
        payload, error = await _read_state()

    latency = time.perf_counter() - started
    if error is not None or payload is None:
        logger.warning("Home Assistant bus query failed: %s", error)
        return LiveBusSnapshot(
            arrivals=[],
            entity_id=entity_id,
            last_updated_s=None,
            data_age_s=None,
            stale=False,
            ha_query_latency_s=latency,
            error=error or "Bus arrival data unavailable",
            fetched_at=time.time(),
        )

    last_updated_s = _parse_ha_timestamp(payload.get("last_updated") or payload.get("last_changed"))
    data_age_s = time.time() - last_updated_s if last_updated_s is not None else None
    stale = data_age_s is not None and data_age_s > STALE_AFTER_S
    if stale:
        logger.warning("Home Assistant bus data is stale age=%.0fs entity=%s", data_age_s or 0.0, entity_id)
        retry_payload, retry_error = await _read_state()
        if retry_error is None and retry_payload is not None:
            payload = retry_payload
            last_updated_s = _parse_ha_timestamp(payload.get("last_updated") or payload.get("last_changed"))
            data_age_s = time.time() - last_updated_s if last_updated_s is not None else None
            stale = data_age_s is not None and data_age_s > STALE_AFTER_S

    arrivals = extract_arrivals(payload, entity_id)
    if stale and not arrivals:
        return LiveBusSnapshot(
            arrivals=[],
            entity_id=entity_id,
            last_updated_s=last_updated_s,
            data_age_s=data_age_s,
            stale=True,
            ha_query_latency_s=latency,
            error="Bus arrival data is stale",
            fetched_at=time.time(),
        )
    return LiveBusSnapshot(
        arrivals=arrivals,
        entity_id=entity_id,
        last_updated_s=last_updated_s,
        data_age_s=data_age_s,
        stale=stale,
        ha_query_latency_s=latency,
        error=None,
        fetched_at=time.time(),
    )


def _state_from_json(value: object) -> BusMonitorState | None:
    if not isinstance(value, Mapping):
        return None
    monitor_id = value.get("monitor_id")
    entity_id = value.get("entity_id")
    status = value.get("status")
    if not isinstance(monitor_id, str) or not isinstance(entity_id, str) or not isinstance(status, str):
        return None
    created_at = value.get("created_at")
    if not isinstance(created_at, int | float):
        return None
    latest = value.get("latest_live_eta")
    if latest is not None and not isinstance(latest, int):
        return None
    preparation = value.get("preparation_threshold")
    urgent = value.get("urgent_threshold")
    if not isinstance(preparation, int) or not isinstance(urgent, int):
        return None
    last_check = value.get("last_check")
    if last_check is not None and not isinstance(last_check, int | float):
        last_check = None
    route = value.get("route") if isinstance(value.get("route"), str) else None
    stop = value.get("stop") if isinstance(value.get("stop"), str) else None
    service_id = value.get("service_id") if isinstance(value.get("service_id"), str) else None
    destination = value.get("destination") if isinstance(value.get("destination"), str) else None
    return BusMonitorState(
        monitor_id=monitor_id,
        entity_id=entity_id,
        route=route,
        stop=stop,
        service_id=service_id,
        created_at=float(created_at),
        latest_live_eta=latest,
        preparation_threshold=preparation,
        urgent_threshold=urgent,
        status=status,
        last_check=float(last_check) if isinstance(last_check, int | float) else None,
        preparation_alert_sent=bool(value.get("preparation_alert_sent")),
        urgent_alert_sent=bool(value.get("urgent_alert_sent")),
        arrival_alert_sent=bool(value.get("arrival_alert_sent")),
        destination=destination,
    )


class BusMonitorManager:
    """In-process Route 311 watch that polls Home Assistant without the LLM."""

    def __init__(
        self,
        *,
        poll_s: float = DEFAULT_POLL_S,
        close_poll_s: float = CLOSE_POLL_S,
        persist_path: Path | None = None,
    ) -> None:
        """Store poll intervals and an optional persistence path."""
        self._poll_s = poll_s
        self._close_poll_s = close_poll_s
        self._persist_path = persist_path
        self._instance_path: str | Path | None = None
        self._notify: NotifyFn | None = None
        self._monitor: BusMonitorState | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._pending_offer = False
        self._pending_threshold = DEFAULT_PREPARATION_MINUTES
        self._query_spoken_at: float | None = None
        self._cached_snapshot: LiveBusSnapshot | None = None

    def attach(
        self,
        *,
        instance_path: str | Path | None,
        notify: NotifyFn,
        persist_path: Path | None = None,
    ) -> None:
        """Bind speech and persistence for the current conversation session."""
        self._instance_path = instance_path
        self._notify = notify
        if persist_path is not None:
            self._persist_path = persist_path

    async def detach(self) -> None:
        """Stop the polling task without clearing persisted state."""
        self._notify = None
        await self._cancel_task()

    def pending_offer(self) -> bool:
        """Return whether the last live report offered monitoring."""
        return self._pending_offer

    def monitor_active(self) -> bool:
        """Return whether a background watch is running."""
        return self._monitor is not None and self._monitor.status == "active"

    def mark_query_spoken(self) -> None:
        """Record that the current live arrival was already spoken this turn."""
        self._query_spoken_at = time.monotonic()

    def query_already_spoken(self) -> bool:
        """Return whether the live arrival was spoken recently enough to skip a repeat."""
        if self._query_spoken_at is None:
            return False
        return (time.monotonic() - self._query_spoken_at) < QUERY_CACHE_TTL_S

    def _path(self) -> Path:
        if self._persist_path is not None:
            return self._persist_path
        return monitor_path_for_instance(self._instance_path)

    def _poll_interval(self, monitor: BusMonitorState) -> float:
        if monitor.latest_live_eta is not None and monitor.latest_live_eta <= CLOSE_POLL_WITHIN_MINUTES:
            return self._close_poll_s
        return self._poll_s

    def _persist(self) -> None:
        path = self._path()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "monitors": [asdict(self._monitor)]
            if self._monitor is not None and self._monitor.status == "active"
            else [],
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:
            logger.warning("Failed to persist bus monitor state: %s", exc)

    def _load(self) -> BusMonitorState | None:
        path = self._path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load bus monitor state: %s", exc)
            return None
        if not isinstance(raw, dict):
            return None
        monitors = raw.get("monitors")
        if not isinstance(monitors, list):
            return None
        for item in monitors:
            state = _state_from_json(item)
            if state is not None and state.status == "active":
                return state
        return None

    async def query(self, *, preparation_threshold: int = DEFAULT_PREPARATION_MINUTES) -> dict[str, Any]:
        """Return the latest live 311 arrival and whether monitoring should be offered."""
        decision_started = time.perf_counter()
        snapshot = await fetch_live_snapshot()
        self._cached_snapshot = snapshot
        arrival = snapshot.next_arrival
        threshold = 10 if preparation_threshold == 10 else DEFAULT_PREPARATION_MINUTES
        if snapshot.stale:
            spoken = "I could not get a fresh 311 reading from Home Assistant right now."
            kind = "none"
            self._pending_offer = False
        else:
            spoken = format_initial_spoken(snapshot, preparation_threshold=threshold)
            kind = offer_kind(arrival.minutes, preparation_threshold=threshold) if arrival is not None else "none"
            self._pending_offer = kind in {"offer_prepare", "offer_urgent"}
        self._pending_threshold = threshold
        decision_latency = time.perf_counter() - decision_started
        logger.info(
            "BUS_QUERY latency=%.2fs DECISION latency=%.2fs END_TO_END latency=%.2fs",
            snapshot.ha_query_latency_s,
            decision_latency,
            snapshot.ha_query_latency_s + decision_latency,
        )
        result: dict[str, Any] = {
            "spoken": spoken,
            "offer": kind,
            "entity_id": snapshot.entity_id,
            "ha_query_latency_s": round(snapshot.ha_query_latency_s, 3),
            "decision_latency_s": round(decision_latency, 3),
            "stale": snapshot.stale,
            "already_spoken": self.query_already_spoken(),
        }
        if snapshot.error:
            result["error"] = snapshot.error
        elif snapshot.stale:
            result["error"] = "Bus arrival data is stale"
        elif arrival is None:
            result["error"] = "Bus arrival data unavailable"
        else:
            result["minutes"] = arrival.minutes
            result["route"] = arrival.route
            result["destination"] = arrival.destination
            result["eta_display"] = arrival.eta_display
            result["realtime"] = arrival.realtime
            result["service_id"] = arrival.service_id
            following = snapshot.following_arrival
            if following is not None:
                result["following_minutes"] = following.minutes
                result["following_destination"] = following.destination
        return result

    async def start(self, *, preparation_threshold: int | None = None) -> dict[str, Any]:
        """Start watching the current live 311 service."""
        async with self._lock:
            threshold = preparation_threshold or self._pending_threshold
            snapshot = self._cached_snapshot
            if snapshot is None or (time.time() - snapshot.fetched_at) > QUERY_CACHE_TTL_S:
                snapshot = await fetch_live_snapshot()
                self._cached_snapshot = snapshot
            if snapshot.error:
                return {"error": snapshot.error, "spoken": format_initial_spoken(snapshot)}
            if snapshot.stale:
                return {
                    "error": "Bus arrival data is stale",
                    "spoken": "I could not get a fresh 311 reading from Home Assistant right now.",
                }
            arrival = snapshot.next_arrival
            if arrival is None:
                return {"error": "Bus arrival data unavailable", "spoken": format_initial_spoken(snapshot)}
            if arrival.minutes <= DEFAULT_URGENT_MINUTES:
                self._pending_offer = False
                return {
                    "status": "not_started",
                    "reason": "already_urgent",
                    "minutes": arrival.minutes,
                    "spoken": format_initial_spoken(snapshot),
                }
            await self._cancel_task()
            now = time.time()
            state = BusMonitorState(
                monitor_id=str(uuid.uuid4()),
                entity_id=snapshot.entity_id,
                route=arrival.route,
                stop=arrival.stop,
                service_id=arrival.service_id,
                created_at=now,
                latest_live_eta=arrival.minutes,
                preparation_threshold=threshold,
                urgent_threshold=DEFAULT_URGENT_MINUTES,
                status="active",
                last_check=now,
                preparation_alert_sent=arrival.minutes <= threshold,
                urgent_alert_sent=False,
                arrival_alert_sent=False,
                destination=arrival.destination,
            )
            self._monitor = state
            self._pending_offer = False
            self._persist()
            self._task = asyncio.create_task(self._run_loop(), name="bus-monitor")
            logger.info(
                "Bus monitor started id=%s eta=%s prep=%s",
                state.monitor_id,
                arrival.minutes,
                threshold,
            )
            return {
                "status": "monitoring",
                "monitor_id": state.monitor_id,
                "minutes": arrival.minutes,
                "preparation_threshold": threshold,
                "urgent_threshold": DEFAULT_URGENT_MINUTES,
                "spoken": f"Okay, I'll watch the {route_label(arrival)} and let you know.",
            }

    async def cancel(self) -> dict[str, Any]:
        """Stop the active watch if one exists."""
        async with self._lock:
            self._pending_offer = False
            monitor = self._monitor
            await self._cancel_task()
            self._monitor = None
            self._persist()
            if monitor is None:
                return {"status": "idle", "spoken": "I'm not watching the bus right now."}
            logger.info("Bus monitor cancelled id=%s", monitor.monitor_id)
            return {
                "status": "cancelled",
                "monitor_id": monitor.monitor_id,
                "spoken": "Okay, I've stopped watching the bus.",
            }

    def status(self) -> dict[str, Any]:
        """Return the current watch, if any."""
        monitor = self._monitor
        if monitor is None or monitor.status != "active":
            return {"status": "idle", "pending_offer": self._pending_offer}
        return {
            "status": "monitoring",
            "monitor_id": monitor.monitor_id,
            "minutes": monitor.latest_live_eta,
            "preparation_threshold": monitor.preparation_threshold,
            "urgent_threshold": monitor.urgent_threshold,
            "preparation_alert_sent": monitor.preparation_alert_sent,
            "urgent_alert_sent": monitor.urgent_alert_sent,
            "arrival_alert_sent": monitor.arrival_alert_sent,
            "pending_offer": self._pending_offer,
        }

    async def resume_from_disk(self) -> dict[str, Any]:
        """Reload a valid watch after process restart."""
        loaded = self._load()
        if loaded is None:
            return {"status": "idle"}
        if time.time() - loaded.created_at > MAX_MONITOR_AGE_S:
            logger.info("Dropping stale bus monitor id=%s", loaded.monitor_id)
            self._monitor = None
            self._persist()
            return {"status": "expired"}
        snapshot = await fetch_live_snapshot()
        if snapshot.error:
            logger.info("Bus monitor resume deferred; Home Assistant data unavailable")
            self._monitor = loaded
            self._task = asyncio.create_task(self._run_loop(), name="bus-monitor")
            return {"status": "resumed", "monitor_id": loaded.monitor_id, "deferred": True}
        if loaded.arrival_alert_sent:
            self._monitor = None
            self._persist()
            return {"status": "expired"}
        arrival = find_monitored_arrival(loaded, snapshot)
        if arrival is None:
            logger.info("Dropping completed bus monitor id=%s; watched service is gone", loaded.monitor_id)
            self._monitor = None
            self._persist()
            return {"status": "expired"}
        loaded.latest_live_eta = arrival.minutes
        if arrival.minutes <= 0:
            loaded.arrival_alert_sent = True
            loaded.urgent_alert_sent = True
            loaded.preparation_alert_sent = True
            self._monitor = None
            self._persist()
            return {"status": "expired"}
        if arrival.minutes <= loaded.urgent_threshold:
            loaded.urgent_alert_sent = True
            loaded.preparation_alert_sent = True
        elif arrival.minutes <= loaded.preparation_threshold:
            loaded.preparation_alert_sent = True
        self._monitor = loaded
        self._persist()
        self._task = asyncio.create_task(self._run_loop(), name="bus-monitor")
        logger.info("Bus monitor resumed id=%s eta=%s", loaded.monitor_id, loaded.latest_live_eta)
        return {"status": "resumed", "monitor_id": loaded.monitor_id, "minutes": loaded.latest_live_eta}

    async def _cancel_task(self) -> None:
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_loop(self) -> None:
        failures = 0
        try:
            while True:
                monitor = self._monitor
                if monitor is None or monitor.status != "active":
                    return
                await asyncio.sleep(self._poll_interval(monitor))
                try:
                    await self._poll_once()
                    failures = 0
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    failures += 1
                    backoff = min(MAX_HA_FAILURE_BACKOFF_S, 5.0 * (2 ** (failures - 1)))
                    logger.warning("Bus monitor poll failed; retrying in %.0fs: %s", backoff, exc)
                    await asyncio.sleep(backoff)
        except asyncio.CancelledError:
            logger.info("Bus monitor loop cancelled")
            raise

    async def _finish_watch(self, *, reason: str, spoken: str | None = None) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        if spoken is not None and not monitor.arrival_alert_sent:
            monitor.arrival_alert_sent = True
            self._persist()
            await self._emit(spoken)
        logger.info("Bus monitor completed id=%s reason=%s", monitor.monitor_id, reason)
        self._monitor = None
        self._persist()

    async def _poll_once(self) -> None:
        monitor = self._monitor
        if monitor is None or monitor.status != "active":
            return
        if time.time() - monitor.created_at > MAX_MONITOR_AGE_S:
            logger.info("Bus monitor timed out id=%s", monitor.monitor_id)
            self._monitor = None
            self._persist()
            return
        poll_started = time.perf_counter()
        snapshot = await fetch_live_snapshot()
        decision_started = time.perf_counter()
        if snapshot.error:
            logger.warning("Bus monitor keeping previous ETA; Home Assistant error: %s", snapshot.error)
            return
        if snapshot.stale:
            logger.warning(
                "Bus monitor keeping previous ETA; Home Assistant data is stale age=%.0fs",
                snapshot.data_age_s or 0.0,
            )
            return
        arrival = find_monitored_arrival(monitor, snapshot)
        if arrival is None:
            await self._finish_watch(reason="service_gone", spoken=format_arrival_alert(None))
            return
        monitor.latest_live_eta = arrival.minutes
        monitor.last_check = time.time()
        alert = evaluate_alerts(monitor, arrival)
        decision_latency = time.perf_counter() - decision_started
        notify_latency = 0.0
        if alert == "preparation":
            notify_started = time.perf_counter()
            await self._emit(format_preparation_alert(arrival))
            notify_latency = time.perf_counter() - notify_started
            monitor.preparation_alert_sent = True
        elif alert == "urgent":
            notify_started = time.perf_counter()
            await self._emit(format_urgent_alert(arrival))
            notify_latency = time.perf_counter() - notify_started
            monitor.urgent_alert_sent = True
            monitor.preparation_alert_sent = True
        elif alert == "arrival":
            notify_started = time.perf_counter()
            await self._finish_watch(reason="eta_zero", spoken=format_arrival_alert(arrival))
            notify_latency = time.perf_counter() - notify_started
            logger.info(
                "BUS_QUERY latency=%.2fs DECISION latency=%.2fs NOTIFICATION latency=%.2fs END_TO_END latency=%.2fs",
                snapshot.ha_query_latency_s,
                decision_latency,
                notify_latency,
                time.perf_counter() - poll_started,
            )
            return
        self._persist()
        if alert is not None:
            logger.info(
                "BUS_QUERY latency=%.2fs DECISION latency=%.2fs NOTIFICATION latency=%.2fs END_TO_END latency=%.2fs",
                snapshot.ha_query_latency_s,
                decision_latency,
                notify_latency,
                time.perf_counter() - poll_started,
            )

    async def _emit(self, text: str) -> None:
        notify = self._notify
        if notify is None:
            logger.warning("Bus monitor alert dropped; no speech callback bound")
            return
        try:
            await notify(text)
        except Exception as exc:
            logger.warning("Bus monitor notification failed: %s", exc)


_MANAGER: BusMonitorManager | None = None


def get_bus_monitor() -> BusMonitorManager:
    """Return the process-wide bus monitor."""
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = BusMonitorManager()
    return _MANAGER


def reset_bus_monitor_for_tests(manager: BusMonitorManager | None = None) -> BusMonitorManager:
    """Replace the process-wide manager. Tests only."""
    global _MANAGER
    _MANAGER = manager if manager is not None else BusMonitorManager()
    return _MANAGER
