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
from enum import Enum
from typing import Any, Callable, Awaitable
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import asdict, replace, dataclass
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
SYDNEY_TIMEZONE = "Australia/Sydney"
_SYDNEY_TZ = ZoneInfo(SYDNEY_TIMEZONE)
DEFAULT_PREPARATION_MINUTES = 15
TEN_MINUTE_THRESHOLD = 10
SEVEN_MINUTE_THRESHOLD = 7
DEFAULT_URGENT_MINUTES = 5
ETA_OSCILLATION_SLACK_MINUTES = 3
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
_SWITCH_RE = re.compile(
    r"\b(?:monitor (?:the )?(?:next|following|later)(?: one| bus| 311)?(?: instead)?"
    r"|what about the (?:next|following|later)(?: one| bus| 311)?"
    r"|switch (?:to )?(?:the )?(?:next|following|later)"
    r"|watch the (?:next|following|later))\b",
    re.IGNORECASE,
)
_CONTINUOUS_RE = re.compile(
    r"\b(?:keep monitoring(?: the)?(?: 311s?| buses)?"
    r"|monitor (?:all|every|each)(?: (?:coming|the))? 311s?"
    r"|keep watching the 311s?)\b",
    re.IGNORECASE,
)
_CLOCK_RE = re.compile(r"\b(\d{1,2}:\d{2})(?::\d{2})?\b")

NotifyFn = Callable[[str], Awaitable[None]]
PlayHelpful1Fn = Callable[[], Awaitable[None]]
TEN_MINUTE_EMOTION = "helpful1"


class BusServiceState(str, Enum):
    """Authoritative Route 311 service state. Zero minutes is not ARRIVED."""

    UPCOMING = "UPCOMING"
    ARRIVING = "ARRIVING"
    ARRIVED = "ARRIVED"
    SERVICE_GONE = "SERVICE_GONE"
    NO_SERVICE = "NO_SERVICE"
    UNKNOWN = "UNKNOWN"


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
    scheduled_at: str | None = None


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
    ten_minute_alert_sent: bool = False
    seven_minute_alert_sent: bool = False
    continuous: bool = False
    destination: str | None = None
    eta_display: str | None = None
    scheduled_at: str | None = None


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


def _scheduled_from_source(source: Mapping[str, Any]) -> str | None:
    return _first_string(
        dict(source),
        (
            "aimed_departure_time",
            "scheduled_departure",
            "departure_time",
            "scheduled",
            "arrival_time",
            "due_at",
        ),
    )


def _parse_ha_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_SYDNEY_TZ)
    return parsed


def _parse_ha_timestamp(value: object) -> float | None:
    parsed = _parse_ha_datetime(value)
    return parsed.timestamp() if parsed is not None else None


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
        scheduled_at=_scheduled_from_source(source),
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
        id_counts: dict[str, int] = {}
        for arrival in arrivals:
            if arrival.service_id:
                id_counts[arrival.service_id] = id_counts.get(arrival.service_id, 0) + 1
        return [
            replace(arrival, service_id=None)
            if arrival.service_id is not None and id_counts.get(arrival.service_id, 0) > 1
            else arrival
            for arrival in arrivals
        ]
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


def _clock_label(arrival: BusArrival) -> str | None:
    for candidate in (arrival.scheduled_at, arrival.eta_display):
        if not candidate:
            continue
        if re.search(r"\b(?:h|hr|hour|min|mins|minute)\b", candidate, re.IGNORECASE):
            continue
        parsed = _parse_ha_datetime(candidate)
        if parsed is not None and any(marker in candidate for marker in ("T", "+", "Z")):
            local = parsed.astimezone(_SYDNEY_TZ)
            return f"{local.hour:02d}:{local.minute:02d}"
        match = _CLOCK_RE.search(candidate)
        if match is None:
            continue
        hour_text, minute_text = match.group(1).split(":")
        if int(hour_text) > 23 or int(minute_text) > 59:
            continue
        return match.group(1)
    return None


def _approx_clock(minutes: int) -> str:
    when = datetime.now(_SYDNEY_TZ) + timedelta(minutes=max(0, minutes))
    hour = when.hour % 12 or 12
    return f"{hour}:{when.minute:02d}"


def format_service_description(arrival: BusArrival) -> str:
    """Return an unambiguous spoken label for one Home Assistant service."""
    route = route_label(arrival)
    clock = _clock_label(arrival)
    if clock is None and arrival.minutes >= 0:
        clock = _approx_clock(arrival.minutes)
    if clock:
        return f"the {route} arriving at approximately {clock}"
    if arrival.minutes <= 0:
        return f"the {route} that is due now"
    return f"the {route} arriving in about {arrival.minutes} minutes"


def format_following_clause(following: BusArrival | None) -> str | None:
    """Return the following-service clause, or None when HA has no later 311."""
    if following is None:
        return None
    return f"The following {route_label(following)} is about {following.minutes} minutes away"


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
    following_clause = format_following_clause(snapshot.following_arrival)
    if arrival.minutes <= 0:
        spoken = f"The next {route} is due now. Please leave now."
    elif arrival.minutes <= 1:
        spoken = f"The next {route} is arriving now. Please leave now."
    elif arrival.minutes <= urgent_threshold:
        spoken = f"The next {route} is arriving in about {arrival.minutes} minutes. Please leave now."
    else:
        spoken = f"The next {route} is currently about {arrival.minutes} minutes away"
        clock = _clock_label(arrival)
        if clock:
            spoken += f", expected at {clock}"
        spoken += "."
        if following_clause is not None:
            spoken += f" {following_clause}."
        if arrival.minutes <= preparation_threshold:
            spoken = spoken.rstrip(".")
            spoken += f". Would you like me to keep monitoring it and let you know when it's about {urgent_threshold} minutes away?"
            return spoken
        spoken = spoken.rstrip(".")
        spoken += f". Would you like me to monitor it and alert you when it's within {preparation_threshold} minutes?"
        return spoken

    if following_clause is not None:
        spoken = f"{spoken.rstrip('.')}. {following_clause}."
    return spoken


def format_preparation_alert(arrival: BusArrival) -> str:
    """Build the one-shot preparation warning."""
    return f"The {route_label(arrival)} is now about {arrival.minutes} minutes away. You have time to get ready."


def format_ten_minute_alert(arrival: BusArrival) -> str:
    """Build the one-shot 10-minute notice."""
    return f"The {route_label(arrival)} is now about {arrival.minutes} minutes away."


def format_seven_minute_alert(arrival: BusArrival) -> str:
    """Build the one-shot get-ready notice."""
    return f"The {route_label(arrival)} is now about {arrival.minutes} minutes away. You better get ready to leave."


def format_urgent_alert(arrival: BusArrival, following: BusArrival | None = None) -> str:
    """Build the one-shot leave-now warning."""
    if arrival.minutes <= 1:
        spoken = f"The {route_label(arrival)} is arriving now. Please leave now."
    else:
        spoken = f"The {route_label(arrival)} is now about {arrival.minutes} minutes away. Please leave now."
    following_clause = format_following_clause(following)
    if following_clause is not None:
        spoken = f"{spoken.rstrip('.')}. {following_clause}."
    return spoken


def classify_bus_state(
    arrival: BusArrival | None,
    *,
    snapshot: LiveBusSnapshot | None = None,
    reason: str | None = None,
    arrival_confirmed: bool = False,
) -> BusServiceState:
    """Classify the live 311 state without treating a 0-minute ETA as arrival."""
    if arrival_confirmed:
        return BusServiceState.ARRIVED
    if reason == "service_gone":
        return BusServiceState.SERVICE_GONE
    if snapshot is not None and (snapshot.error or snapshot.stale):
        return BusServiceState.UNKNOWN
    if arrival is None:
        return BusServiceState.NO_SERVICE
    if arrival.minutes <= 1:
        return BusServiceState.ARRIVING
    return BusServiceState.UPCOMING


def format_service_gone_alert(arrival: BusArrival | None) -> str:
    """Speak a lost-feed state without claiming the bus arrived."""
    return f"The live {route_label(arrival)} feed has lost that service. I can't confirm that it has arrived."


def format_arrival_alert(arrival: BusArrival | None, *, confirmed: bool = False) -> str:
    """Build the terminal notice for the watched service."""
    if confirmed:
        return f"The {route_label(arrival)} has arrived."
    if arrival is None or arrival.minutes > 0:
        return format_service_gone_alert(arrival)
    return f"The {route_label(arrival)} is due now."


def format_monitoring_started(arrival: BusArrival, *, switched: bool = False) -> str:
    """Confirm which Home Assistant service the watch is following."""
    description = format_service_description(arrival)
    if switched:
        return f"Okay. I've switched the monitor to {description}."
    return f"Okay. I'm monitoring {description}."


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
    threshold = TEN_MINUTE_THRESHOLD if _TEN_MINUTE_RE.search(text) else DEFAULT_PREPARATION_MINUTES
    if pending_offer or monitor_active:
        if _CANCEL_RE.search(text) or _CANCEL_THAT_RE.search(text):
            return BusUtterance(action="cancel", preparation_threshold=threshold)
    elif _CANCEL_RE.search(text) and _BUS_QUERY_RE.search(text):
        return BusUtterance(action="cancel", preparation_threshold=threshold)
    if (monitor_active or pending_offer) and _SWITCH_RE.search(text):
        return BusUtterance(action="switch", preparation_threshold=threshold)
    if _CONTINUOUS_RE.search(text):
        return BusUtterance(action="continuous", preparation_threshold=threshold)
    if pending_offer and _CONFIRM_RE.search(text) and not _QUESTION_PREFIX.search(text):
        return BusUtterance(action="confirm", preparation_threshold=threshold)
    if (pending_offer or monitor_active) and _STATUS_RE.search(text):
        return BusUtterance(action="query", preparation_threshold=threshold)
    if _BUS_QUERY_RE.search(text):
        return BusUtterance(action="query", preparation_threshold=threshold)
    return None


def same_service(monitor: BusMonitorState, arrival: BusArrival) -> bool:
    """Return whether this live arrival is still the monitored service."""
    if monitor.service_id and arrival.service_id and monitor.service_id != arrival.service_id:
        return False
    if monitor.scheduled_at and arrival.scheduled_at and monitor.scheduled_at != arrival.scheduled_at:
        return False
    if monitor.route and arrival.route and monitor.route != arrival.route:
        return False
    if monitor.destination and arrival.destination and monitor.destination != arrival.destination:
        return False
    if monitor.latest_live_eta is not None:
        slack = ETA_OSCILLATION_SLACK_MINUTES if monitor.latest_live_eta <= CLOSE_POLL_WITHIN_MINUTES else 8
        if arrival.minutes > monitor.latest_live_eta + slack:
            return False
    return True


def find_monitored_arrival(monitor: BusMonitorState, snapshot: LiveBusSnapshot) -> BusArrival | None:
    """Return the watched service from the live snapshot, if it is still present."""
    for arrival in snapshot.arrivals:
        if same_service(monitor, arrival):
            return arrival
    return None


def find_following_arrival(monitor: BusMonitorState | None, snapshot: LiveBusSnapshot) -> BusArrival | None:
    """Return the live service after the current/monitored one."""
    if monitor is None:
        return snapshot.following_arrival
    found = False
    for arrival in snapshot.arrivals:
        if found:
            return arrival
        if same_service(monitor, arrival):
            found = True
    return None


def next_unmonitored_arrival(monitor: BusMonitorState, snapshot: LiveBusSnapshot) -> BusArrival | None:
    """Return the first live service that is not the watched one."""
    for arrival in snapshot.arrivals:
        if not same_service(monitor, arrival):
            return arrival
    return None


def _threshold_crossed(previous_eta: int | None, current_eta: int, threshold: int) -> bool:
    if current_eta > threshold:
        return False
    if previous_eta is None:
        return False
    return previous_eta > threshold


def evaluate_alerts(monitor: BusMonitorState, arrival: BusArrival) -> list[str]:
    """Return crossed alert kinds in order for this poll, without retrospective start alerts."""
    previous_eta = monitor.latest_live_eta
    current_eta = arrival.minutes
    plan: list[tuple[str, int, bool]] = [
        ("preparation", monitor.preparation_threshold, monitor.preparation_alert_sent),
    ]
    if monitor.preparation_threshold != TEN_MINUTE_THRESHOLD:
        plan.append(("ten", TEN_MINUTE_THRESHOLD, monitor.ten_minute_alert_sent))
    plan.extend(
        (
            ("seven", SEVEN_MINUTE_THRESHOLD, monitor.seven_minute_alert_sent),
            ("urgent", monitor.urgent_threshold, monitor.urgent_alert_sent),
            ("arrival", 0, monitor.arrival_alert_sent),
        )
    )
    alerts: list[str] = []
    for kind, threshold, already_sent in plan:
        if already_sent:
            continue
        if _threshold_crossed(previous_eta, current_eta, threshold):
            alerts.append(kind)
    return alerts


def _arrival_diagnostic(arrival: BusArrival) -> dict[str, Any]:
    return {
        "minutes": arrival.minutes,
        "route": arrival.route,
        "stop": arrival.stop,
        "destination": arrival.destination,
        "eta_display": arrival.eta_display,
        "scheduled_at": arrival.scheduled_at,
        "realtime": arrival.realtime,
        "service_id": arrival.service_id,
    }


def _arrival_payload(arrival: BusArrival) -> dict[str, Any]:
    return {
        "route": arrival.route,
        "stop": arrival.stop,
        "direction": arrival.destination,
        "realtime": arrival.realtime,
        "next_minutes": arrival.minutes,
        "eta_display": arrival.eta_display,
        "scheduled_at": arrival.scheduled_at,
        "service_id": arrival.service_id,
        "destination": arrival.destination,
    }


def _log_bus_state(
    arrival: BusArrival | None,
    *,
    previous_state: BusServiceState | None,
    new_state: BusServiceState,
    arrival_confirmed: bool,
    reason: str,
) -> None:
    logger.info("[BUS] route=%s", arrival.route if arrival is not None else DEFAULT_ROUTE)
    logger.info("[BUS] stop=%s", arrival.stop if arrival is not None else None)
    logger.info("[BUS] direction=%s", arrival.destination if arrival is not None else None)
    logger.info("[BUS] realtime=%s", arrival.realtime if arrival is not None else None)
    logger.info("[BUS] next_minutes=%s", arrival.minutes if arrival is not None else None)
    logger.info("[BUS] eta_display=%s", arrival.eta_display if arrival is not None else None)
    logger.info("[BUS] previous_state=%s", previous_state.value if previous_state is not None else None)
    logger.info("[BUS] new_state=%s", new_state.value)
    logger.info("[BUS] arrival_confirmed=%s", arrival_confirmed)
    logger.info("[BUS] reason=%s", reason)


def _sydney_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, _SYDNEY_TZ).isoformat()


def _log_ha_bus_payload(payload: dict[str, Any], entity_id: str, arrivals: list[BusArrival]) -> None:
    attributes = payload.get("attributes")
    if not isinstance(attributes, dict):
        attributes = {}
    last_updated = payload.get("last_updated") or payload.get("last_changed")
    parsed = _parse_ha_datetime(last_updated)
    next_arrival = arrivals[0] if arrivals else None
    route = attributes.get("route") or attributes.get("route_short_name")
    stop = attributes.get("stop") or attributes.get("stop_name")
    direction = attributes.get("direction") or attributes.get("headsign") or attributes.get("destination")
    if next_arrival is not None:
        route = next_arrival.route or route
        stop = next_arrival.stop or stop
        direction = next_arrival.destination or direction
    logger.info(
        "[BUS] ha_payload entity_id=%s state=%s friendly_name=%s route=%s stop=%s direction=%s "
        "last_updated=%s last_updated_sydney=%s timezone=%s next_minutes=%s realtime=%s "
        "scheduled_at=%s eta_display=%s arrivals=%s",
        entity_id,
        payload.get("state"),
        attributes.get("friendly_name"),
        route,
        stop,
        direction,
        last_updated,
        parsed.astimezone(_SYDNEY_TZ).isoformat() if parsed is not None else None,
        SYDNEY_TIMEZONE,
        next_arrival.minutes if next_arrival is not None else None,
        next_arrival.realtime if next_arrival is not None else None,
        next_arrival.scheduled_at if next_arrival is not None else None,
        next_arrival.eta_display if next_arrival is not None else None,
        [_arrival_diagnostic(item) for item in arrivals],
    )


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
    _log_ha_bus_payload(payload, entity_id, arrivals)
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
    eta_display = value.get("eta_display") if isinstance(value.get("eta_display"), str) else None
    scheduled_at = value.get("scheduled_at") if isinstance(value.get("scheduled_at"), str) else None
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
        ten_minute_alert_sent=bool(value.get("ten_minute_alert_sent")),
        seven_minute_alert_sent=bool(value.get("seven_minute_alert_sent")),
        continuous=bool(value.get("continuous")),
        destination=destination,
        eta_display=eta_display,
        scheduled_at=scheduled_at,
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
        self._play_helpful1: PlayHelpful1Fn | None = None
        self._helpful1_task: asyncio.Task[None] | None = None
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
        play_helpful1: PlayHelpful1Fn | None = None,
    ) -> None:
        """Bind speech, emotion playback, and persistence for the current conversation session."""
        self._instance_path = instance_path
        self._notify = notify
        self._play_helpful1 = play_helpful1
        if persist_path is not None:
            self._persist_path = persist_path

    async def detach(self) -> None:
        """Stop the polling task without clearing persisted state."""
        self._notify = None
        self._play_helpful1 = None
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
            self._pending_offer = kind in {"offer_prepare", "offer_urgent", "leave_now"}
        self._pending_threshold = threshold
        decision_latency = time.perf_counter() - decision_started
        logger.info(
            "BUS_QUERY latency=%.2fs DECISION latency=%.2fs END_TO_END latency=%.2fs",
            snapshot.ha_query_latency_s,
            decision_latency,
            snapshot.ha_query_latency_s + decision_latency,
        )
        service_state = classify_bus_state(arrival, snapshot=snapshot)
        result: dict[str, Any] = {
            "spoken": spoken,
            "offer": kind,
            "entity_id": snapshot.entity_id,
            "ha_query_latency_s": round(snapshot.ha_query_latency_s, 3),
            "decision_latency_s": round(decision_latency, 3),
            "stale": snapshot.stale,
            "already_spoken": self.query_already_spoken(),
            "timezone": SYDNEY_TIMEZONE,
            "last_updated": _sydney_iso(snapshot.last_updated_s),
            "service_state": service_state.value,
            "arrival_confirmed": False,
            "arrivals": [_arrival_payload(item) for item in snapshot.arrivals],
        }
        if snapshot.error:
            result["error"] = snapshot.error
        elif snapshot.stale:
            result["error"] = "Bus arrival data is stale"
        elif arrival is None:
            result["error"] = "Bus arrival data unavailable"
        else:
            result["minutes"] = arrival.minutes
            result["next_minutes"] = arrival.minutes
            result["route"] = arrival.route
            result["stop"] = arrival.stop
            result["direction"] = arrival.destination
            result["destination"] = arrival.destination
            result["eta_display"] = arrival.eta_display
            result["scheduled_at"] = arrival.scheduled_at
            result["realtime"] = arrival.realtime
            result["service_id"] = arrival.service_id
            following = snapshot.following_arrival
            if following is not None:
                result["following_minutes"] = following.minutes
                result["following_destination"] = following.destination
                result["following_eta_display"] = following.eta_display
                result["following_service_id"] = following.service_id
        _log_bus_state(
            arrival,
            previous_state=None,
            new_state=service_state,
            arrival_confirmed=False,
            reason="query",
        )
        logger.info("[BUS] tool_result=%s", result)
        return result

    def _bind_service(
        self,
        snapshot: LiveBusSnapshot,
        arrival: BusArrival,
        *,
        threshold: int,
        continuous: bool,
        monitor_id: str | None = None,
    ) -> BusMonitorState:
        now = time.time()
        return BusMonitorState(
            monitor_id=monitor_id or str(uuid.uuid4()),
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
            preparation_alert_sent=False,
            urgent_alert_sent=False,
            arrival_alert_sent=False,
            ten_minute_alert_sent=False,
            seven_minute_alert_sent=False,
            continuous=continuous,
            destination=arrival.destination,
            eta_display=arrival.eta_display,
            scheduled_at=arrival.scheduled_at,
        )

    async def _start_unlocked(
        self,
        *,
        preparation_threshold: int | None = None,
        continuous: bool = False,
        arrival: BusArrival | None = None,
        snapshot: LiveBusSnapshot | None = None,
        switched: bool = False,
    ) -> dict[str, Any]:
        threshold = preparation_threshold or self._pending_threshold
        live = snapshot
        if live is None:
            live = self._cached_snapshot
            if live is None or (time.time() - live.fetched_at) > QUERY_CACHE_TTL_S:
                live = await fetch_live_snapshot()
                self._cached_snapshot = live
        if live.error:
            return {"error": live.error, "spoken": format_initial_spoken(live)}
        if live.stale:
            return {
                "error": "Bus arrival data is stale",
                "spoken": "I could not get a fresh 311 reading from Home Assistant right now.",
            }
        selected = arrival or live.next_arrival
        if selected is None:
            return {"error": "Bus arrival data unavailable", "spoken": format_initial_spoken(live)}
        await self._cancel_task()
        state = self._bind_service(live, selected, threshold=threshold, continuous=continuous)
        self._monitor = state
        self._pending_offer = False
        self._persist()
        self._task = asyncio.create_task(self._run_loop(), name="bus-monitor")
        logger.info(
            "Bus monitor started id=%s eta=%s prep=%s service=%s continuous=%s",
            state.monitor_id,
            selected.minutes,
            threshold,
            selected.service_id,
            continuous,
        )
        return {
            "status": "monitoring",
            "monitor_id": state.monitor_id,
            "minutes": selected.minutes,
            "service_id": selected.service_id,
            "preparation_threshold": threshold,
            "urgent_threshold": DEFAULT_URGENT_MINUTES,
            "continuous": continuous,
            "spoken": format_monitoring_started(selected, switched=switched),
        }

    async def start(
        self,
        *,
        preparation_threshold: int | None = None,
        continuous: bool = False,
        arrival: BusArrival | None = None,
        snapshot: LiveBusSnapshot | None = None,
        switched: bool = False,
    ) -> dict[str, Any]:
        """Start watching a specific live 311 service."""
        async with self._lock:
            return await self._start_unlocked(
                preparation_threshold=preparation_threshold,
                continuous=continuous,
                arrival=arrival,
                snapshot=snapshot,
                switched=switched,
            )

    async def switch(self) -> dict[str, Any]:
        """Move the watch to the following live 311 after an explicit user request."""
        snapshot = await fetch_live_snapshot()
        self._cached_snapshot = snapshot
        if snapshot.error or snapshot.stale:
            spoken = (
                "I could not get a fresh 311 reading from Home Assistant right now."
                if snapshot.stale
                else format_initial_spoken(snapshot)
            )
            return {"error": snapshot.error or "Bus arrival data is stale", "spoken": spoken}
        async with self._lock:
            monitor = self._monitor
            following = (
                next_unmonitored_arrival(monitor, snapshot)
                if monitor is not None and monitor.status == "active"
                else snapshot.following_arrival
            )
            if following is None:
                return {
                    "status": "unchanged",
                    "spoken": "There isn't a later 311 in Home Assistant right now.",
                }
            threshold = monitor.preparation_threshold if monitor is not None else self._pending_threshold
            continuous = monitor.continuous if monitor is not None else False
            return await self._start_unlocked(
                preparation_threshold=threshold,
                continuous=continuous,
                arrival=following,
                snapshot=snapshot,
                switched=True,
            )

    async def keep_monitoring(self, *, preparation_threshold: int | None = None) -> dict[str, Any]:
        """Enable continuous 311 watches after an explicit user request."""
        async with self._lock:
            monitor = self._monitor
            if monitor is not None and monitor.status == "active":
                monitor.continuous = True
                self._persist()
                return {
                    "status": "monitoring",
                    "monitor_id": monitor.monitor_id,
                    "continuous": True,
                    "spoken": "Okay. I'll keep monitoring the 311s after this one arrives.",
                }
            return await self._start_unlocked(
                preparation_threshold=preparation_threshold,
                continuous=True,
            )

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
            "ten_minute_alert_sent": monitor.ten_minute_alert_sent,
            "seven_minute_alert_sent": monitor.seven_minute_alert_sent,
            "urgent_alert_sent": monitor.urgent_alert_sent,
            "arrival_alert_sent": monitor.arrival_alert_sent,
            "continuous": monitor.continuous,
            "service_id": monitor.service_id,
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
            loaded.seven_minute_alert_sent = True
            loaded.ten_minute_alert_sent = True
            loaded.preparation_alert_sent = True
            self._monitor = None
            self._persist()
            return {"status": "expired"}
        if arrival.minutes <= loaded.urgent_threshold:
            loaded.urgent_alert_sent = True
            loaded.seven_minute_alert_sent = True
            loaded.ten_minute_alert_sent = True
            loaded.preparation_alert_sent = True
        elif arrival.minutes <= SEVEN_MINUTE_THRESHOLD:
            loaded.seven_minute_alert_sent = True
            loaded.ten_minute_alert_sent = True
            loaded.preparation_alert_sent = True
        elif arrival.minutes <= TEN_MINUTE_THRESHOLD:
            loaded.ten_minute_alert_sent = True
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

    def _mark_alert_sent(self, monitor: BusMonitorState, kind: str) -> None:
        if kind == "preparation":
            monitor.preparation_alert_sent = True
        elif kind == "ten":
            monitor.ten_minute_alert_sent = True
            monitor.preparation_alert_sent = True
        elif kind == "seven":
            monitor.seven_minute_alert_sent = True
            monitor.ten_minute_alert_sent = True
            monitor.preparation_alert_sent = True
        elif kind == "urgent":
            monitor.urgent_alert_sent = True
            monitor.seven_minute_alert_sent = True
            monitor.ten_minute_alert_sent = True
            monitor.preparation_alert_sent = True
        elif kind == "arrival":
            monitor.arrival_alert_sent = True
            monitor.urgent_alert_sent = True
            monitor.seven_minute_alert_sent = True
            monitor.ten_minute_alert_sent = True
            monitor.preparation_alert_sent = True

    def _spoken_for_alert(
        self,
        kind: str,
        arrival: BusArrival | None,
        following: BusArrival | None,
    ) -> str:
        if kind == "preparation" and arrival is not None:
            return format_preparation_alert(arrival)
        if kind == "ten" and arrival is not None:
            return format_ten_minute_alert(arrival)
        if kind == "seven" and arrival is not None:
            return format_seven_minute_alert(arrival)
        if kind == "urgent" and arrival is not None:
            return format_urgent_alert(arrival, following)
        return format_arrival_alert(arrival)

    async def _handoff_continuous(self, snapshot: LiveBusSnapshot, previous: BusArrival | None) -> bool:
        monitor = self._monitor
        if monitor is None or not monitor.continuous:
            return False
        following = next_unmonitored_arrival(monitor, snapshot)
        if following is None or following.minutes <= 0:
            return False
        threshold = monitor.preparation_threshold
        state = self._bind_service(snapshot, following, threshold=threshold, continuous=True)
        self._monitor = state
        self._persist()
        spoken = f"{format_arrival_alert(previous)} I'm now monitoring {format_service_description(following)}."
        await self._emit(spoken)
        logger.info(
            "Bus monitor handed off to following service id=%s eta=%s service=%s",
            state.monitor_id,
            following.minutes,
            following.service_id,
        )
        return True

    async def _finish_watch(
        self,
        *,
        reason: str,
        spoken: str | None = None,
        snapshot: LiveBusSnapshot | None = None,
        arrival: BusArrival | None = None,
    ) -> None:
        monitor = self._monitor
        if monitor is None:
            return
        if snapshot is not None and await self._handoff_continuous(snapshot, arrival):
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
        previous_eta = monitor.latest_live_eta
        previous_state = classify_bus_state(
            BusArrival(
                minutes=previous_eta if previous_eta is not None else -1,
                entity_id=monitor.entity_id,
                route=monitor.route,
                destination=monitor.destination,
                eta_display=monitor.eta_display,
                service_id=monitor.service_id,
                stop=monitor.stop,
                scheduled_at=monitor.scheduled_at,
            )
            if previous_eta is not None
            else None
        )
        if arrival is None:
            new_state = classify_bus_state(None, snapshot=snapshot, reason="service_gone")
            _log_bus_state(
                None,
                previous_state=previous_state,
                new_state=new_state,
                arrival_confirmed=False,
                reason="service_gone",
            )
            await self._finish_watch(
                reason="service_gone",
                spoken=format_arrival_alert(None),
                snapshot=snapshot,
            )
            return
        following = find_following_arrival(monitor, snapshot)
        alerts = evaluate_alerts(monitor, arrival)
        helpful1_sent = (
            monitor.preparation_alert_sent
            if monitor.preparation_threshold == TEN_MINUTE_THRESHOLD
            else monitor.ten_minute_alert_sent
        )
        if helpful1_sent and _threshold_crossed(previous_eta, arrival.minutes, TEN_MINUTE_THRESHOLD):
            logger.info(
                "[311] %s already triggered for bus %s; skipping duplicate",
                TEN_MINUTE_EMOTION,
                monitor.service_id or monitor.monitor_id,
            )
        monitor.latest_live_eta = arrival.minutes
        monitor.last_check = time.time()
        decision_latency = time.perf_counter() - decision_started
        notify_latency = 0.0
        for alert in alerts:
            notify_started = time.perf_counter()
            spoken = self._spoken_for_alert(alert, arrival, following)
            if alert == "arrival":
                new_state = classify_bus_state(arrival, arrival_confirmed=False)
                _log_bus_state(
                    arrival,
                    previous_state=previous_state,
                    new_state=new_state,
                    arrival_confirmed=False,
                    reason="eta_zero",
                )
                await self._finish_watch(
                    reason="eta_zero",
                    spoken=spoken,
                    snapshot=snapshot,
                    arrival=arrival,
                )
                notify_latency += time.perf_counter() - notify_started
                break
            await self._emit(spoken)
            notify_latency += time.perf_counter() - notify_started
            if alert == "ten" or (alert == "preparation" and monitor.preparation_threshold == TEN_MINUTE_THRESHOLD):
                logger.info(
                    "[311] Bus %s is approximately 10 minutes away",
                    monitor.service_id or monitor.monitor_id,
                )
                self._schedule_helpful1()
            if self._monitor is monitor:
                self._mark_alert_sent(monitor, alert)
        if self._monitor is monitor:
            self._persist()
        if alerts:
            logger.info(
                "BUS_QUERY latency=%.2fs DECISION latency=%.2fs NOTIFICATION latency=%.2fs END_TO_END latency=%.2fs",
                snapshot.ha_query_latency_s,
                decision_latency,
                notify_latency,
                time.perf_counter() - poll_started,
            )

    def _schedule_helpful1(self) -> None:
        if self._play_helpful1 is None:
            return
        try:
            self._helpful1_task = asyncio.create_task(self._play_helpful1_emotion(), name="bus-helpful1")
        except Exception as exc:
            logger.warning("[311] Unable to play %s emotion: %s", TEN_MINUTE_EMOTION, exc)

    async def _play_helpful1_emotion(self) -> None:
        play = self._play_helpful1
        if play is None:
            return
        logger.info("[311] Playing %s emotion", TEN_MINUTE_EMOTION)
        try:
            await play()
        except Exception as exc:
            logger.warning("[311] Unable to play %s emotion: %s", TEN_MINUTE_EMOTION, exc)
            return
        logger.info("[311] %s emotion completed", TEN_MINUTE_EMOTION)

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
