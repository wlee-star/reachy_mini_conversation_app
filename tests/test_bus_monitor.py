import time
import asyncio
import logging
from typing import Any
from pathlib import Path
from datetime import tzinfo, datetime, timezone

import pytest

from reachy_mini_conversation_app import bus_monitor as bus_monitor_mod
from reachy_mini_conversation_app.bus_monitor import (
    BusArrival,
    BusMonitorState,
    BusServiceState,
    LiveBusSnapshot,
    BusMonitorManager,
    same_service,
    evaluate_alerts,
    extract_arrivals,
    match_bus_intent,
    classify_bus_state,
    format_urgent_alert,
    format_arrival_alert,
    format_initial_spoken,
    format_ten_minute_alert,
    format_monitoring_started,
    format_seven_minute_alert,
    reset_bus_monitor_for_tests,
)


ENTITY = "sensor.route_311_at_rockwall_cres"


def _arrival(
    minutes: int,
    *,
    service_id: str | None = "trip-1",
    destination: str | None = "Central",
    eta_display: str | None = None,
    scheduled_at: str | None = None,
) -> BusArrival:
    return BusArrival(
        minutes=minutes,
        entity_id=ENTITY,
        route="311",
        destination=destination,
        eta_display=eta_display,
        realtime=True,
        service_id=service_id,
        stop="Macleay St @ Rockwall Cres",
        scheduled_at=scheduled_at,
    )


def _snapshot(
    minutes: int | list[int],
    *,
    error: str | None = None,
    stale: bool = False,
    service_ids: list[str] | None = None,
    eta_displays: list[str | None] | None = None,
) -> LiveBusSnapshot:
    values = [minutes] if isinstance(minutes, int) else minutes
    arrivals = [
        _arrival(
            value,
            service_id=service_ids[index] if service_ids is not None else f"trip-{index}",
            eta_display=eta_displays[index] if eta_displays is not None else None,
        )
        for index, value in enumerate(values)
    ]
    return LiveBusSnapshot(
        arrivals=[] if error else arrivals,
        entity_id=ENTITY,
        last_updated_s=time.time() - 400.0 if stale else time.time(),
        data_age_s=400.0 if stale else 1.0,
        stale=stale,
        ha_query_latency_s=0.02,
        error=error,
        fetched_at=time.time(),
    )


async def _wait_until_idle(manager: BusMonitorManager, *, timeout_s: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if manager.status()["status"] != "monitoring":
            return
        await asyncio.sleep(0.03)


async def _await_helpful1(manager: BusMonitorManager) -> None:
    task = manager._helpful1_task
    if task is not None:
        await task


def _crossing_watch(
    *,
    previous_eta: int,
    service_id: str = "trip-a",
    preparation_threshold: int = 15,
    preparation_alert_sent: bool = True,
    ten_minute_alert_sent: bool = False,
) -> BusMonitorState:
    return BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id=service_id,
        created_at=time.time(),
        latest_live_eta=previous_eta,
        preparation_threshold=preparation_threshold,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=preparation_alert_sent,
        urgent_alert_sent=False,
        ten_minute_alert_sent=ten_minute_alert_sent,
    )


@pytest.fixture(autouse=True)
def _reset_manager(tmp_path: Path) -> BusMonitorManager:
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus_monitors.v1.json")
    reset_bus_monitor_for_tests(manager)
    return manager


def test_format_initial_spoken_reports_live_eta_immediately() -> None:
    """Live arrival is spoken first at every threshold, including inside the prep window."""
    cases = {
        25: "The next 311 is currently about 25 minutes away. Would you like me to monitor it and alert you when it's within 15 minutes?",
        15: "The next 311 is currently about 15 minutes away. Would you like me to keep monitoring it and let you know when it's about 5 minutes away?",
        10: "The next 311 is currently about 10 minutes away. Would you like me to keep monitoring it and let you know when it's about 5 minutes away?",
        8: "The next 311 is currently about 8 minutes away. Would you like me to keep monitoring it and let you know when it's about 5 minutes away?",
        5: "The next 311 is arriving in about 5 minutes. Please leave now.",
        3: "The next 311 is arriving in about 3 minutes. Please leave now.",
        1: "The next 311 is arriving now. Please leave now.",
        0: "The next 311 is due now. Please leave now.",
    }
    for minutes, expected in cases.items():
        assert format_initial_spoken(_snapshot(minutes)) == expected


def test_format_initial_spoken_includes_following_service() -> None:
    """The earliest service is reported first; the next 311 is mentioned, not selected."""
    spoken = format_initial_spoken(_snapshot([8, 27]))
    assert spoken.startswith("The next 311 is currently about 8 minutes away.")
    assert "The following 311 is about 27 minutes away" in spoken


def test_extract_arrivals_keeps_multiple_departures() -> None:
    """Home Assistant next_departures lists are parsed in earliest-first order."""
    payload = {
        "entity_id": ENTITY,
        "state": "on",
        "attributes": {
            "friendly_name": "Route 311",
            "next_departures": [
                {"route_short_name": "311", "due_in": "27 min", "headsign": "Later", "trip_id": "b"},
                {
                    "route_short_name": "311",
                    "due_in": "8 min",
                    "headsign": "Soon",
                    "trip_id": "a",
                    "is_realtime": True,
                },
            ],
        },
    }
    arrivals = extract_arrivals(payload, ENTITY)
    assert [item.minutes for item in arrivals] == [8, 27]
    assert arrivals[0].service_id == "a"
    assert arrivals[0].destination == "Soon"


def test_extract_arrivals_clears_reused_trip_ids() -> None:
    """A shared Home Assistant trip id is not treated as a unique service identity."""
    payload = {
        "entity_id": ENTITY,
        "state": "on",
        "attributes": {
            "friendly_name": "Route 311",
            "next_departures": [
                {"route_short_name": "311", "due_in": "26 min", "trip_id": "201137", "display": "26:37"},
                {"route_short_name": "311", "due_in": "67 min", "trip_id": "201137", "display": "1h 07m"},
            ],
        },
    }
    arrivals = extract_arrivals(payload, ENTITY)
    assert [item.minutes for item in arrivals] == [26, 67]
    assert arrivals[0].service_id is None
    assert arrivals[1].service_id is None


def test_same_service_rejects_reused_trip_id_when_eta_jumps() -> None:
    """A later 311 that reuses the same HA trip id is not the watched service."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="201137",
        created_at=time.time(),
        latest_live_eta=5,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=True,
        urgent_alert_sent=True,
        destination="Central",
    )
    assert same_service(monitor, _arrival(4, service_id="201137")) is True
    assert same_service(monitor, _arrival(28, service_id="201137")) is False


def test_monitoring_confirmation_does_not_treat_duration_as_clock() -> None:
    """Home Assistant countdown text such as 26:37 is not spoken as a wall-clock time."""
    spoken = format_monitoring_started(_arrival(26, eta_display="26:37"))
    assert "26:37" not in spoken
    assert "arriving at approximately" in spoken or "arriving in about 26 minutes" in spoken


def test_evaluate_alerts_do_not_repeat_when_eta_oscillates() -> None:
    """A preparation alert is delivered once even if live ETA flickers around the threshold."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-1",
        created_at=time.time(),
        latest_live_eta=16,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=False,
        urgent_alert_sent=False,
    )
    assert evaluate_alerts(monitor, _arrival(15)) == ["preparation"]
    monitor.preparation_alert_sent = True
    monitor.latest_live_eta = 15
    assert evaluate_alerts(monitor, _arrival(16)) == []
    monitor.latest_live_eta = 16
    assert evaluate_alerts(monitor, _arrival(15)) == []
    monitor.latest_live_eta = 15
    assert evaluate_alerts(monitor, _arrival(14)) == []
    monitor.latest_live_eta = 14
    assert evaluate_alerts(monitor, _arrival(5)) == ["ten", "seven", "urgent"]
    monitor.ten_minute_alert_sent = True
    monitor.seven_minute_alert_sent = True
    monitor.urgent_alert_sent = True
    assert evaluate_alerts(monitor, _arrival(4)) == []
    monitor.latest_live_eta = 4
    assert evaluate_alerts(monitor, _arrival(0)) == ["arrival"]
    monitor.arrival_alert_sent = True
    assert evaluate_alerts(monitor, _arrival(0)) == []


def test_same_service_rejects_a_later_trip_when_eta_jumps_up() -> None:
    """A sudden ETA increase without a trip id is treated as a different service."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id=None,
        created_at=time.time(),
        latest_live_eta=4,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=True,
        urgent_alert_sent=True,
        destination="Central",
    )
    assert same_service(monitor, _arrival(3, service_id=None)) is True
    assert same_service(monitor, _arrival(27, service_id=None)) is False


@pytest.mark.parametrize(
    ("transcript", "pending", "active", "action"),
    [
        ("Let me know when the next 311 is coming.", False, False, "query"),
        ("Give me a 10-minute warning for the 311.", False, False, "query"),
        ("Where's my three eleven bus arriving?", False, False, "query"),
        ("When's the new three one one bus coming?", False, False, "query"),
        ("Yes.", True, False, "confirm"),
        ("What's the status of it now?", False, True, "query"),
        ("Cancel the bus reminder.", False, True, "cancel"),
        ("Stop watching the bus.", True, False, "cancel"),
        ("Monitor the next one instead.", False, True, "switch"),
        ("What about the following 311?", True, False, "switch"),
        ("Monitor the later bus.", False, True, "switch"),
        ("Keep monitoring the 311s.", False, True, "continuous"),
        ("what's the weather", False, False, None),
        ("What's the status of it now?", False, False, None),
        ("Yes.", False, False, None),
    ],
)
def test_match_bus_intent(transcript: str, pending: bool, active: bool, action: str | None) -> None:
    """Spoken bus requests map to query, confirm, or cancel without the LLM."""
    intent = match_bus_intent(transcript, pending_offer=pending, monitor_active=active)
    if action is None:
        assert intent is None
    else:
        assert intent is not None
        assert intent.action == action
        if "10" in transcript:
            assert intent.preparation_threshold == 10


@pytest.mark.asyncio
async def test_query_does_not_fabricate_eta_when_home_assistant_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing Home Assistant result is an error, not a zero-minute arrival."""

    async def _fail() -> LiveBusSnapshot:
        return _snapshot(0, error="Home Assistant is currently unavailable.")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _fail)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert result["error"] == "Home Assistant is currently unavailable."
    assert "minutes" not in result
    assert "could not read" in result["spoken"].lower()


@pytest.mark.asyncio
async def test_start_follows_live_eta_changes_and_alerts_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The watch tracks live Home Assistant minutes: 17 → 20 → 14 → 8 → 5."""
    etas = iter([17, 20, 14, 8, 5, 0])
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return _snapshot(next(etas), error=None)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    started = await manager.start(preparation_threshold=15)
    assert started["status"] == "monitoring"
    assert started["minutes"] == 17
    await _wait_until_idle(manager)
    assert any("15 minutes away" in item or "14 minutes away" in item for item in alerts)
    assert any("You better get ready to leave" in item for item in alerts)
    assert any("5 minutes away" in item and "Please leave now" in item for item in alerts)
    assert any("is due now" in item for item in alerts)
    assert sum("You have time to get ready" in item for item in alerts) == 1
    assert sum("Please leave now" in item and "5 minutes away" in item for item in alerts) == 1
    assert sum("is due now" in item for item in alerts) == 1
    assert not any("has arrived" in item for item in alerts)
    assert manager.status()["status"] == "idle"


@pytest.mark.asyncio
async def test_cancel_actually_stops_the_watch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cancel terminates the background task instead of only claiming it did."""

    async def _live() -> LiveBusSnapshot:
        return _snapshot(22)

    async def _noop(_text: str) -> None:
        return None

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(poll_s=0.05, close_poll_s=0.05, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_noop)
    await manager.start()
    assert manager.monitor_active() is True
    cancelled = await manager.cancel()
    assert cancelled["status"] == "cancelled"
    assert manager.monitor_active() is False
    assert manager._task is None or manager._task.done()


@pytest.mark.asyncio
async def test_resume_does_not_repeat_alerts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Restart recovery keeps alert flags so a 12-minute bus is not re-announced."""
    persist = tmp_path / "bus.json"
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    manager._monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-0",
        created_at=time.time(),
        latest_live_eta=12,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=True,
        urgent_alert_sent=False,
    )
    manager._persist()

    alerts: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(12)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    restored = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    restored.attach(instance_path=tmp_path, notify=_notify, persist_path=persist)
    result = await restored.resume_from_disk()
    assert result["status"] == "resumed"
    assert restored._monitor is not None
    assert restored._monitor.preparation_alert_sent is True
    await restored._poll_once()
    assert alerts == []
    await restored.cancel()


@pytest.mark.asyncio
async def test_unavailable_home_assistant_does_not_complete_the_watch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A temporary Home Assistant failure is retried without fabricating arrival."""

    async def _fail() -> LiveBusSnapshot:
        return _snapshot(0, error="Home Assistant is currently unavailable.")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _fail)
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=tmp_path / "bus.json")
    manager._monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-0",
        created_at=time.time(),
        latest_live_eta=18,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=False,
        urgent_alert_sent=False,
    )
    await manager._poll_once()
    assert manager.monitor_active() is True
    assert manager._monitor is not None
    assert manager._monitor.latest_live_eta == 18


@pytest.mark.asyncio
async def test_fetch_live_snapshot_uses_existing_ha_entity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The monitor reads the same Home Assistant 311 entity as the existing tool."""
    captured: dict[str, Any] = {}

    async def _request(
        _client: object, method: str, url: str, *, json_body: object | None = None
    ) -> tuple[object, None]:
        captured["method"] = method
        captured["url"] = url
        captured["json_body"] = json_body
        return {
            "entity_id": ENTITY,
            "state": "on",
            "last_updated": "2099-01-01T00:00:00+00:00",
            "attributes": {
                "friendly_name": "Route 311",
                "next_departures": [{"route_short_name": "311", "due_in": "22 min", "trip_id": "trip-22"}],
            },
        }, None

    monkeypatch.setattr(bus_monitor_mod, "_home_assistant_config", lambda: ("http://ha.local", "token"))
    monkeypatch.setattr(bus_monitor_mod, "_bus_entity_id", lambda _kwargs: ENTITY)
    monkeypatch.setattr(bus_monitor_mod, "_request_json", _request)
    snapshot = await bus_monitor_mod.fetch_live_snapshot()
    assert captured["method"] == "GET"
    assert captured["url"] == f"http://ha.local/api/states/{ENTITY}"
    assert captured["json_body"] is None
    assert snapshot.next_arrival is not None
    assert snapshot.next_arrival.minutes == 22
    assert snapshot.error is None


def test_urgent_message_is_deterministic() -> None:
    """The 5-minute alert does not require the LLM to invent wording."""
    assert format_urgent_alert(_arrival(5)) == "The 311 is now about 5 minutes away. Please leave now."
    assert format_urgent_alert(_arrival(1)) == "The 311 is arriving now. Please leave now."
    assert format_ten_minute_alert(_arrival(10)) == "The 311 is now about 10 minutes away."
    assert format_seven_minute_alert(_arrival(7)) == (
        "The 311 is now about 7 minutes away. You better get ready to leave."
    )
    assert format_arrival_alert(_arrival(0)) == "The 311 is due now."
    assert format_arrival_alert(None) == (
        "The live 311 feed has lost that service. I can't confirm that it has arrived."
    )
    assert format_arrival_alert(_arrival(0), confirmed=True) == "The 311 has arrived."


def test_extract_arrivals_keeps_zero_minute_eta() -> None:
    """A due-now service is a valid arrival, not missing data."""
    payload = {
        "entity_id": ENTITY,
        "state": "300",
        "attributes": {
            "friendly_name": "Route 311",
            "next_departures": [
                {"route_short_name": "311", "minutes": 0, "headsign": "City", "trip_id": "due"},
                {"route_short_name": "311", "due_in": "28 min", "headsign": "Later", "trip_id": "next"},
            ],
        },
    }
    arrivals = extract_arrivals(payload, ENTITY)
    assert [item.minutes for item in arrivals] == [0, 28]
    assert arrivals[0].service_id == "due"


def test_extract_arrivals_empty_schedule_does_not_use_sensor_state() -> None:
    """A cleared next_departures list is empty, not leftover sensor-state minutes."""
    payload = {
        "entity_id": ENTITY,
        "state": "300",
        "last_updated": "2099-01-01T00:00:00+00:00",
        "attributes": {
            "friendly_name": "Route 311",
            "next_departures": [],
        },
    }
    assert extract_arrivals(payload, ENTITY) == []


@pytest.mark.asyncio
async def test_live_failure_22_15_5_terminal_then_next_28(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """22 → 15 → 5 → 0 → 28: one prep, one urgent, one arrival, then a fresh 28-minute query."""
    snapshots = iter(
        [
            _snapshot(22, service_ids=["trip-a"]),
            _snapshot(15, service_ids=["trip-a"]),
            _snapshot(5, service_ids=["trip-a"]),
            _snapshot(0, service_ids=["trip-a"]),
            _snapshot(28, service_ids=["trip-b"]),
            _snapshot(28, service_ids=["trip-b"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    started = await manager.start(preparation_threshold=15)
    assert started["status"] == "monitoring"
    assert started["minutes"] == 22
    await _wait_until_idle(manager)
    assert sum("You have time to get ready" in item for item in alerts) == 1
    assert sum("You better get ready to leave" in item for item in alerts) == 1
    assert sum("5 minutes away" in item and "Please leave now" in item for item in alerts) == 1
    assert sum("is due now" in item for item in alerts) == 1
    assert not any("has arrived" in item for item in alerts)
    queried = await manager.query()
    assert queried["minutes"] == 28
    assert "28 minutes away" in queried["spoken"]
    assert manager.status()["status"] == "idle"


@pytest.mark.asyncio
async def test_skipping_zero_from_one_minute_to_next_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Polling may miss 0 and jump from 1 minute to the following service."""
    snapshots = iter(
        [
            _snapshot(8, service_ids=["trip-a"]),
            _snapshot(5, service_ids=["trip-a"]),
            _snapshot(4, service_ids=["trip-a"]),
            _snapshot(3, service_ids=["trip-a"]),
            _snapshot(2, service_ids=["trip-a"]),
            _snapshot(1, service_ids=["trip-a"]),
            _snapshot(28, service_ids=["trip-b"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await _wait_until_idle(manager)
    assert sum("Please leave now" in item and "5 minutes away" in item for item in alerts) == 1
    assert sum("can't confirm that it has arrived" in item for item in alerts) == 1
    assert not any(item == "The 311 has arrived." for item in alerts)
    assert manager.status()["status"] == "idle"


@pytest.mark.asyncio
async def test_skipping_zero_from_five_minutes_to_next_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 5-minute watch must complete when Home Assistant replaces that service."""
    snapshots = iter(
        [
            _snapshot(8, service_ids=["trip-a"]),
            _snapshot(5, service_ids=["trip-a"]),
            _snapshot(28, service_ids=["trip-b"]),
            _snapshot(28, service_ids=["trip-b"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await _wait_until_idle(manager)
    assert sum("Please leave now" in item and "5 minutes away" in item for item in alerts) == 1
    assert sum("can't confirm that it has arrived" in item for item in alerts) == 1
    assert not any(item == "The 311 has arrived." for item in alerts)
    queried = await manager.query()
    assert queried["minutes"] == 28
    assert manager.monitor_active() is False


@pytest.mark.asyncio
async def test_stale_five_minute_query_does_not_repeat_eta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale Home Assistant minutes are not spoken as the current 311 time."""

    async def _stale() -> LiveBusSnapshot:
        return _snapshot(5, stale=True)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _stale)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert "minutes" not in result
    assert result["error"] == "Bus arrival data is stale"
    assert "fresh" in result["spoken"].lower()


@pytest.mark.asyncio
async def test_zero_minute_alert_is_sent_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Repeated due-now polls produce a single terminal notification."""
    snapshots = iter(
        [
            _snapshot(6, service_ids=["trip-a"]),
            _snapshot(0, service_ids=["trip-a"]),
            _snapshot(0, service_ids=["trip-a"]),
            _snapshot(0, service_ids=["trip-a"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await _wait_until_idle(manager)
    await asyncio.sleep(0.08)
    assert sum("is due now" in item for item in alerts) == 1
    assert not any("has arrived" in item for item in alerts)


@pytest.mark.asyncio
async def test_cancel_does_not_send_terminal_alert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Stopping the watch must not produce a later arrival notice."""
    alerts: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(8, service_ids=["trip-a"])

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    cancelled = await manager.cancel()
    assert cancelled["status"] == "cancelled"
    await manager._poll_once()
    assert alerts == []


@pytest.mark.asyncio
async def test_resume_after_urgent_does_not_repeat_or_follow_next_bus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Restart recovery keeps a 4-minute watch and does not adopt a later 28-minute service."""
    persist = tmp_path / "bus.json"
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    manager._monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-a",
        created_at=time.time(),
        latest_live_eta=4,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=True,
        urgent_alert_sent=True,
        arrival_alert_sent=False,
    )
    manager._persist()
    alerts: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(4, service_ids=["trip-a"])

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    restored = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    restored.attach(instance_path=tmp_path, notify=_notify, persist_path=persist)
    result = await restored.resume_from_disk()
    assert result["status"] == "resumed"
    assert restored._monitor is not None
    assert restored._monitor.urgent_alert_sent is True
    await restored._poll_once()
    assert alerts == []

    async def _next_bus() -> LiveBusSnapshot:
        return _snapshot(28, service_ids=["trip-b"])

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next_bus)
    await restored._poll_once()
    assert any("can't confirm that it has arrived" in item for item in alerts)
    assert not any("The 311 has arrived." in item for item in alerts)
    assert restored.status()["status"] == "idle"
    await restored.cancel()


@pytest.mark.asyncio
async def test_resume_does_not_restore_a_completed_watch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A persisted watch that already sent the arrival alert is not resumed."""
    persist = tmp_path / "bus.json"
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    manager._monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-a",
        created_at=time.time(),
        latest_live_eta=0,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=True,
        urgent_alert_sent=True,
        arrival_alert_sent=True,
    )
    manager._persist()

    async def _live() -> LiveBusSnapshot:
        return _snapshot(28, service_ids=["trip-b"])

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    restored = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=persist)
    result = await restored.resume_from_disk()
    assert result["status"] == "expired"
    assert restored.monitor_active() is False


@pytest.mark.asyncio
async def test_empty_schedule_completes_the_watch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cleared Home Assistant schedule is terminal, not a reason to keep the last ETA."""
    snapshots = iter(
        [
            _snapshot(8, service_ids=["trip-a"]),
            _snapshot(5, service_ids=["trip-a"]),
            _snapshot([]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await _wait_until_idle(manager)
    assert any("can't confirm that it has arrived" in item for item in alerts)
    assert not any("The 311 has arrived." in item for item in alerts)
    assert manager.status()["status"] == "idle"


def test_format_initial_spoken_includes_following_when_current_is_close() -> None:
    """A 2-minute 311 still reports the following service instead of hiding it."""
    spoken = format_initial_spoken(_snapshot([2, 18]))
    assert spoken.startswith("The next 311 is arriving in about 2 minutes. Please leave now.")
    assert "The following 311 is about 18 minutes away" in spoken


def test_format_urgent_alert_includes_following_when_available() -> None:
    """Leave-now wording keeps the later 311 visible."""
    spoken = format_urgent_alert(_arrival(2), _arrival(17, service_id="trip-2"))
    assert "2 minutes away" in spoken
    assert "Please leave now" in spoken
    assert "The following 311 is about 17 minutes away" in spoken


def test_format_monitoring_started_names_the_service() -> None:
    """Confirmation names the watched 311 instead of a generic okay."""
    spoken = format_monitoring_started(_arrival(22, eta_display="8:42"))
    assert spoken == "Okay. I'm monitoring the 311 arriving at approximately 8:42."


def test_evaluate_alerts_missed_thresholds_still_fire_in_order() -> None:
    """A skipped poll still raises 15, 10, 7, and 5 in crossing order."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-1",
        created_at=time.time(),
        latest_live_eta=16,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=False,
        urgent_alert_sent=False,
    )
    assert evaluate_alerts(monitor, _arrival(12)) == ["preparation"]
    monitor.preparation_alert_sent = True
    monitor.latest_live_eta = 12
    assert evaluate_alerts(monitor, _arrival(9)) == ["ten"]
    monitor.ten_minute_alert_sent = True
    monitor.latest_live_eta = 9
    assert evaluate_alerts(monitor, _arrival(6)) == ["seven"]
    monitor.seven_minute_alert_sent = True
    monitor.latest_live_eta = 6
    assert evaluate_alerts(monitor, _arrival(4)) == ["urgent"]


def test_evaluate_alerts_do_not_fire_retrospective_alerts_from_eight_minutes() -> None:
    """Starting at 8 minutes watches 7 and 5, and does not replay 15 or 10."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-1",
        created_at=time.time(),
        latest_live_eta=8,
        preparation_threshold=15,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=False,
        urgent_alert_sent=False,
    )
    assert evaluate_alerts(monitor, _arrival(8)) == []
    assert evaluate_alerts(monitor, _arrival(7)) == ["seven"]


def test_evaluate_alerts_ten_minute_prep_does_not_duplicate() -> None:
    """A 10-minute preparation threshold is one message, not prep plus ten."""
    monitor = BusMonitorState(
        monitor_id="m1",
        entity_id=ENTITY,
        route="311",
        stop=None,
        service_id="trip-1",
        created_at=time.time(),
        latest_live_eta=12,
        preparation_threshold=10,
        urgent_threshold=5,
        status="active",
        last_check=time.time(),
        preparation_alert_sent=False,
        urgent_alert_sent=False,
    )
    assert evaluate_alerts(monitor, _arrival(10)) == ["preparation"]


@pytest.mark.asyncio
async def test_query_22_minute_bus_offers_monitoring(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 22-minute live 311 is spoken first and offers monitoring."""

    async def _live() -> LiveBusSnapshot:
        return _snapshot(22, service_ids=["trip-22"], eta_displays=["8:42"])

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert result["minutes"] == 22
    assert result["offer"] == "offer_prepare"
    assert "22 minutes away" in result["spoken"]
    assert "Would you like me to monitor it" in result["spoken"]
    assert result["timezone"] == "Australia/Sydney"


@pytest.mark.asyncio
async def test_start_confirms_the_monitored_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """After the user confirms, Reachy names the watched 311."""

    async def _live() -> LiveBusSnapshot:
        return _snapshot(22, service_ids=["trip-22"], eta_displays=["8:42"])

    async def _noop(_text: str) -> None:
        return None

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_noop)
    started = await manager.start(preparation_threshold=15)
    assert started["status"] == "monitoring"
    assert started["spoken"] == "Okay. I'm monitoring the 311 arriving at approximately 8:42."
    await manager.cancel()


@pytest.mark.asyncio
async def test_missed_threshold_poll_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """16 → 15 → 12 → 9 → 6 → 4 still produces 15, 10, 7, and 5 alerts."""
    snapshots = iter(
        [
            _snapshot(16, service_ids=["trip-a"]),
            _snapshot(15, service_ids=["trip-a"]),
            _snapshot(12, service_ids=["trip-a"]),
            _snapshot(9, service_ids=["trip-a"]),
            _snapshot(6, service_ids=["trip-a"]),
            _snapshot(4, service_ids=["trip-a"]),
            _snapshot(0, service_ids=["trip-a"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await _wait_until_idle(manager)
    assert sum("You have time to get ready" in item for item in alerts) == 1
    assert sum(item.endswith("10 minutes away.") or "now about 9 minutes away." in item for item in alerts) == 1
    assert sum("You better get ready to leave" in item for item in alerts) == 1
    assert sum("Please leave now" in item for item in alerts) == 1
    assert sum("is due now" in item for item in alerts) == 1
    assert not any("has arrived" in item for item in alerts)


@pytest.mark.asyncio
async def test_four_minute_start_skips_retrospective_alerts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Starting at 4 minutes does not replay 15/10/7 and still reaches terminal."""
    snapshots = iter(
        [
            _snapshot(4, service_ids=["trip-a"], eta_displays=["9:05"]),
            _snapshot(3, service_ids=["trip-a"]),
            _snapshot(28, service_ids=["trip-b"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    started = await manager.start()
    assert "arriving at approximately 9:05" in started["spoken"]
    await _wait_until_idle(manager)
    assert not any("You have time to get ready" in item for item in alerts)
    assert not any("You better get ready to leave" in item for item in alerts)
    assert any("can't confirm that it has arrived" in item for item in alerts)
    assert not any("The 311 has arrived." in item for item in alerts)


@pytest.mark.asyncio
async def test_query_uses_fresh_ha_not_monitor_eta(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A later user query reads Home Assistant again instead of the last 5-minute alert."""
    snapshots = iter(
        [
            _snapshot(5, service_ids=["trip-a"]),
            _snapshot(28, service_ids=["trip-b"]),
        ]
    )

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _noop(_text: str) -> None:
        return None

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_noop)
    await manager.start()
    assert manager.status()["minutes"] == 5
    queried = await manager.query()
    assert queried["minutes"] == 28
    assert "28 minutes away" in queried["spoken"]
    assert "about 5 minutes" not in queried["spoken"]
    await manager.cancel()


@pytest.mark.asyncio
async def test_switch_moves_to_the_following_service(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An explicit switch cancels the old watch and names the later 311."""
    snapshots = iter(
        [
            _snapshot([22, 45], service_ids=["trip-a", "trip-b"], eta_displays=["8:42", "9:05"]),
            _snapshot([21, 44], service_ids=["trip-a", "trip-b"], eta_displays=["8:42", "9:05"]),
        ]
    )

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _noop(_text: str) -> None:
        return None

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=5.0, close_poll_s=5.0, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_noop)
    started = await manager.start()
    assert started["service_id"] == "trip-a"
    switched = await manager.switch()
    assert switched["status"] == "monitoring"
    assert switched["service_id"] == "trip-b"
    assert "switched the monitor" in switched["spoken"]
    assert "9:05" in switched["spoken"]
    assert manager._monitor is not None
    assert manager._monitor.preparation_alert_sent is False
    await manager.cancel()


@pytest.mark.asyncio
async def test_following_bus_does_not_replace_monitored_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A later 311 is informational only until the user asks to switch."""
    snapshots = iter(
        [
            _snapshot([8, 27], service_ids=["trip-a", "trip-b"]),
            _snapshot([2, 18], service_ids=["trip-a", "trip-b"]),
            _snapshot([1, 17], service_ids=["trip-a", "trip-b"]),
            _snapshot(16, service_ids=["trip-b"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    started = await manager.start()
    assert started["service_id"] == "trip-a"
    await _wait_until_idle(manager)
    assert any("18 minutes away" in item or "17 minutes away" in item for item in alerts)
    assert any("can't confirm that it has arrived" in item for item in alerts)
    assert not any("The 311 has arrived." in item for item in alerts)
    assert manager.status()["status"] == "idle"


@pytest.mark.asyncio
async def test_continuous_handoff_requires_explicit_request(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Continuous mode may follow the next 311; the default watch does not."""
    snapshots = iter(
        [
            _snapshot([6, 18], service_ids=["trip-a", "trip-b"], eta_displays=["8:50", "9:05"]),
            _snapshot(17, service_ids=["trip-b"], eta_displays=["9:05"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    started = await manager.start(continuous=True)
    assert started["continuous"] is True
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        monitor = manager._monitor
        if monitor is not None and monitor.service_id == "trip-b":
            break
        await asyncio.sleep(0.03)
    assert manager._monitor is not None
    assert manager._monitor.service_id == "trip-b"
    assert any("can't confirm that it has arrived" in item and "9:05" in item for item in alerts)
    assert not any(item.startswith("The 311 has arrived") for item in alerts)
    await manager.cancel()


@pytest.mark.asyncio
async def test_eta_oscillation_sends_one_preparation_alert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """15 → 16 → 14 produces a single preparation notification."""
    snapshots = iter(
        [
            _snapshot(16, service_ids=["trip-a"]),
            _snapshot(15, service_ids=["trip-a"]),
            _snapshot(16, service_ids=["trip-a"]),
            _snapshot(14, service_ids=["trip-a"]),
        ]
    )
    alerts: list[str] = []

    async def _next() -> LiveBusSnapshot:
        return next(snapshots)

    async def _notify(text: str) -> None:
        alerts.append(text)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _next)
    manager = BusMonitorManager(poll_s=0.02, close_poll_s=0.02, persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, persist_path=tmp_path / "bus.json")
    await manager.start()
    await asyncio.sleep(0.2)
    assert sum("You have time to get ready" in item for item in alerts) == 1
    assert manager.monitor_active() is True
    await manager.cancel()


def test_approx_clock_uses_australia_sydney(monkeypatch: pytest.MonkeyPatch) -> None:
    """Approximate clock labels are Australia/Sydney civil time, not a fixed UTC offset."""

    class _FrozenDateTime:
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            moment = datetime(2026, 8, 31, 3, 30, tzinfo=timezone.utc)
            return moment if tz is None else moment.astimezone(tz)

    monkeypatch.setattr(bus_monitor_mod, "datetime", _FrozenDateTime)
    assert bus_monitor_mod._approx_clock(7) == "1:37"


def test_clock_label_converts_utc_iso_to_sydney() -> None:
    """ISO arrival timestamps are spoken in Australia/Sydney, not UTC."""
    arrival = _arrival(22, scheduled_at="2026-08-31T03:22:00Z")
    assert bus_monitor_mod._clock_label(arrival) == "13:22"


def test_format_initial_spoken_uses_sydney_clock_from_utc_iso() -> None:
    """Spoken 'expected at' times come from the Sydney clock, not the raw UTC string."""
    snapshot = _snapshot(22, eta_displays=["2026-08-31T03:22:00Z"])
    spoken = format_initial_spoken(snapshot)
    assert "03:22" not in spoken
    assert "expected at 13:22" in spoken


@pytest.mark.asyncio
async def test_ten_minute_alert_plays_helpful1_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Crossing 10 minutes speaks the existing notice and queues helpful1 once."""
    events: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(10, service_ids=["trip-a"])

    async def _notify(text: str) -> None:
        events.append(f"notify:{text}")

    async def _play() -> None:
        events.append("helpful1")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, play_helpful1=_play)
    manager._monitor = _crossing_watch(previous_eta=12)
    await manager._poll_once()
    await _await_helpful1(manager)
    assert any("10 minutes away" in item for item in events)
    assert events[-1] == "helpful1"
    assert sum(item == "helpful1" for item in events) == 1
    assert manager._monitor is not None
    assert manager._monitor.ten_minute_alert_sent is True


@pytest.mark.asyncio
async def test_ten_minute_helpful1_does_not_replay_while_eta_stays_near_ten(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """helpful1 stays one-shot if the watched 311 remains around 10 minutes."""
    played: list[str] = []
    etas = iter([10, 10, 11, 10])

    async def _live() -> LiveBusSnapshot:
        return _snapshot(next(etas), service_ids=["trip-a"])

    async def _notify(_text: str) -> None:
        return None

    async def _play() -> None:
        played.append("helpful1")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, play_helpful1=_play)
    manager._monitor = _crossing_watch(previous_eta=12)
    with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.bus_monitor"):
        await manager._poll_once()
        await _await_helpful1(manager)
        await manager._poll_once()
        await manager._poll_once()
        await manager._poll_once()
        await _await_helpful1(manager)
    assert played == ["helpful1"]
    assert "helpful1 already triggered for bus trip-a; skipping duplicate" in caplog.text


@pytest.mark.asyncio
async def test_ten_minute_helpful1_plays_again_for_a_new_bus(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A later 311 crossing 10 minutes plays helpful1 again."""
    played: list[str] = []
    current = {"minutes": 10, "service_id": "trip-a"}

    async def _live() -> LiveBusSnapshot:
        return _snapshot(current["minutes"], service_ids=[str(current["service_id"])])

    async def _notify(_text: str) -> None:
        return None

    async def _play() -> None:
        played.append(str(current["service_id"]))

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, play_helpful1=_play)
    manager._monitor = _crossing_watch(previous_eta=12, service_id="trip-a")
    await manager._poll_once()
    await _await_helpful1(manager)
    current["service_id"] = "trip-b"
    manager._monitor = _crossing_watch(previous_eta=12, service_id="trip-b")
    await manager._poll_once()
    await _await_helpful1(manager)
    assert played == ["trip-a", "trip-b"]


@pytest.mark.asyncio
async def test_ten_minute_helpful1_failure_does_not_block_notification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed helpful1 load still speaks the 10-minute notice and keeps the watch."""
    alerts: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(10, service_ids=["trip-a"])

    async def _notify(text: str) -> None:
        alerts.append(text)

    async def _fail() -> None:
        raise RuntimeError("dataset missing")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, play_helpful1=_fail)
    manager._monitor = _crossing_watch(previous_eta=12)
    with caplog.at_level(logging.WARNING, logger="reachy_mini_conversation_app.bus_monitor"):
        await manager._poll_once()
        await _await_helpful1(manager)
    assert any("10 minutes away" in item for item in alerts)
    assert manager.monitor_active() is True
    assert "Unable to play helpful1 emotion: dataset missing" in caplog.text


@pytest.mark.asyncio
async def test_ten_minute_prep_threshold_also_plays_helpful1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A 10-minute preparation watch still queues helpful1 with that existing notice."""
    played: list[str] = []
    alerts: list[str] = []

    async def _live() -> LiveBusSnapshot:
        return _snapshot(10, service_ids=["trip-a"])

    async def _notify(text: str) -> None:
        alerts.append(text)

    async def _play() -> None:
        played.append("helpful1")

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    manager = BusMonitorManager(persist_path=tmp_path / "bus.json")
    manager.attach(instance_path=tmp_path, notify=_notify, play_helpful1=_play)
    manager._monitor = _crossing_watch(
        previous_eta=12,
        preparation_threshold=10,
        preparation_alert_sent=False,
    )
    await manager._poll_once()
    await _await_helpful1(manager)
    assert played == ["helpful1"]
    assert any("10 minutes away" in item for item in alerts)
    assert any("You have time to get ready" in item for item in alerts)


def test_zero_minutes_is_arriving_not_arrived() -> None:
    """A 0-minute ETA is due/arriving, not a confirmed arrival."""
    arrival = _arrival(0)
    assert classify_bus_state(arrival) is BusServiceState.ARRIVING
    assert format_arrival_alert(arrival) == "The 311 is due now."
    assert format_arrival_alert(arrival, confirmed=False) != "The 311 has arrived."


def test_service_gone_is_not_arrived() -> None:
    """A vanished service is an unconfirmed feed loss, not an arrival."""
    assert classify_bus_state(None, reason="service_gone") is BusServiceState.SERVICE_GONE
    spoken = format_arrival_alert(None)
    assert "can't confirm" in spoken
    assert spoken != "The 311 has arrived."


@pytest.mark.asyncio
async def test_query_reports_twelve_minutes_and_route_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 12-minute live 311 is reported as 12 minutes with route, stop, and direction."""

    async def _live() -> LiveBusSnapshot:
        return _snapshot(12, service_ids=["trip-12"], eta_displays=["12:00"])

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert result["route"] == "311"
    assert result["stop"] == "Macleay St @ Rockwall Cres"
    assert result["direction"] == "Central"
    assert result["next_minutes"] == 12
    assert result["minutes"] == 12
    assert result["eta_display"] == "12:00"
    assert result["realtime"] is True
    assert result["service_state"] == BusServiceState.UPCOMING.value
    assert result["arrival_confirmed"] is False
    assert "12 minutes away" in result["spoken"]
    assert "has arrived" not in result["spoken"]


@pytest.mark.asyncio
async def test_query_does_not_treat_zero_minutes_as_arrived(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking about a due-now 311 must not claim it has arrived."""

    async def _live() -> LiveBusSnapshot:
        return _snapshot(0, service_ids=["due"])

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _live)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert result["next_minutes"] == 0
    assert result["service_state"] == BusServiceState.ARRIVING.value
    assert result["arrival_confirmed"] is False
    assert "due now" in result["spoken"].lower()
    assert "has arrived" not in result["spoken"]


@pytest.mark.asyncio
async def test_stale_query_is_unknown_not_fact(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stale Home Assistant data is unknown, not spoken as a live arrival."""

    async def _stale() -> LiveBusSnapshot:
        return _snapshot(12, stale=True)

    monkeypatch.setattr(bus_monitor_mod, "fetch_live_snapshot", _stale)
    result = await bus_monitor_mod.get_bus_monitor().query()
    assert result["service_state"] == BusServiceState.UNKNOWN.value
    assert result["arrival_confirmed"] is False
    assert "minutes" not in result


@pytest.mark.asyncio
async def test_arrival_alert_only_when_confirmed() -> None:
    """The arrived sentence is reserved for an explicit confirmation."""
    assert format_arrival_alert(_arrival(0), confirmed=True) == "The 311 has arrived."
    assert format_arrival_alert(_arrival(0), confirmed=False) == "The 311 is due now."
