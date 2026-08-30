import time
import asyncio
from typing import Any
from pathlib import Path

import pytest

from reachy_mini_conversation_app import bus_monitor as bus_monitor_mod
from reachy_mini_conversation_app.bus_monitor import (
    BusArrival,
    BusMonitorState,
    LiveBusSnapshot,
    BusMonitorManager,
    same_service,
    evaluate_alerts,
    extract_arrivals,
    match_bus_intent,
    format_urgent_alert,
    format_initial_spoken,
    reset_bus_monitor_for_tests,
)


ENTITY = "sensor.route_311_at_rockwall_cres"


def _arrival(minutes: int, *, service_id: str | None = "trip-1", destination: str | None = "Central") -> BusArrival:
    return BusArrival(
        minutes=minutes,
        entity_id=ENTITY,
        route="311",
        destination=destination,
        eta_display=None,
        realtime=True,
        service_id=service_id,
        stop="Macleay St @ Rockwall Cres",
    )


def _snapshot(minutes: int | list[int], *, error: str | None = None, stale: bool = False) -> LiveBusSnapshot:
    values = [minutes] if isinstance(minutes, int) else minutes
    arrivals = [_arrival(value, service_id=f"trip-{index}") for index, value in enumerate(values)]
    return LiveBusSnapshot(
        arrivals=[] if error else arrivals,
        entity_id=ENTITY,
        last_updated_s=time.time(),
        data_age_s=1.0,
        stale=stale,
        ha_query_latency_s=0.02,
        error=error,
        fetched_at=time.time(),
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
    assert evaluate_alerts(monitor, _arrival(15)) == "preparation"
    monitor.preparation_alert_sent = True
    assert evaluate_alerts(monitor, _arrival(16)) is None
    assert evaluate_alerts(monitor, _arrival(15)) is None
    assert evaluate_alerts(monitor, _arrival(14)) is None
    assert evaluate_alerts(monitor, _arrival(5)) == "urgent"


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
        ("Yes.", True, False, "confirm"),
        ("Cancel the bus reminder.", False, True, "cancel"),
        ("Stop watching the bus.", True, False, "cancel"),
        ("what's the weather", False, False, None),
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
    etas = iter([17, 20, 14, 8, 5])
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
    for _ in range(40):
        if manager.status()["status"] != "monitoring":
            break
        await asyncio.sleep(0.03)
    assert any("15 minutes away" in item or "14 minutes away" in item for item in alerts)
    assert any("Please leave now" in item for item in alerts)
    assert sum("You have time to get ready" in item for item in alerts) == 1
    assert sum("Please leave now" in item for item in alerts) == 1
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
