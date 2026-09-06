"""Tests for the isolated boot greeting and startup diagnostic."""

from types import SimpleNamespace
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from reachy_mini_conversation_app.local_time import (
    get_startup_time_context,
    reset_startup_time_context,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.startup_diagnostic import (
    SYDNEY_TIMEZONE,
    TimeOfDay,
    OverallStatus,
    SubsystemResult,
    DiagnosticStatus,
    evaluate_overall,
    read_sydney_clock,
    classify_time_of_day,
    deliver_boot_sequence,
    run_startup_diagnostic,
    build_report_instruction,
    reset_boot_sequence_guard,
    build_greeting_instruction,
    boot_sequence_already_delivered,
)


def _status(**fields: object) -> SimpleNamespace:
    payload: dict[str, object] = {
        "state": "running",
        "error": None,
        "simulation_enabled": True,
        "backend_status": SimpleNamespace(error=None, motor_control_mode="enabled"),
        "hardware_id": "sim",
    }
    payload.update(fields)
    return SimpleNamespace(**payload)


def _robot(
    *,
    jpeg: bytes | None = b"jpeg",
    rate: float = 16000,
    pose: object | None = None,
    joints: tuple[list[float], list[float]] | None = None,
    status: object | None = None,
    pose_error: Exception | None = None,
    status_error: Exception | None = None,
    audio: object | None = object(),
) -> SimpleNamespace:
    client = SimpleNamespace(host="127.0.0.1", port=8000)

    def get_status(*, wait: bool = False) -> object:
        if status_error is not None:
            raise status_error
        return status if status is not None else _status()

    client.get_status = get_status

    def get_current_head_pose() -> object:
        if pose_error is not None:
            raise pose_error
        return np.eye(4) if pose is None else pose

    def get_current_joint_positions() -> tuple[list[float], list[float]]:
        if pose_error is not None:
            raise pose_error
        if joints is not None:
            return joints
        return [0.0] * 7, [-0.17, 0.17]

    media = SimpleNamespace(
        audio=audio,
        get_frame_jpeg=lambda: jpeg,
        get_input_audio_samplerate=lambda: rate,
    )
    return SimpleNamespace(
        client=client,
        media=media,
        get_current_head_pose=get_current_head_pose,
        get_current_joint_positions=get_current_joint_positions,
    )


def _deps(robot: object | None = None, *, camera_enabled: bool = True) -> ToolDependencies:
    return ToolDependencies(
        reachy_mini=robot if robot is not None else _robot(),
        movement_manager=MagicMock(),
        camera_enabled=camera_enabled,
    )


class _FakeAsyncClient:
    def __init__(
        self,
        routes: dict[str, tuple[int, object | None]] | None = None,
        error_urls: tuple[str, ...] = (),
    ) -> None:
        self.routes = routes or {}
        self.error_urls = error_urls
        self.requests: list[str] = []

    async def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
        self.requests.append(url)
        for marker in self.error_urls:
            if marker in url:
                request = httpx.Request("GET", url)
                raise httpx.ConnectError("connection refused", request=request)
        for marker, (status_code, payload) in self.routes.items():
            if marker in url:
                request = httpx.Request("GET", url)
                if payload is None:
                    return httpx.Response(status_code, text="not-json", request=request)
                return httpx.Response(status_code, json=payload, request=request)
        request = httpx.Request("GET", url)
        raise httpx.ConnectError("connection refused", request=request)


def _healthy_routes() -> dict[str, tuple[int, object | None]]:
    return {
        ":8080/health": (200, {"status": "ok"}),
        ":8080/v1/models": (200, {"data": [{"id": "local-gguf"}]}),
        ":8765/v1/models": (200, {"data": []}),
        ":8765/v1/pool": (200, {"size": 1, "in_use": 0, "units": [{"state": "idle"}]}),
        ":8642/v1/models": (200, {"data": []}),
        "/api/states/": (200, {"entity_id": "sensor.route_311_at_rockwall_cres", "state": "4"}),
        "homeassistant.local": (200, {"message": "API running."}),
        ":8080/status": (200, {"probes": {"temperature": 26.1}}),
    }


@pytest.fixture(autouse=True)
def _isolate_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_boot_sequence_guard()
    reset_startup_time_context()
    monkeypatch.setattr(
        "reachy_mini_conversation_app.startup_diagnostic.get_tool_specs",
        lambda: [{"name": "camera"}],
    )
    monkeypatch.setattr("reachy_mini_conversation_app.startup_diagnostic.EMOTION_AVAILABLE", True)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.startup_diagnostic.get_hf_direct_ws_url",
        lambda: "ws://127.0.0.1:8765/v1/realtime",
    )
    monkeypatch.setattr(
        "reachy_mini_conversation_app.startup_diagnostic.config",
        SimpleNamespace(
            HERMES_GATEWAY_URL="http://127.0.0.1:8642/v1/chat/completions",
            HERMES_API_KEY="test-key-12345678",
            HA_URL="http://homeassistant.local:8123",
            HA_TOKEN="token",
            HA_BUS_ENTITY_ID="sensor.route_311_at_rockwall_cres",
            APEX_STATUS_URL="http://192.168.0.143:8080/status",
        ),
    )
    yield
    reset_boot_sequence_guard()
    reset_startup_time_context()


@pytest.mark.parametrize(
    ("hour", "period"),
    [
        (5, TimeOfDay.MORNING),
        (11, TimeOfDay.MORNING),
        (12, TimeOfDay.AFTERNOON),
        (16, TimeOfDay.AFTERNOON),
        (17, TimeOfDay.EVENING),
        (21, TimeOfDay.EVENING),
        (22, TimeOfDay.LATE_NIGHT),
        (0, TimeOfDay.LATE_NIGHT),
        (4, TimeOfDay.LATE_NIGHT),
    ],
)
def test_time_of_day_buckets(hour: int, period: TimeOfDay) -> None:
    """Boot greetings should follow the Sydney morning/afternoon/evening/late-night windows."""
    assert classify_time_of_day(hour) is period


def test_sydney_clock_names_walter_and_uses_zoneinfo() -> None:
    """The greeting should address Walter using Australia/Sydney civil time."""
    moment = datetime(2026, 8, 30, 21, 28, tzinfo=ZoneInfo(SYDNEY_TIMEZONE))
    clock = read_sydney_clock(at=moment)
    assert clock.weekday == "Sunday"
    assert clock.month == "August"
    assert clock.day_ordinal == "30th"
    assert clock.time_12h == "9:28 PM"
    assert clock.period is TimeOfDay.EVENING
    assert clock.greeting == "Good evening, Walter."
    instruction = build_greeting_instruction(clock)
    assert "Walter" in instruction
    assert "I'm Reachy Mini." in instruction
    assert "Sunday, August 30th, 9:28 PM" in instruction
    assert "Sydney" in instruction


def test_late_night_greeting() -> None:
    """Late night should keep an evening greeting and mention that Walter is up late."""
    clock = read_sydney_clock(at=datetime(2026, 8, 31, 0, 15, tzinfo=ZoneInfo(SYDNEY_TIMEZONE)))
    assert clock.period is TimeOfDay.LATE_NIGHT
    assert clock.greeting == "Good evening, Walter. You're up late."


def test_sydney_clock_uses_daylight_saving() -> None:
    """Australia/Sydney must follow DST rather than a fixed UTC offset."""
    tz = ZoneInfo(SYDNEY_TIMEZONE)
    utc = timezone.utc
    summer = datetime(2026, 1, 15, 0, 0, tzinfo=utc).astimezone(tz)
    winter = datetime(2026, 7, 15, 0, 0, tzinfo=utc).astimezone(tz)
    assert summer.utcoffset() == timedelta(hours=11)
    assert winter.utcoffset() == timedelta(hours=10)
    assert read_sydney_clock(at=summer).period is TimeOfDay.MORNING
    assert read_sydney_clock(at=winter).period is TimeOfDay.MORNING


def test_overall_status_ignores_not_configured() -> None:
    """Unset optional integrations must not be treated as boot failures."""
    results = [
        SubsystemResult("reachy_connection", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("sdk_communication", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("actuators", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("speech_to_speech", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("microphone", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("speaker", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("apex", DiagnosticStatus.NOT_CONFIGURED, critical=False),
    ]
    assert evaluate_overall(results) is OverallStatus.ALL_GREEN


def test_overall_status_minor_and_significant() -> None:
    """Optional failures are minor; critical robot/speech failures are significant."""
    minor = [
        SubsystemResult("speech_to_speech", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("hermes", DiagnosticStatus.UNAVAILABLE, critical=False),
    ]
    significant = [
        SubsystemResult("reachy_connection", DiagnosticStatus.UNAVAILABLE, critical=True),
        SubsystemResult("hermes", DiagnosticStatus.HEALTHY, critical=False),
    ]
    assert evaluate_overall(minor) is OverallStatus.MINOR_ISSUE
    assert evaluate_overall(significant) is OverallStatus.SIGNIFICANT_ISSUE


@pytest.mark.asyncio
async def test_normal_startup_checks_simulator_and_ai_stack() -> None:
    """A healthy simulator and AI stack should produce an all-green report."""
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert report.overall is OverallStatus.ALL_GREEN
    assert by_name["conversation_app"].status is DiagnosticStatus.HEALTHY
    assert by_name["local_llm"].status is DiagnosticStatus.HEALTHY
    assert by_name["speech_to_speech"].status is DiagnosticStatus.HEALTHY
    assert by_name["hermes"].status is DiagnosticStatus.HEALTHY
    assert by_name["home_assistant"].status is DiagnosticStatus.HEALTHY
    assert by_name["apex"].status is DiagnosticStatus.HEALTHY
    assert by_name["bus"].status is DiagnosticStatus.HEALTHY
    assert by_name["reachy_connection"].status is DiagnosticStatus.HEALTHY
    assert "simulator" in by_name["reachy_connection"].detail
    assert by_name["sdk_communication"].status is DiagnosticStatus.HEALTHY
    assert by_name["actuators"].status is DiagnosticStatus.HEALTHY
    assert by_name["head"].status is DiagnosticStatus.HEALTHY
    assert by_name["antennas"].status is DiagnosticStatus.HEALTHY
    assert by_name["camera"].status is DiagnosticStatus.HEALTHY
    assert by_name["microphone"].status is DiagnosticStatus.HEALTHY
    assert by_name["speaker"].status is DiagnosticStatus.HEALTHY
    instruction = build_report_instruction(report)
    assert "All systems are green" in instruction
    assert "Reachy Mini is online" in instruction
    assert "I'm Reachy Mini." in build_greeting_instruction(report.clock)
    assert "Walter" in build_greeting_instruction(report.clock)
    context = get_startup_time_context()
    assert context is not None
    assert context["timezone"] == SYDNEY_TIMEZONE
    assert context["startup_local_datetime"]
    assert context["startup_utc_offset"] in {"+10:00", "+11:00"}


@pytest.mark.asyncio
async def test_hermes_unavailable_does_not_stop_remaining_checks() -> None:
    """A Hermes failure must be isolated and must not prevent other checks."""
    routes = _healthy_routes()
    del routes[":8642/v1/models"]
    client = _FakeAsyncClient(routes, error_urls=(":8642",))
    report = await run_startup_diagnostic(_deps(), http_client=client, speech_session_open=True)
    by_name = {result.name: result for result in report.results}
    assert by_name["hermes"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["home_assistant"].status is DiagnosticStatus.HEALTHY
    assert by_name["reachy_connection"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.MINOR_ISSUE
    assert "Hermes" in build_report_instruction(report)


@pytest.mark.asyncio
async def test_home_assistant_unavailable_continues(monkeypatch: pytest.MonkeyPatch) -> None:
    """Home Assistant unavailability must not crash startup."""
    routes = _healthy_routes()
    del routes["homeassistant.local"]
    del routes["/api/states/"]
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes, error_urls=("homeassistant.local",)),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["home_assistant"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["bus"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["apex"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.MINOR_ISSUE


@pytest.mark.asyncio
async def test_apex_not_configured_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unset Apex URL should be NOT_CONFIGURED, not a boot failure."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.startup_diagnostic.config",
        SimpleNamespace(
            HERMES_GATEWAY_URL="http://127.0.0.1:8642/v1/chat/completions",
            HERMES_API_KEY="test-key-12345678",
            HA_URL="http://homeassistant.local:8123",
            HA_TOKEN="token",
            HA_BUS_ENTITY_ID="sensor.route_311_at_rockwall_cres",
            APEX_STATUS_URL="",
        ),
    )
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["apex"].status is DiagnosticStatus.NOT_CONFIGURED
    assert report.overall is OverallStatus.ALL_GREEN


@pytest.mark.asyncio
async def test_camera_unavailable_continues() -> None:
    """A missing camera frame should be reported without aborting boot."""
    report = await run_startup_diagnostic(
        _deps(_robot(jpeg=None)),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["camera"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["reachy_connection"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.MINOR_ISSUE


@pytest.mark.asyncio
async def test_hardware_status_failure_is_caught() -> None:
    """SDK/hardware exceptions must be logged as failures and must not crash boot."""
    robot = _robot(status_error=RuntimeError("daemon down"), pose_error=RuntimeError("no joints"))
    report = await run_startup_diagnostic(
        _deps(robot),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["reachy_connection"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["actuators"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["local_llm"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE
    assert "Reachy" in build_report_instruction(report)


@pytest.mark.asyncio
async def test_reachy_connection_healthy_when_daemon_state_error_is_empty() -> None:
    """A sticky daemon state=error with no error field is not a connection failure."""
    robot = _robot(status=_status(state="error", error=None))
    report = await run_startup_diagnostic(
        _deps(robot),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["reachy_connection"].status is DiagnosticStatus.HEALTHY
    assert "simulator" in by_name["reachy_connection"].detail
    assert by_name["sdk_communication"].status is DiagnosticStatus.HEALTHY
    assert by_name["actuators"].status is DiagnosticStatus.HEALTHY
    assert by_name["head"].status is DiagnosticStatus.HEALTHY
    assert by_name["antennas"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.ALL_GREEN
    instruction = build_report_instruction(report)
    assert "All systems are green" in instruction
    assert "Reachy connection" not in instruction


@pytest.mark.asyncio
async def test_reachy_connection_unhealthy_when_sdk_communication_fails() -> None:
    """A reachable daemon is not a healthy connection if head pose and joints cannot be read."""
    robot = _robot(status=_status(state="error", error=None), pose_error=RuntimeError("no joints"))
    report = await run_startup_diagnostic(
        _deps(robot),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["reachy_connection"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["sdk_communication"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["actuators"].status is DiagnosticStatus.UNAVAILABLE
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE
    assert "Reachy" in build_report_instruction(report)


@pytest.mark.asyncio
async def test_reachy_connection_not_healthy_when_healthy_daemon_cannot_read_joints() -> None:
    """Daemon state=running must not hide a functional SDK communication failure."""
    robot = _robot(status=_status(state="running"), pose_error=RuntimeError("stale bus"))
    report = await run_startup_diagnostic(
        _deps(robot),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["reachy_connection"].status is not DiagnosticStatus.HEALTHY
    assert by_name["sdk_communication"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["head"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["antennas"].status is DiagnosticStatus.UNAVAILABLE


@pytest.mark.asyncio
async def test_reachy_connection_degraded_when_daemon_reports_error_message() -> None:
    """A real daemon error string remains a connection problem even if joints are readable."""
    robot = _robot(status=_status(state="error", error="motor bus timeout"))
    report = await run_startup_diagnostic(
        _deps(robot),
        http_client=_FakeAsyncClient(_healthy_routes()),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["reachy_connection"].status is DiagnosticStatus.DEGRADED
    assert by_name["reachy_connection"].detail == "motor bus timeout"
    assert by_name["sdk_communication"].status is DiagnosticStatus.DEGRADED
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE


@pytest.mark.asyncio
async def test_llm_unavailable_is_reported() -> None:
    """llama.cpp should be marked unavailable when its health API cannot be reached."""
    routes = _healthy_routes()
    del routes[":8080/health"]
    del routes[":8080/v1/models"]
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes, error_urls=(":8080/health", ":8080/v1/models")),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["local_llm"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["speech_to_speech"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.MINOR_ISSUE


@pytest.mark.asyncio
async def test_llm_uses_speech_realtime_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """On wireless Reachy, llama.cpp lives on the AI PC host from HF_REALTIME_WS_URL."""
    monkeypatch.setattr(
        "reachy_mini_conversation_app.startup_diagnostic.get_hf_direct_ws_url",
        lambda: "ws://192.168.0.196:8765/v1/realtime",
    )
    client = _FakeAsyncClient(_healthy_routes())
    report = await run_startup_diagnostic(
        _deps(),
        http_client=client,
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["local_llm"].status is DiagnosticStatus.HEALTHY
    assert any(url.startswith("http://192.168.0.196:8080/") for url in client.requests)


@pytest.mark.asyncio
async def test_speech_models_404_with_healthy_pool_is_not_degraded() -> None:
    """A missing /v1/models route must not mark speech/STT/TTS down when /v1/pool is healthy."""
    routes = _healthy_routes()
    routes[":8765/v1/models"] = (404, {"error": "not found"})
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["speech_to_speech"].status is DiagnosticStatus.HEALTHY
    assert by_name["stt"].status is DiagnosticStatus.HEALTHY
    assert by_name["tts"].status is DiagnosticStatus.HEALTHY
    assert report.overall is OverallStatus.ALL_GREEN
    instruction = build_report_instruction(report)
    assert "All systems are green" in instruction
    assert "not fully operational" not in instruction


@pytest.mark.asyncio
async def test_speech_unavailable_when_pool_and_models_unreachable() -> None:
    """Speech must still be reported down when the realtime HTTP APIs cannot be reached."""
    routes = _healthy_routes()
    del routes[":8765/v1/models"]
    del routes[":8765/v1/pool"]
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes, error_urls=(":8765/v1/models", ":8765/v1/pool")),
        speech_session_open=False,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["speech_to_speech"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["stt"].status is DiagnosticStatus.UNAVAILABLE
    assert by_name["tts"].status is DiagnosticStatus.UNAVAILABLE
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE
    assert "speech-to-speech" in build_report_instruction(report)


@pytest.mark.asyncio
async def test_speech_degraded_when_session_open_but_http_unreachable() -> None:
    """An open session must not hide a realtime HTTP outage."""
    routes = _healthy_routes()
    del routes[":8765/v1/models"]
    del routes[":8765/v1/pool"]
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes, error_urls=(":8765/v1/models", ":8765/v1/pool")),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["speech_to_speech"].status is DiagnosticStatus.DEGRADED
    assert by_name["stt"].status is DiagnosticStatus.DEGRADED
    assert by_name["tts"].status is DiagnosticStatus.DEGRADED
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE


@pytest.mark.asyncio
async def test_speech_stuck_pool_is_degraded_even_without_models() -> None:
    """A stuck realtime slot must remain degraded when /v1/models is unimplemented."""
    routes = _healthy_routes()
    routes[":8765/v1/models"] = (404, {"error": "not found"})
    routes[":8765/v1/pool"] = (200, {"size": 1, "in_use": 1, "units": [{"state": "stuck"}]})
    report = await run_startup_diagnostic(
        _deps(),
        http_client=_FakeAsyncClient(routes),
        speech_session_open=True,
    )
    by_name = {result.name: result for result in report.results}
    assert by_name["speech_to_speech"].status is DiagnosticStatus.DEGRADED
    assert by_name["stt"].status is DiagnosticStatus.DEGRADED
    assert by_name["tts"].status is DiagnosticStatus.DEGRADED
    assert "stuck" in by_name["speech_to_speech"].detail
    assert report.overall is OverallStatus.SIGNIFICANT_ISSUE


@pytest.mark.asyncio
async def test_duplicate_boot_protection() -> None:
    """Reconnects must not speak the full startup sequence a second time."""
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    first = await deliver_boot_sequence(_deps(), speak)
    second = await deliver_boot_sequence(_deps(), speak)
    assert first is True
    assert second is True
    assert boot_sequence_already_delivered() is True
    assert len(spoken) == 2
    assert "startup diagnostics" in spoken[0]
    assert "All systems are green" in spoken[1] or "operational" in spoken[1]


@pytest.mark.asyncio
async def test_hermes_exception_does_not_abort_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected Hermes exception should be contained and the sequence should finish."""

    async def boom(_client: httpx.AsyncClient) -> SubsystemResult:
        raise RuntimeError("hermes exploded")

    monkeypatch.setattr("reachy_mini_conversation_app.startup_diagnostic._check_hermes", boom)
    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    delivered = await deliver_boot_sequence(_deps(), speak)
    assert delivered is True
    assert len(spoken) >= 1
    assert "startup diagnostics" in spoken[0]


def test_unknown_is_not_treated_as_healthy() -> None:
    """UNKNOWN results must not roll up as all-green."""
    results = [
        SubsystemResult("speech_to_speech", DiagnosticStatus.HEALTHY, critical=True),
        SubsystemResult("tools", DiagnosticStatus.UNKNOWN, critical=False),
    ]
    assert evaluate_overall(results) is OverallStatus.MINOR_ISSUE
