"""Isolated boot greeting, Sydney clock, and startup diagnostics."""

import asyncio
import logging
from enum import Enum
from datetime import datetime
from zoneinfo import ZoneInfo
from dataclasses import dataclass
from urllib.parse import quote, urlsplit
from collections.abc import Callable, Sequence, Awaitable

import httpx
import numpy as np

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.config import (
    config,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.tools.play_emotion import EMOTION_AVAILABLE


logger = logging.getLogger(__name__)

USER_DISPLAY_NAME = "Walter"
SYDNEY_TIMEZONE = "Australia/Sydney"
_HTTP_TIMEOUT_S = 3.0
_LLAMA_HOST = "127.0.0.1"
_LLAMA_PORT = 8080
_boot_sequence_delivered = False

_CRITICAL_NAMES = frozenset(
    {
        "reachy_connection",
        "sdk_communication",
        "actuators",
        "speech_to_speech",
        "microphone",
        "speaker",
    }
)
_SPOKEN_NAMES = {
    "reachy_connection": "the Reachy connection",
    "sdk_communication": "Reachy SDK communication",
    "actuators": "Reachy actuators",
    "head": "the head",
    "antennas": "the antennas",
    "camera": "the camera",
    "microphone": "the microphone",
    "speaker": "the speaker",
    "conversation_app": "the conversation app",
    "local_llm": "the local LLM",
    "speech_to_speech": "speech-to-speech",
    "stt": "speech recognition",
    "tts": "speech synthesis",
    "hermes": "Hermes",
    "home_assistant": "Home Assistant",
    "apex": "Neptune Apex",
    "bus": "the bus integration",
    "tools": "the tool layer",
    "emotion": "the emotion system",
}


class DiagnosticStatus(str, Enum):
    """Structured status for one startup subsystem."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OverallStatus(str, Enum):
    """Roll-up of a completed startup diagnostic."""

    ALL_GREEN = "ALL_GREEN"
    MINOR_ISSUE = "MINOR_ISSUE"
    SIGNIFICANT_ISSUE = "SIGNIFICANT_ISSUE"


class TimeOfDay(str, Enum):
    """Sydney time-of-day bucket used for the boot greeting."""

    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening"
    LATE_NIGHT = "late_night"


@dataclass(frozen=True)
class SubsystemResult:
    """Result of one subsystem check."""

    name: str
    status: DiagnosticStatus
    detail: str = ""
    critical: bool = False


@dataclass(frozen=True)
class SydneyClock:
    """Current civil time in Australia/Sydney."""

    moment: datetime
    weekday: str
    month: str
    day: int
    day_ordinal: str
    time_12h: str
    period: TimeOfDay

    @property
    def spoken_datetime(self) -> str:
        """Return a short spoken date and time."""
        return f"{self.weekday}, {self.month} {self.day_ordinal}, {self.time_12h}"

    @property
    def greeting(self) -> str:
        """Return the time-of-day greeting addressed to Walter."""
        if self.period is TimeOfDay.LATE_NIGHT:
            return f"Good evening, {USER_DISPLAY_NAME}. You're up late."
        label = {
            TimeOfDay.MORNING: "Good morning",
            TimeOfDay.AFTERNOON: "Good afternoon",
            TimeOfDay.EVENING: "Good evening",
        }[self.period]
        return f"{label}, {USER_DISPLAY_NAME}."


@dataclass(frozen=True)
class StartupReport:
    """Complete structured startup diagnostic."""

    clock: SydneyClock
    results: tuple[SubsystemResult, ...]
    overall: OverallStatus


def reset_boot_sequence_guard() -> None:
    """Allow another spoken boot sequence. Tests only."""
    global _boot_sequence_delivered
    _boot_sequence_delivered = False


def boot_sequence_already_delivered() -> bool:
    """Return whether this process already spoke the boot sequence."""
    return _boot_sequence_delivered


def classify_time_of_day(hour: int) -> TimeOfDay:
    """Classify a 24-hour clock hour into the boot greeting period."""
    if 5 <= hour < 12:
        return TimeOfDay.MORNING
    if 12 <= hour < 17:
        return TimeOfDay.AFTERNOON
    if 17 <= hour < 22:
        return TimeOfDay.EVENING
    return TimeOfDay.LATE_NIGHT


def read_sydney_clock(at: datetime | None = None) -> SydneyClock:
    """Read the current civil time in Australia/Sydney, including DST."""
    tz = ZoneInfo(SYDNEY_TIMEZONE)
    if at is None:
        moment = datetime.now(tz)
    elif at.tzinfo is None:
        moment = at.replace(tzinfo=tz)
    else:
        moment = at.astimezone(tz)
    return SydneyClock(
        moment=moment,
        weekday=moment.strftime("%A"),
        month=moment.strftime("%B"),
        day=moment.day,
        day_ordinal=_ordinal(moment.day),
        time_12h=moment.strftime("%I:%M %p").lstrip("0"),
        period=classify_time_of_day(moment.hour),
    )


def evaluate_overall(results: Sequence[SubsystemResult]) -> OverallStatus:
    """Roll subsystem results into the spoken overall boot state."""
    significant = False
    minor = False
    for result in results:
        if result.status in {DiagnosticStatus.NOT_CONFIGURED, DiagnosticStatus.NOT_APPLICABLE}:
            continue
        if result.status is DiagnosticStatus.HEALTHY:
            continue
        if result.critical:
            significant = True
        else:
            minor = True
    if significant:
        return OverallStatus.SIGNIFICANT_ISSUE
    if minor:
        return OverallStatus.MINOR_ISSUE
    return OverallStatus.ALL_GREEN


def build_greeting_instruction(clock: SydneyClock) -> str:
    """Return the first spoken boot instruction for the realtime model."""
    return (
        "Speak this startup line now, in character, as 1-2 short sentences. "
        f"Address the user as {USER_DISPLAY_NAME}. Do not ask a question or add extra topics. "
        f"{clock.greeting} It's {clock.spoken_datetime} here in Sydney. "
        "I'm running my startup diagnostics now."
    )


def build_report_instruction(report: StartupReport) -> str:
    """Return the spoken diagnostic result instruction for the realtime model."""
    issues = [
        result
        for result in report.results
        if result.status in {DiagnosticStatus.UNAVAILABLE, DiagnosticStatus.DEGRADED, DiagnosticStatus.UNKNOWN}
    ]
    issue_names = ", ".join(_spoken_name(result.name) for result in issues)
    if report.overall is OverallStatus.ALL_GREEN:
        bits = ["AI stack online", "Reachy hardware responding"]
        if _named_status(report, "speech_to_speech") is DiagnosticStatus.HEALTHY:
            bits.append("speech ready")
        if _named_status(report, "camera") is DiagnosticStatus.HEALTHY:
            bits.append("vision ready")
        if _named_status(report, "tools") is DiagnosticStatus.HEALTHY:
            bits.append("tools standing by")
        if _named_status(report, "home_assistant") is DiagnosticStatus.HEALTHY:
            bits.append("home automation standing by")
        body = "All systems are green. " + ", ".join(bits) + ". Reachy Mini is online and ready to go."
    elif report.overall is OverallStatus.SIGNIFICANT_ISSUE:
        focus = issue_names or "a required system"
        body = f"Startup diagnostics found a problem with {focus}. I'm not fully operational yet."
    else:
        focus = issue_names or "a non-critical system"
        body = (
            f"Startup diagnostics are complete. Most systems are ready, but {focus} "
            f"{'is' if len(issues) == 1 else 'are'} not fully available. "
            "Reachy Mini is still operational."
        )
    return (
        "Speak this startup result now, in character, as 2 short sentences. "
        "Do not list raw logs, invent systems, or claim that motors physically moved. "
        "Do not ask a question. Use these facts and this wording closely:\n"
        f"{body}"
    )


async def run_startup_diagnostic(
    deps: ToolDependencies,
    *,
    http_client: httpx.AsyncClient | None = None,
    speech_session_open: bool = False,
) -> StartupReport:
    """Run AI-stack and Reachy hardware checks without crashing the boot path."""
    clock = read_sydney_clock()
    logger.info("startup diagnostic started")
    logger.info("Sydney date/time: %s (%s)", clock.spoken_datetime, clock.period.value)

    own_client = http_client is None
    client = http_client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S)
    try:
        software = await _check_software_stack(client, speech_session_open=speech_session_open)
        hardware = await _check_reachy_hardware(deps)
    finally:
        if own_client:
            await client.aclose()

    results = tuple(software + hardware)
    overall = evaluate_overall(results)
    for result in results:
        logger.info(
            "startup diagnostic %s: %s%s",
            result.name,
            result.status.value,
            f" ({result.detail})" if result.detail else "",
        )
    logger.info("startup diagnostic overall: %s", overall.value)
    logger.info("startup diagnostic completed")
    return StartupReport(clock=clock, results=results, overall=overall)


async def deliver_boot_sequence(
    deps: ToolDependencies,
    speak: Callable[[str], Awaitable[None]],
) -> bool:
    """Greet Walter, run diagnostics once, then speak the readiness report."""
    global _boot_sequence_delivered
    if _boot_sequence_delivered:
        logger.info("Skipping startup diagnostic; already delivered this boot")
        return True
    _boot_sequence_delivered = True

    clock = read_sydney_clock()
    greeting_sent = await _speak_quietly(speak, build_greeting_instruction(clock))
    try:
        report = await run_startup_diagnostic(deps, speech_session_open=True)
    except Exception as exc:
        logger.warning("Startup diagnostic failed: %s", exc)
        fallback = StartupReport(clock=clock, results=(), overall=OverallStatus.SIGNIFICANT_ISSUE)
        report_sent = await _speak_quietly(speak, build_report_instruction(fallback))
        return greeting_sent or report_sent
    report_sent = await _speak_quietly(speak, build_report_instruction(report))
    return greeting_sent or report_sent


async def _speak_quietly(speak: Callable[[str], Awaitable[None]], text: str) -> bool:
    try:
        await speak(text)
    except Exception as exc:
        logger.warning("Startup speech failed: %s", exc)
        return False
    return True


def _ordinal(day: int) -> str:
    if 10 <= day % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _spoken_name(name: str) -> str:
    return _SPOKEN_NAMES.get(name, name.replace("_", " "))


def _named_status(report: StartupReport, name: str) -> DiagnosticStatus | None:
    for result in report.results:
        if result.name == name:
            return result.status
    return None


def _result(name: str, status: DiagnosticStatus, detail: str = "") -> SubsystemResult:
    return SubsystemResult(name=name, status=status, detail=detail, critical=name in _CRITICAL_NAMES)


async def _isolated(name: str, check: Awaitable[SubsystemResult]) -> SubsystemResult:
    try:
        return await check
    except Exception as exc:
        logger.warning("%s diagnostic failed: %s", name, exc)
        return _result(name, DiagnosticStatus.UNKNOWN, str(exc))


async def _check_software_stack(
    client: httpx.AsyncClient,
    *,
    speech_session_open: bool,
) -> list[SubsystemResult]:
    llama, speech, hermes, home_assistant, apex = await asyncio.gather(
        _isolated("local_llm", _check_llama(client)),
        _isolated("speech_to_speech", _check_speech(client, session_open=speech_session_open)),
        _isolated("hermes", _check_hermes(client)),
        _isolated("home_assistant", _check_home_assistant(client)),
        _isolated("apex", _check_apex(client)),
    )
    bus = await _isolated("bus", _check_bus(client, home_assistant))
    return [
        _result("conversation_app", DiagnosticStatus.HEALTHY, "this process is running"),
        llama,
        speech,
        _bundled_speech_component("stt", speech, "Parakeet STT bundled in speech-to-speech"),
        _bundled_speech_component("tts", speech, "Qwen TTS bundled in speech-to-speech"),
        hermes,
        home_assistant,
        apex,
        bus,
        _check_tools(),
        _check_emotion(),
    ]


def _bundled_speech_component(name: str, speech: SubsystemResult, detail: str) -> SubsystemResult:
    if speech.status is DiagnosticStatus.HEALTHY:
        return _result(name, DiagnosticStatus.HEALTHY, detail)
    if speech.status is DiagnosticStatus.DEGRADED:
        return _result(name, DiagnosticStatus.DEGRADED, speech.detail or detail)
    return _result(name, speech.status, speech.detail or detail)


async def _check_llama(client: httpx.AsyncClient) -> SubsystemResult:
    health_url = f"http://{_LLAMA_HOST}:{_LLAMA_PORT}/health"
    models_url = f"http://{_LLAMA_HOST}:{_LLAMA_PORT}/v1/models"
    health_code, _, health_error = await _http_get(client, health_url)
    models_code, models_payload, models_error = await _http_get(client, models_url)
    if health_code == 200 and models_code == 200:
        model_id = _loaded_model_id(models_payload)
        return _result("local_llm", DiagnosticStatus.HEALTHY, model_id or "llama.cpp API responding")
    if health_code == 503:
        return _result("local_llm", DiagnosticStatus.DEGRADED, "llama.cpp is loading a model")
    if health_code is None and models_code is None:
        return _result(
            "local_llm", DiagnosticStatus.UNAVAILABLE, health_error or models_error or "llama.cpp not reachable"
        )
    return _result("local_llm", DiagnosticStatus.DEGRADED, health_error or models_error or "llama.cpp API not healthy")


async def _check_speech(client: httpx.AsyncClient, *, session_open: bool) -> SubsystemResult:
    ws_url = get_hf_direct_ws_url()
    if not ws_url:
        if session_open:
            return _result("speech_to_speech", DiagnosticStatus.HEALTHY, "realtime session is open")
        return _result("speech_to_speech", DiagnosticStatus.NOT_CONFIGURED, "HF_REALTIME_WS_URL is not set")
    try:
        parsed = parse_hf_realtime_url(ws_url)
    except ValueError as exc:
        if session_open:
            return _result("speech_to_speech", DiagnosticStatus.DEGRADED, str(exc))
        return _result("speech_to_speech", DiagnosticStatus.UNAVAILABLE, str(exc))
    host = parsed.host or "127.0.0.1"
    port = parsed.port or 8765
    models_url = f"http://{host}:{port}/v1/models"
    pool_url = f"http://{host}:{port}/v1/pool"
    models_code, _, models_error = await _http_get(client, models_url)
    pool_code, pool_payload, pool_error = await _http_get(client, pool_url)
    if pool_code == 200 and isinstance(pool_payload, dict):
        units = pool_payload.get("units")
        if isinstance(units, list) and any(isinstance(unit, dict) and unit.get("state") == "stuck" for unit in units):
            return _result("speech_to_speech", DiagnosticStatus.DEGRADED, "realtime session slot is stuck")
    # /v1/pool is the speech health API; /v1/models is optional and 404s on this server.
    if models_code == 200:
        return _result("speech_to_speech", DiagnosticStatus.HEALTHY, "speech-to-speech realtime API responding")
    if pool_code == 200:
        return _result("speech_to_speech", DiagnosticStatus.HEALTHY, "speech-to-speech pool API responding")
    if session_open:
        return _result(
            "speech_to_speech",
            DiagnosticStatus.DEGRADED,
            models_error or pool_error or "session is open but HTTP health failed",
        )
    return _result(
        "speech_to_speech",
        DiagnosticStatus.UNAVAILABLE,
        models_error or pool_error or "speech-to-speech not reachable",
    )


async def _check_hermes(client: httpx.AsyncClient) -> SubsystemResult:
    gateway = (config.HERMES_GATEWAY_URL or "").strip()
    api_key = (config.HERMES_API_KEY or "").strip()
    if not gateway or not api_key:
        return _result("hermes", DiagnosticStatus.NOT_CONFIGURED, "HERMES_GATEWAY_URL or HERMES_API_KEY is unset")
    models_url = gateway.replace("/v1/chat/completions", "/v1/models")
    status_code, _payload, error = await _http_get(
        client,
        models_url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )
    if status_code in {401, 403}:
        return _result("hermes", DiagnosticStatus.DEGRADED, f"HTTP {status_code}")
    if status_code in {200, 404}:
        return _result("hermes", DiagnosticStatus.HEALTHY, "Hermes API reachable")
    if status_code is None:
        return _result("hermes", DiagnosticStatus.UNAVAILABLE, error or "Hermes is currently unavailable")
    return _result("hermes", DiagnosticStatus.DEGRADED, error or f"HTTP {status_code}")


async def _check_home_assistant(client: httpx.AsyncClient) -> SubsystemResult:
    base_url = (config.HA_URL or "").strip().rstrip("/")
    token = (config.HA_TOKEN or "").strip()
    if not base_url or not token:
        return _result("home_assistant", DiagnosticStatus.NOT_CONFIGURED, "HA_URL or HA_TOKEN is unset")
    status_code, _payload, error = await _http_get(
        client,
        f"{base_url}/api/",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    if status_code in {401, 403}:
        return _result("home_assistant", DiagnosticStatus.DEGRADED, f"HTTP {status_code}")
    if status_code == 200:
        return _result("home_assistant", DiagnosticStatus.HEALTHY, "Home Assistant API accepted the token")
    if status_code is None:
        return _result(
            "home_assistant", DiagnosticStatus.UNAVAILABLE, error or "Home Assistant is currently unavailable"
        )
    return _result("home_assistant", DiagnosticStatus.UNAVAILABLE, error or f"HTTP {status_code}")


async def _check_apex(client: httpx.AsyncClient) -> SubsystemResult:
    url = (config.APEX_STATUS_URL or "").strip()
    if not url:
        return _result("apex", DiagnosticStatus.NOT_CONFIGURED, "APEX_STATUS_URL is unset")
    status_code, payload, error = await _http_get(client, url)
    if status_code == 200 and isinstance(payload, dict):
        return _result("apex", DiagnosticStatus.HEALTHY, "Apex /status returned JSON")
    if status_code == 200:
        return _result("apex", DiagnosticStatus.DEGRADED, "Apex responded without JSON")
    if status_code is None:
        return _result("apex", DiagnosticStatus.UNAVAILABLE, error or "Apex status is currently unavailable")
    return _result("apex", DiagnosticStatus.UNAVAILABLE, error or f"HTTP {status_code}")


async def _check_bus(client: httpx.AsyncClient, home_assistant: SubsystemResult) -> SubsystemResult:
    if home_assistant.status is DiagnosticStatus.NOT_CONFIGURED:
        return _result("bus", DiagnosticStatus.NOT_CONFIGURED, "Home Assistant is not configured")
    if home_assistant.status is not DiagnosticStatus.HEALTHY:
        return _result("bus", home_assistant.status, "bus arrivals are read through Home Assistant")
    base_url = (config.HA_URL or "").strip().rstrip("/")
    token = (config.HA_TOKEN or "").strip()
    if not base_url or not token:
        return _result("bus", DiagnosticStatus.NOT_CONFIGURED, "HA_URL or HA_TOKEN is unset")
    entity_id = (config.HA_BUS_ENTITY_ID or "").strip() or "sensor.route_311_at_rockwall_cres"
    status_code, payload, error = await _http_get(
        client,
        f"{base_url}/api/states/{quote(entity_id, safe='')}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    if status_code == 200 and isinstance(payload, dict):
        return _result("bus", DiagnosticStatus.HEALTHY, f"{entity_id} readable")
    if status_code == 404:
        return _result("bus", DiagnosticStatus.DEGRADED, f"{entity_id} was not found")
    return _result("bus", DiagnosticStatus.UNAVAILABLE, error or f"HTTP {status_code}")


def _check_tools() -> SubsystemResult:
    try:
        specs = get_tool_specs()
    except Exception as exc:
        logger.warning("Tool layer diagnostic failed: %s", exc)
        return _result("tools", DiagnosticStatus.UNKNOWN, str(exc))
    if not specs:
        return _result("tools", DiagnosticStatus.DEGRADED, "no tools are registered")
    return _result("tools", DiagnosticStatus.HEALTHY, f"{len(specs)} tools registered")


def _check_emotion() -> SubsystemResult:
    if EMOTION_AVAILABLE:
        return _result("emotion", DiagnosticStatus.HEALTHY, "emotion library importable")
    return _result("emotion", DiagnosticStatus.UNAVAILABLE, "emotion library is not available")


async def _check_reachy_hardware(deps: ToolDependencies) -> list[SubsystemResult]:
    try:
        return await asyncio.to_thread(_inspect_reachy_hardware, deps)
    except Exception as exc:
        logger.warning("Reachy hardware diagnostic failed: %s", exc)
        return [
            _result("reachy_connection", DiagnosticStatus.UNKNOWN, str(exc)),
            _result("sdk_communication", DiagnosticStatus.UNKNOWN, str(exc)),
            _result("actuators", DiagnosticStatus.UNKNOWN, str(exc)),
        ]


def _inspect_reachy_hardware(deps: ToolDependencies) -> list[SubsystemResult]:
    robot = deps.reachy_mini
    connection = _check_reachy_connection(robot)
    sdk = _check_sdk_and_actuators(robot, connection)
    hardware = _check_head_and_antennas(robot)
    if connection.status is DiagnosticStatus.HEALTHY and sdk.status is not DiagnosticStatus.HEALTHY:
        connection = _result("reachy_connection", sdk.status, sdk.detail)
    elif connection.status is DiagnosticStatus.HEALTHY:
        failing = next(
            (
                item
                for item in hardware
                if item.name in {"actuators", "head", "antennas"} and item.status is not DiagnosticStatus.HEALTHY
            ),
            None,
        )
        if failing is not None:
            connection = _result("reachy_connection", failing.status, failing.detail)
    return [connection, sdk, *hardware, _check_camera(deps), *_check_audio(robot)]


def _check_reachy_connection(robot: ReachyMini) -> SubsystemResult:
    try:
        client = robot.client
        status = client.get_status(wait=False)
    except Exception as exc:
        logger.warning("Reachy connection diagnostic failed: %s", exc)
        return _result("reachy_connection", DiagnosticStatus.UNAVAILABLE, str(exc))
    state = getattr(status, "state", None)
    error = getattr(status, "error", None)
    simulation = getattr(status, "simulation_enabled", None)
    host = getattr(client, "host", None)
    port = getattr(client, "port", None)
    if error:
        return _result("reachy_connection", DiagnosticStatus.DEGRADED, str(error))
    state_value = getattr(state, "value", state)
    normalized = str(state_value).lower() if state_value is not None else "running"
    environment = "simulator" if simulation else "daemon"
    location = f"{environment} at {host}:{port}"
    if normalized in {"stopped", "stopping"}:
        return _result("reachy_connection", DiagnosticStatus.DEGRADED, f"daemon state={state_value}")
    if normalized not in {"running"}:
        # DaemonState.ERROR is sticky: start() can set it when the backend misses a
        # 2s ready timeout, then status() clears error but never restores RUNNING.
        logger.info(
            "Reachy daemon state=%s with no error field; judging connection from SDK communication",
            state_value,
        )
    return _result("reachy_connection", DiagnosticStatus.HEALTHY, location)


def _check_sdk_and_actuators(robot: ReachyMini, connection: SubsystemResult) -> SubsystemResult:
    if connection.status is DiagnosticStatus.UNAVAILABLE:
        return _result("sdk_communication", DiagnosticStatus.UNAVAILABLE, "Reachy connection is unavailable")
    try:
        pose = np.asarray(robot.get_current_head_pose(), dtype=np.float64)
        head_joints, antennas = robot.get_current_joint_positions()
    except Exception as exc:
        logger.warning("Reachy SDK communication diagnostic failed: %s", exc)
        return _result("sdk_communication", DiagnosticStatus.UNAVAILABLE, str(exc))
    if pose.shape != (4, 4):
        return _result("sdk_communication", DiagnosticStatus.DEGRADED, f"unexpected head pose shape {pose.shape}")
    if not _is_joint_list(head_joints, 7) or not _is_joint_list(antennas, 2):
        return _result("sdk_communication", DiagnosticStatus.DEGRADED, "unexpected joint vector length")
    backend_error = _backend_error(robot)
    if backend_error:
        return _result("sdk_communication", DiagnosticStatus.DEGRADED, backend_error)
    return _result("sdk_communication", DiagnosticStatus.HEALTHY, "head pose and joint state readable")


def _check_head_and_antennas(robot: ReachyMini) -> list[SubsystemResult]:
    try:
        head_joints, antennas = robot.get_current_joint_positions()
        pose = np.asarray(robot.get_current_head_pose(), dtype=np.float64)
    except Exception as exc:
        logger.warning("Reachy actuator diagnostic failed: %s", exc)
        return [
            _result("actuators", DiagnosticStatus.UNAVAILABLE, str(exc)),
            _result("head", DiagnosticStatus.UNAVAILABLE, str(exc)),
            _result("antennas", DiagnosticStatus.UNAVAILABLE, str(exc)),
        ]
    backend_error = _backend_error(robot)
    head_ok = pose.shape == (4, 4) and _is_joint_list(head_joints, 7)
    antenna_ok = _is_joint_list(antennas, 2)
    if backend_error:
        actuator_status = DiagnosticStatus.DEGRADED
        actuator_detail = backend_error
    elif head_ok and antenna_ok:
        actuator_status = DiagnosticStatus.HEALTHY
        actuator_detail = "head and antenna joint communication healthy"
    else:
        actuator_status = DiagnosticStatus.DEGRADED
        actuator_detail = "partial joint state"
    return [
        _result("actuators", actuator_status, actuator_detail),
        _result(
            "head",
            DiagnosticStatus.HEALTHY if head_ok else DiagnosticStatus.DEGRADED,
            "head pose and 7 joints readable",
        ),
        _result(
            "antennas",
            DiagnosticStatus.HEALTHY if antenna_ok else DiagnosticStatus.DEGRADED,
            "2 antenna joints readable",
        ),
    ]


def _check_camera(deps: ToolDependencies) -> SubsystemResult:
    if not deps.camera_enabled:
        return _result("camera", DiagnosticStatus.NOT_CONFIGURED, "camera disabled with --no-camera")
    try:
        jpeg_bytes = deps.reachy_mini.media.get_frame_jpeg()
    except Exception as exc:
        logger.warning("Camera diagnostic failed: %s", exc)
        return _result("camera", DiagnosticStatus.UNAVAILABLE, str(exc))
    if jpeg_bytes is None:
        return _result("camera", DiagnosticStatus.UNAVAILABLE, "no camera frame available")
    return _result("camera", DiagnosticStatus.HEALTHY, "camera frame available")


def _check_audio(robot: ReachyMini) -> list[SubsystemResult]:
    return [_check_microphone(robot), _check_speaker(robot)]


def _check_microphone(robot: ReachyMini) -> SubsystemResult:
    try:
        rate = robot.media.get_input_audio_samplerate()
    except Exception as exc:
        logger.warning("Microphone diagnostic failed: %s", exc)
        return _result("microphone", DiagnosticStatus.UNAVAILABLE, str(exc))
    try:
        numeric_rate = float(rate)
    except (TypeError, ValueError):
        return _result("microphone", DiagnosticStatus.UNKNOWN, "input sample rate was not numeric")
    if numeric_rate <= 0:
        return _result("microphone", DiagnosticStatus.DEGRADED, f"input sample rate {numeric_rate}")
    return _result("microphone", DiagnosticStatus.HEALTHY, f"input sample rate {int(numeric_rate)}")


def _check_speaker(robot: ReachyMini) -> SubsystemResult:
    audio = getattr(robot.media, "audio", None)
    if audio is None:
        return _result("speaker", DiagnosticStatus.UNAVAILABLE, "media audio output is unavailable")
    return _result("speaker", DiagnosticStatus.HEALTHY, "speaker output interface available")


def _backend_error(robot: ReachyMini) -> str | None:
    try:
        status = robot.client.get_status(wait=False)
    except Exception:
        return None
    daemon_error = getattr(status, "error", None)
    if daemon_error:
        return str(daemon_error)
    backend = getattr(status, "backend_status", None)
    backend_error = getattr(backend, "error", None) if backend is not None else None
    if backend_error:
        return str(backend_error)
    return None


def _is_joint_list(values: object, expected: int) -> bool:
    if not isinstance(values, (list, tuple)):
        return False
    if len(values) != expected:
        return False
    try:
        return all(np.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _loaded_model_id(payload: object) -> str | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return None
    first = data[0]
    if not isinstance(first, dict):
        return None
    model_id = first.get("id")
    return model_id.strip() if isinstance(model_id, str) and model_id.strip() else None


async def _http_get(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int | None, object | None, str | None]:
    try:
        response = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        logger.warning("Startup diagnostic timed out for GET %s", url)
        return None, None, "timed out"
    except httpx.RequestError as exc:
        logger.warning("Startup diagnostic request failed for GET %s: %s", url, exc)
        return None, None, str(exc)
    payload: object | None = None
    try:
        if response.content:
            payload = response.json()
    except ValueError:
        payload = None
    host = urlsplit(url).hostname
    logger.debug("startup diagnostic HTTP %s %s host=%s", response.status_code, url, host)
    return response.status_code, payload, None
