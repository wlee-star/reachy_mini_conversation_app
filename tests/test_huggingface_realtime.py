import json
import time
import base64
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.streaming import AdditionalOutputs
from reachy_mini_conversation_app.tools.apex import classify_reef_intent
from reachy_mini_conversation_app.hermes_client import HermesTimeoutError
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import (
    _HERMES_SPEECH_FALLBACK,
    RealtimeSessionSlotsBusy,
    HuggingFaceRealtimeHandler,
    _hermes_result_text,
)
from reachy_mini_conversation_app.tools.home_assistant import HomeAssistant
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


def test_hermes_result_text_prefers_report_over_errors() -> None:
    """A structured reef history report is spoken instead of a vague retry line."""
    assert _hermes_result_text({"report": "Reef stable - temp 24.0C.", "error": "no"}) == "Reef stable - temp 24.0C."
    assert (
        _hermes_result_text({"status": "error", "trend_available": False})
        == "Historical reef data is currently unavailable."
    )
    cached = "Reef stable - temp 24.0C (-0.012/6h)."
    spoken = _hermes_result_text(
        {
            "status": "degraded",
            "stale": True,
            "source": "cache",
            "report": cached,
            "spoken": f"I couldn't reach the live Reef data, but I have a cached Hermes report from approximately 3 minutes ago. {cached}",
        }
    )
    assert cached in spoken
    assert "cached" in spoken.lower()


@pytest.fixture(autouse=True)
def _skip_boot_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing realtime tests must not run live startup diagnostics."""

    async def _fake_boot(_deps: object, _speak: object) -> bool:
        return True

    monkeypatch.setattr(hf_mod, "deliver_boot_sequence", _fake_boot)
    monkeypatch.setattr(hf_mod, "boot_sequence_already_delivered", lambda: False)


class _FakeEvent:
    """A minimal realtime event: a `type` plus arbitrary attributes."""

    def __init__(self, event_type: str, **fields: Any) -> None:
        """Store the event type and any extra attributes."""
        self.type = event_type
        self.__dict__.update(fields)


def _make_fake_realtime_client(
    *,
    events: tuple[_FakeEvent, ...] = (),
    captured_update: dict[str, Any] | None = None,
    captured_connect: dict[str, Any] | None = None,
    captured_cancels: list[int] | None = None,
    update_errors: list[BaseException] | None = None,
) -> Any:
    """Build a fake AsyncOpenAI-shaped client whose realtime session yields `events`.

    When given, `captured_update`/`captured_connect` record the kwargs passed to
    `session.update(...)` / `realtime.connect(...)`. `update_errors` are raised
    from `session.update` in order before a successful update.
    """

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            if update_errors:
                raise update_errors.pop(0)
            if captured_update is not None:
                captured_update.update(kwargs)

    class FakeNoop:
        async def append(self, **_kw: Any) -> None:
            pass

        async def create(self, **_kw: Any) -> None:
            pass

        async def cancel(self, **_kw: Any) -> None:
            if captured_cancels is not None:
                captured_cancels.append(1)

    class FakeConversation:
        item = FakeNoop()

    class FakeConn:
        session = FakeSession()
        input_audio_buffer = FakeNoop()
        conversation = FakeConversation()
        response = FakeNoop()

        def __init__(self) -> None:
            self._events = iter(events)

        async def __aenter__(self) -> "FakeConn":
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def close(self) -> None:
            pass

        def __aiter__(self) -> "FakeConn":
            return self

        async def __anext__(self) -> _FakeEvent:
            try:
                return next(self._events)
            except StopIteration:
                raise StopAsyncIteration

    class FakeRealtime:
        def connect(self, **kwargs: Any) -> FakeConn:
            if captured_connect is not None:
                captured_connect.update(kwargs)
            return FakeConn()

    class FakeClient:
        realtime = FakeRealtime()

    return FakeClient()


def _fake_openai_client(captured_kwargs: dict[str, Any]) -> type:
    """Return a fake AsyncOpenAI class that records its constructor kwargs."""

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured_kwargs.update(kwargs)

    return FakeClient


def _fake_allocator(
    connect_url: str,
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]],
) -> type:
    """Return a fake httpx.AsyncClient that records allocator requests."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"session_id": "session-123", "connect_url": connect_url}

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: dict[str, str] | None = None,
        ) -> FakeResponse:
            posts.append((url, headers, json))
            return FakeResponse()

    return FakeAsyncClient


def _fake_pool_client(payload: dict[str, Any]) -> type:
    """Return a fake httpx.AsyncClient that serves speech-to-speech `/v1/pool`."""

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return payload

    class FakeAsyncClient:
        def __init__(self, **_kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_a: Any) -> bool:
            return False

        async def get(self, url: str) -> FakeResponse:
            assert url.endswith("/v1/pool")
            return FakeResponse()

    return FakeAsyncClient


def _drain_handler_outputs(handler: HuggingFaceRealtimeHandler) -> list[object]:
    """Return every item currently sitting on the handler output queue."""
    items: list[object] = []
    while not handler.output_queue.empty():
        items.append(handler.output_queue.get_nowait())
    return items


def _assistant_transcripts(items: list[object]) -> list[str]:
    """Return assistant transcript strings from queued AdditionalOutputs."""
    transcripts: list[str] = []
    for item in items:
        if not isinstance(item, AdditionalOutputs):
            continue
        for message in item.args:
            content = message.get("content")
            if message.get("role") == "assistant" and isinstance(content, str):
                transcripts.append(content)
    return transcripts


def _audio_frame_count(items: list[object]) -> int:
    """Return how many PCM frames were queued for playback."""
    return sum(1 for item in items if isinstance(item, tuple))


def _silent_pcm_delta() -> str:
    """Return a tiny silent PCM frame encoded like a realtime audio delta."""
    return base64.b64encode(b"\x00\x00" * 16).decode("ascii")


@pytest.mark.asyncio
async def test_partial_transcription_uses_latest_snapshot(monkeypatch: Any) -> None:
    """Partial transcription snapshots should replace older snapshots for the same item."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey"),
            _FakeEvent(
                "conversation.item.input_audio_transcription.delta", item_id="item-1", delta="Hey, how are you?"
            ),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler.input_transcript_chunks_by_item.item_id == "item-1"
    assert handler.input_transcript_chunks_by_item.deltas == ["Hey, how are you?"]


@pytest.mark.asyncio
async def test_emit_skips_idle_signal_while_response_active(monkeypatch: Any) -> None:
    """Idle tools should not trigger while a response is still active."""
    movement_manager = MagicMock()
    movement_manager.is_idle.return_value = True
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager)
    handler = HuggingFaceRealtimeHandler(deps)
    handler.last_activity_time = time.monotonic() - (handler.IDLE_BEHAVIOR_THRESHOLD_S + 10.0)
    handler._response_done_event.clear()

    send_idle_signal = AsyncMock()
    monkeypatch.setattr(handler, "send_idle_signal", send_idle_signal)
    monkeypatch.setattr(conv_mod, "wait_for_item", AsyncMock(return_value=None))

    result = await handler.emit()

    assert result is None
    send_idle_signal.assert_not_awaited()


@pytest.mark.asyncio
async def test_tool_choice_is_restored_after_spoken_followup(monkeypatch: Any) -> None:
    """Tool suppression must end when the tool-result response completes."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_session_greeting_prompt", lambda: "")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    captured_update: dict[str, Any] = {}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._tool_followup_tools_disabled = True
    handler._active_response_reason = "tool_result:apex"
    handler.client = _make_fake_realtime_client(
        events=(_FakeEvent("response.done"),),
        captured_update=captured_update,
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert handler._tool_followup_tools_disabled is False
    assert captured_update["session"] == {"type": "realtime", "tool_choice": "auto"}


@pytest.mark.asyncio
async def test_parallel_tool_calls_trigger_single_response(monkeypatch: Any) -> None:
    """Parallel tool calls in one turn should yield one response, not one per completed tool."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._in_flight_tool_calls = {"call_a", "call_b"}

    def _completed(call_id: str) -> ToolNotification:
        return ToolNotification(
            id=call_id,
            tool_name="test__parallel_probe",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"ok": True},
        )

    await handler._handle_tool_result(_completed("call_a"))
    assert create.await_count == 0

    await handler._handle_tool_result(_completed("call_b"))
    assert create.await_count == 1
    create.assert_awaited_once_with(
        reason="tool_result:test__parallel_probe",
        response={"tool_choice": "none"},
    )
    handler.connection.session.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_short_tool_result_does_not_trigger_response(monkeypatch: Any) -> None:
    """A short tool that finishes after a newer user turn must not speak into that turn."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    handler._turn_generation = 2
    handler._tool_call_generation = {"call_old": 1}
    handler._in_flight_tool_calls = {"call_old"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_old",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"reply": "stale"},
        )
    )

    handler.connection.conversation.item.create.assert_not_awaited()
    assert create.await_count == 0
    assert "call_old" not in handler._in_flight_tool_calls


@pytest.mark.asyncio
async def test_home_assistant_control_speaks_and_plays_success(monkeypatch: Any) -> None:
    """Successful local HA control plays success once, then starts a spoken follow-up."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    play = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)
    handler._in_flight_tool_calls = {"call_1"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "success", "service": "switch.turn_off", "entity_id": "switch.lamp_3"},
        )
    )

    created_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert created_item["type"] == "function_call_output"
    assert created_item["output"] == json.dumps(
        {"status": "success", "service": "switch.turn_off", "entity_id": "switch.lamp_3"}
    )
    play.assert_awaited_once()
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})


@pytest.mark.asyncio
async def test_home_assistant_control_failure_does_not_play_success(monkeypatch: Any) -> None:
    """A failed device action must not play success or claim the device changed."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    play = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)
    handler._in_flight_tool_calls = {"call_1"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"error": "Home Assistant could not find that entity or service."},
        )
    )

    play.assert_not_awaited()
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})


@pytest.mark.asyncio
async def test_play_device_success_emotion_queues_success_once(monkeypatch: Any) -> None:
    """The existing play_emotion tool is used, and a second call in the same turn is ignored."""
    play_calls: list[dict[str, Any]] = []

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append(kwargs)
            return {"status": "queued", "emotion": "success1"}

    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    await handler._play_device_success_emotion()
    await handler._play_device_success_emotion()

    assert play_calls == [{"emotion": "success", "allow_random": False}]


@pytest.mark.asyncio
async def test_play_bus_helpful1_uses_existing_play_emotion(monkeypatch: Any) -> None:
    """The 311 10-minute callback reuses play_emotion and the session Reachy connection."""
    play_calls: list[tuple[ToolDependencies, dict[str, Any]]] = []
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    class FakePlayEmotion:
        async def __call__(self, tool_deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append((tool_deps, kwargs))
            return {"status": "queued", "emotion": "helpful1"}

    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)
    handler = HuggingFaceRealtimeHandler(deps)

    await handler._play_bus_helpful1()

    assert play_calls == [(deps, {"emotion": "helpful1", "allow_random": False})]


@pytest.mark.asyncio
async def test_success_emotion_tool_skipped_when_device_control_owns_turn(monkeypatch: Any) -> None:
    """LLM play_emotion(success) must not add a second success animation after device control."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"play_emotion": hf_mod.PlayEmotion()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    handler._turn_generation = 1
    handler._in_flight_tool_calls = {"call_emotion"}
    handler._turn_device_control_call_ids = {"call_lamp"}

    await handler._start_or_skip_success_emotion_tool("call_emotion", '{"emotion": "success"}', 1)

    start_tool.assert_not_awaited()
    created_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert created_item["output"] == json.dumps({"status": "skipped", "emotion": "success"})


@pytest.mark.asyncio
async def test_start_fast_dance_emotion_plays_dance1_once(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """A spoken dance request queues official dance1 once and leaves conversation running."""
    play_calls: list[dict[str, Any]] = []

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append(kwargs)
            return {"status": "queued", "emotion": "dance1"}

    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    with caplog.at_level("INFO"):
        handler._start_fast_dance_emotion("Reachy, can you dance?")
        handler._start_fast_dance_emotion("Reachy, can you dance?")
        assert handler._fast_dance_task is not None
        await handler._fast_dance_task

    assert play_calls == [{"emotion": "dance1", "allow_random": False}]
    assert handler._dance_emotion_played is True
    assert '[EMOTION] Dance intent detected: "Reachy, can you dance?"' in caplog.text
    assert "[EMOTION] Playing recorded emotion: dance1" in caplog.text


@pytest.mark.asyncio
async def test_start_fast_dance_emotion_uses_dance3_when_energetic(monkeypatch: Any) -> None:
    """Energetic dance requests may queue official dance3."""
    play_calls: list[dict[str, Any]] = []

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append(kwargs)
            return {"status": "queued", "emotion": "dance3"}

    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._start_fast_dance_emotion("Give me an energetic dance")
    assert handler._fast_dance_task is not None
    await handler._fast_dance_task

    assert play_calls == [{"emotion": "dance3", "allow_random": False}]


@pytest.mark.asyncio
async def test_start_fast_dance_emotion_ignores_unrelated_speech(monkeypatch: Any) -> None:
    """Talking about dance without asking Reachy to perform must not queue motion."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    play = AsyncMock()
    monkeypatch.setattr(handler, "_queue_dance_emotion", play)

    handler._start_fast_dance_emotion("I watched a dance video.")

    assert handler._fast_dance_task is None
    play.assert_not_called()


@pytest.mark.asyncio
async def test_play_dance_emotion_queues_once(monkeypatch: Any) -> None:
    """One user utterance must produce only one official dance trigger."""
    play_calls: list[dict[str, Any]] = []

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append(kwargs)
            return {"status": "queued", "emotion": "dance1"}

    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    await handler._play_dance_emotion("dance1")
    await handler._play_dance_emotion("dance1")

    assert play_calls == [{"emotion": "dance1", "allow_random": False}]


@pytest.mark.asyncio
async def test_duplicate_dance_tool_skipped_when_emotion_already_queued(monkeypatch: Any) -> None:
    """LLM dance or play_emotion(dance) must not add a second dance animation."""

    class FakeDanceTool:
        def wants_spoken_followup(self, result: dict[str, Any] | None, error: str | None) -> bool:
            return False

    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"dance": FakeDanceTool()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    handler._turn_generation = 1
    handler._in_flight_tool_calls = {"call_dance"}
    handler._dance_emotion_played = True

    await handler._skip_duplicate_dance_tool("call_dance", "dance", 1)

    start_tool.assert_not_awaited()
    created_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert created_item["output"] == json.dumps({"status": "skipped", "emotion": "dance1"})


@pytest.mark.asyncio
async def test_home_assistant_query_still_speaks(monkeypatch: Any) -> None:
    """HA reads still need a spoken follow-up so the user hears the answer."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", AsyncMock())
    handler._in_flight_tool_calls = {"call_1"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"minutes": 4, "route": "311"},
        )
    )

    handler._play_device_success_emotion.assert_not_awaited()
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})


@pytest.mark.asyncio
async def test_start_fast_ha_command_runs_lamp_toggle(monkeypatch: Any) -> None:
    """Lamp phrases should fire Home Assistant immediately, without waiting for the LLM."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_ha_commands", run)

    handler._start_fast_ha_command("Turn off lamb three.")
    assert handler._fast_ha_task is not None
    await handler._fast_ha_task

    run.assert_awaited_once_with([{"action": "turn_switch_off", "entity_id": "switch.lamp_3"}])


@pytest.mark.asyncio
async def test_start_fast_ha_command_runs_on_then_off(monkeypatch: Any) -> None:
    """A compound lamp phrase should run every matched action in order."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_ha_commands", run)

    handler._start_fast_ha_command("Rishi, turn on lamp three and turn off lamp three.")
    assert handler._fast_ha_task is not None
    await handler._fast_ha_task

    run.assert_awaited_once_with(
        [
            {"action": "turn_switch_on", "entity_id": "switch.lamp_3"},
            {"action": "turn_switch_off", "entity_id": "switch.lamp_3"},
        ]
    )


@pytest.mark.asyncio
async def test_start_fast_ha_command_ignores_unrelated_speech(monkeypatch: Any) -> None:
    """Non-HA speech must not trigger a local service call."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_ha_commands", run)

    handler._start_fast_ha_command("what's the weather")

    assert handler._fast_ha_task is None
    run.assert_not_called()


@pytest.mark.asyncio
async def test_start_fast_bus_command_queries_live_arrival(monkeypatch: Any) -> None:
    """A 311 request should query Home Assistant immediately, without waiting for the LLM."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_bus_command", run)

    handler._start_fast_bus_command("Let me know when the next 311 is coming.")
    assert handler._fast_bus_task is not None
    await handler._fast_bus_task

    run.assert_awaited_once_with("query", 15)


@pytest.mark.asyncio
async def test_start_fast_bus_command_switches_when_watch_is_active(monkeypatch: Any) -> None:
    """An explicit 'next one instead' request uses the local switch path."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_bus_command", run)
    monkeypatch.setattr(hf_mod.get_bus_monitor(), "pending_offer", lambda: False)
    monkeypatch.setattr(hf_mod.get_bus_monitor(), "monitor_active", lambda: True)

    handler._start_fast_bus_command("Monitor the next one instead.")
    assert handler._fast_bus_task is not None
    await handler._fast_bus_task

    run.assert_awaited_once_with("switch", 15)


@pytest.mark.asyncio
async def test_start_fast_bus_command_ignores_unrelated_speech(monkeypatch: Any) -> None:
    """Non-bus speech must not start the bus monitor fast path."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_bus_command", run)

    handler._start_fast_bus_command("what's the weather")

    run.assert_not_awaited()
    assert handler._fast_bus_task is None


@pytest.mark.asyncio
async def test_start_fast_apex_command_queries_live_status(monkeypatch: Any) -> None:
    """A live reef request should call Apex immediately, without waiting for the LLM."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", run)

    handler._start_fast_apex_command("What is my LLSATO value?")
    assert handler._fast_apex_task is not None
    assert handler._suppress_unsolicited_response_turn == handler._turn_generation
    assert handler._reef_router_owns_turn == handler._turn_generation
    await handler._fast_apex_task

    run.assert_awaited_once_with("llsato")


@pytest.mark.asyncio
async def test_start_fast_apex_command_ignores_unrelated_speech(monkeypatch: Any) -> None:
    """Non-reef speech must not start the Apex fast path."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", run)

    handler._start_fast_apex_command("what's the weather")

    run.assert_not_awaited()
    assert handler._fast_apex_task is None
    assert handler._reef_router_owns_turn is None


@pytest.mark.asyncio
async def test_start_fast_apex_command_routes_report_to_hermes(monkeypatch: Any) -> None:
    """Report/trend questions must not use the local Apex status fast-path."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    apex_run = AsyncMock()
    hermes_run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", apex_run)
    monkeypatch.setattr(handler, "_run_fast_hermes_reef_command", hermes_run)

    handler._start_fast_apex_command("Can you give me a reef tank report?")
    assert handler._fast_apex_task is None
    assert handler._fast_hermes_reef_task is not None
    assert handler._suppress_unsolicited_response_turn == handler._turn_generation
    assert handler._reef_router_owns_turn == handler._turn_generation
    await handler._fast_hermes_reef_task

    apex_run.assert_not_awaited()
    hermes_run.assert_awaited_once()
    assert hermes_run.await_args.args[1] == "detailed_report"
    assert hermes_run.await_args.args[2] is False


@pytest.mark.asyncio
async def test_start_fast_apex_command_routes_trends_to_hermes(monkeypatch: Any) -> None:
    """Trend questions are handed to Hermes, not current Apex readings."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    apex_run = AsyncMock()
    hermes_run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", apex_run)
    monkeypatch.setattr(handler, "_run_fast_hermes_reef_command", hermes_run)

    handler._start_fast_apex_command("Can you give me a reef trending report?")
    assert handler._fast_hermes_reef_task is not None
    assert handler._reef_router_owns_turn == handler._turn_generation
    await handler._fast_hermes_reef_task

    apex_run.assert_not_awaited()
    hermes_run.assert_awaited_once()
    assert hermes_run.await_args.args[1] == "trends"
    assert hermes_run.await_args.args[2] is False


@pytest.mark.asyncio
async def test_hermes_reef_fast_path_speaks_live_report(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Cache presence must not skip Hermes; the live report is spoken."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    live_report = "Live Hermes Reef report: temp 24.1C, nitrate falling."
    send = AsyncMock(return_value=live_report)
    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", send)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: {
            "report": "Reef stable - temp 24.0C (-0.012/6h).",
            "source": "hermes",
            "trends": {
                "Tmp": {"trend_6h": -0.0118, "trend_str": "-0.012/6h"},
                "pH": {"trend_6h": 0.0118, "trend_str": "+0.012/6h"},
                "ORP": {"trend_6h": -0.4714, "trend_str": "-0.471/6h"},
                "FS100": {"trend_6h": 0.4716, "trend_str": "+0.471/6h"},
                "LLSATO": {"trend_6h": -0.0002, "trend_str": "-0.000/6h"},
            },
            "handoff": {"for_reachy": {"ask_first": True, "source": "hermes"}},
        },
    )

    with caplog.at_level("INFO"):
        await handler._run_fast_hermes_reef_command("Can you give me a reef trending report?", "trends")

    send.assert_awaited()
    created = handler.connection.conversation.item.create
    created.assert_awaited_once()
    spoken_prompt = created.await_args.kwargs["item"]["content"][0]["text"]
    assert live_report in spoken_prompt
    create.assert_awaited_once_with(reason="say", response={"tool_choice": "none"})
    assert handler._hermes_spoke_turn == 1
    assert handler._suppress_unsolicited_response_turn == 1
    assert handler._last_reef_response_source == "hermes"
    assert handler._last_reef_response_type == "trends"
    assert handler._last_reef_response_route == "ask_hermes"
    assert "[REEF_ROUTER] intent=trends" in caplog.text
    assert "[REEF_ROUTER] route=ask_hermes" in caplog.text
    assert "[REEF_ROUTER] source=live" in caplog.text
    assert "[REEF_ROUTER] source_type=live_trends" in caplog.text
    assert "[REEF_ROUTER] cache_used=false" in caplog.text
    assert "[REEF_ROUTER] response_owner=ask_hermes" in caplog.text
    assert "[REEF_ROUTER] normal_llm_bypass=true" in caplog.text
    assert "[REEF_ROUTER] explicit_hermes_request=false" in caplog.text
    assert "trend_keys=" in caplog.text
    assert "FS100" in caplog.text
    assert "LLSATO" in caplog.text
    assert "skipping competing ask_hermes" not in caplog.text


@pytest.mark.asyncio
async def test_explicit_hermes_report_calls_live_hermes(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The exact spoken workflow hard-routes to ask_hermes and still calls Hermes."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    live_report = "Live Hermes Reef Tank report: temp 24.1C."
    send = AsyncMock(return_value=live_report)
    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", send)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: {"report": "Reef stable - temp 24.0C (-0.012/6h).", "source": "hermes", "trends": {}},
    )

    with caplog.at_level("INFO"):
        await handler._run_fast_hermes_reef_command(
            "Reachy, ask Hermes what my Reef Tank report is.",
            "detailed_report",
            True,
        )

    send.assert_awaited()
    spoken_prompt = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert live_report in spoken_prompt
    assert handler._last_reef_response_source == "hermes"
    assert handler._last_reef_response_type == "detailed_report"
    assert handler._last_reef_response_route == "ask_hermes"
    assert "[REEF_ROUTER] intent=detailed_report" in caplog.text
    assert "[REEF_ROUTER] route=ask_hermes" in caplog.text
    assert "[REEF_ROUTER] explicit_hermes_request=true" in caplog.text
    assert "[REEF_ROUTER] source=live" in caplog.text
    assert "[REEF_ROUTER] source_type=live_report" in caplog.text
    assert "[REEF_ROUTER] cache_used=false" in caplog.text
    assert "skipping competing ask_hermes" not in caplog.text


@pytest.mark.asyncio
async def test_hermes_reef_fast_path_speaks_stale_cache_on_timeout(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """If live Hermes times out, the cached Reef report is still spoken as stale."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        raise HermesTimeoutError("timed out")

    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", _send)
    cached = "Reef stable - temp 24.0C (-0.012/6h)."
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: {
            "report": cached,
            "source": "hermes",
            "cache_age_seconds": 180.0,
            "trends": {"Tmp": {"trend_6h": -0.012, "trend_str": "-0.012/6h"}},
        },
    )

    with caplog.at_level("INFO"):
        await handler._run_fast_hermes_reef_command("What did Hermes report about my Reef?", "detailed_report")

    spoken_prompt = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert cached in spoken_prompt
    assert "cached" in spoken_prompt.lower()
    assert "[REEF_ROUTER] cache_used=true" in caplog.text
    assert "[REEF_ROUTER] source=cache" in caplog.text
    assert "[REEF_ROUTER] source_type=cached_report" in caplog.text


TREND_QUERY = "Can you tell me what my reef tank is trending at?"
_TREND_CACHE = {
    "report": "Reef trends: Tmp -0.012/6h, pH +0.012/6h, ORP -0.471/6h, FS100 +0.472/6h, LLSATO -0.071/6h.",
    "source": "hermes",
    "cache_age_seconds": 1018.0,
    "trends": {
        "FS100": {"trend_6h": 0.4716, "trend_str": "+0.472/6h"},
        "LLSATO": {"trend_6h": -0.071, "trend_str": "-0.071/6h"},
        "ORP": {"trend_6h": -0.4714, "trend_str": "-0.471/6h"},
        "Tmp": {"trend_6h": -0.012, "trend_str": "-0.012/6h"},
        "pH": {"trend_6h": 0.0118, "trend_str": "+0.012/6h"},
    },
}


@pytest.mark.asyncio
async def test_reef_trend_query_routes_and_speaks_cache_when_hermes_pending(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The exact spoken trend question must reach Reachy with the cached Hermes report."""
    route = classify_reef_intent(TREND_QUERY)
    assert route is not None
    assert route.intent == "trends"
    assert route.route == "ask_hermes"

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    send = AsyncMock(side_effect=AssertionError("pending request must not start a second Hermes call"))
    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", send)
    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.hermes_is_busy", lambda: True)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: dict(_TREND_CACHE),
    )

    with caplog.at_level("INFO"):
        await handler._run_fast_hermes_reef_command(TREND_QUERY, "trends")

    send.assert_not_awaited()
    spoken_prompt = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert _TREND_CACHE["report"] in spoken_prompt
    assert "FS100" in spoken_prompt
    assert "LLSATO" in spoken_prompt
    assert "ORP" in spoken_prompt
    assert handler._hermes_spoke_turn == 1
    assert handler._last_reef_response_source == "hermes"
    assert handler._last_reef_response_type == "trends"
    assert "[REEF_ROUTER] intent=trends" in caplog.text
    assert "[REEF_ROUTER] route=ask_hermes" in caplog.text
    assert "[REEF_ROUTER] source=cache" in caplog.text
    assert "[REEF_ROUTER] source_type=cached_trends" in caplog.text
    assert "[REEF_ROUTER] cache_used=true" in caplog.text
    assert "[REEF_ROUTER] response_owner=ask_hermes" in caplog.text
    assert "trend_keys=" in caplog.text
    assert "FS100" in caplog.text
    assert "LLSATO" in caplog.text
    assert "ORP" in caplog.text
    assert "Tmp" in caplog.text
    assert "pH" in caplog.text
    assert "returning cached Reef report rather than empty result" in caplog.text


@pytest.mark.asyncio
async def test_start_fast_apex_command_routes_exact_trend_question(monkeypatch: Any) -> None:
    """The user's exact trend question is owned by ask_hermes, not live Apex."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    apex_run = AsyncMock()
    hermes_run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", apex_run)
    monkeypatch.setattr(handler, "_run_fast_hermes_reef_command", hermes_run)

    handler._start_fast_apex_command(TREND_QUERY)
    assert handler._fast_hermes_reef_task is not None
    await handler._fast_hermes_reef_task

    apex_run.assert_not_awaited()
    hermes_run.assert_awaited_once()
    assert hermes_run.await_args.args[0] == TREND_QUERY
    assert hermes_run.await_args.args[1] == "trends"


@pytest.mark.asyncio
async def test_late_live_hermes_result_is_not_spoken_after_cache_fallback(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """If cache was already spoken while Hermes was pending, a late live result must not also speak."""
    hermes_client._HERMES_REQUEST_LOCK = asyncio.Lock()
    hermes_client._HERMES_IN_FLIGHT_REQUEST_ID = None
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    started = asyncio.Event()
    release = asyncio.Event()
    live_report = "LATE LIVE REPORT that must not be spoken"

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        async with hermes_client._HERMES_REQUEST_LOCK:
            started.set()
            await release.wait()
        return live_report

    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", _send)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: dict(_TREND_CACHE),
    )

    with caplog.at_level("INFO"):
        first = asyncio.create_task(handler._run_fast_hermes_reef_command(TREND_QUERY, "trends"))
        await started.wait()
        await handler._run_fast_hermes_reef_command(TREND_QUERY, "trends")
        release.set()
        await first

    spoken = [
        call.kwargs["item"]["content"][0]["text"]
        for call in handler.connection.conversation.item.create.await_args_list
    ]
    assert any(str(_TREND_CACHE["report"]) in text for text in spoken)
    assert all(live_report not in text for text in spoken)
    assert "skipping late Hermes speech" in caplog.text


@pytest.mark.asyncio
async def test_fast_path_skips_speech_when_turn_advances_during_hermes_wait(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A Hermes result from an earlier turn must not be spoken after a newer turn starts."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    async def _send(_text: str, _session_id: str, request_id: str | None = None) -> str:
        handler._turn_generation = 2
        return "Live report after turn change"

    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.send_to_hermes", _send)
    monkeypatch.setattr(
        "reachy_mini_conversation_app.hermes_client.load_latest_reef_thread",
        lambda path=None: dict(_TREND_CACHE),
    )

    with caplog.at_level("INFO"):
        await handler._run_fast_hermes_reef_command(TREND_QUERY, "trends")

    handler.connection.conversation.item.create.assert_not_awaited()
    create.assert_not_awaited()
    assert "newer turn is active" in caplog.text
    assert handler._hermes_spoke_turn is None


@pytest.mark.asyncio
async def test_hermes_reef_fast_path_calls_ask_hermes_when_cache_missing(monkeypatch: Any) -> None:
    """Without a Reefy cache, the Hermes tool remains the report owner."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr("reachy_mini_conversation_app.hermes_client.load_latest_reef_thread", lambda path=None: None)

    class _FakeAskHermes:
        async def __call__(self, deps: object, **kwargs: object) -> dict[str, object]:
            del deps
            assert kwargs.get("query") == "Give me a reef tank report."
            return {"spoken": "Nitrate has been falling from 20 to 8.", "status": "success"}

    monkeypatch.setattr(hf_mod, "AskHermes", _FakeAskHermes)

    await handler._run_fast_hermes_reef_command("Give me a reef tank report.", "detailed_report")

    created = handler.connection.conversation.item.create
    created.assert_awaited_once()
    spoken_prompt = created.await_args.kwargs["item"]["content"][0]["text"]
    assert "Nitrate has been falling from 20 to 8." in spoken_prompt
    create.assert_awaited_once_with(reason="say", response={"tool_choice": "none"})
    assert handler._hermes_spoke_turn == 1
    assert handler._last_reef_response_source == "hermes"
    assert handler._last_reef_response_type == "detailed_report"
    assert handler._last_reef_response_route == "ask_hermes"


@pytest.mark.asyncio
async def test_reef_router_does_not_skip_ask_hermes_as_competing(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """The LLM must not start a second ask_hermes after the router already dispatched it."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._turn_generation = 1
    monkeypatch.setattr(handler, "_run_fast_hermes_reef_command", AsyncMock())
    handler._start_fast_apex_command("Can you give me a reef tank report.")
    start_tool = AsyncMock()
    monkeypatch.setattr(type(handler.tool_manager), "start_tool", start_tool)
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent(
                "response.function_call_arguments.done",
                name="ask_hermes",
                arguments='{"query": "Give me a reef tank report."}',
                call_id="call_hermes",
            ),
        ),
    )

    with caplog.at_level("INFO"):
        await handler._run_realtime_session()

    start_tool.assert_not_awaited()
    assert handler._reef_router_owns_turn == 1
    assert "skipping competing ask_hermes" not in caplog.text
    assert "ask_hermes already dispatched by deterministic route" in caplog.text


@pytest.mark.asyncio
async def test_start_fast_apex_command_routes_explicit_ask_hermes(monkeypatch: Any) -> None:
    """Explicit Ask Hermes reef requests hard-route to ask_hermes."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    apex_run = AsyncMock()
    hermes_run = AsyncMock()
    monkeypatch.setattr(handler, "_run_fast_apex_command", apex_run)
    monkeypatch.setattr(handler, "_run_fast_hermes_reef_command", hermes_run)

    handler._start_fast_apex_command("Reachy, ask Hermes what my Reef Tank report is.")
    assert handler._fast_hermes_reef_task is not None
    await handler._fast_hermes_reef_task

    apex_run.assert_not_awaited()
    hermes_run.assert_awaited_once()
    assert hermes_run.await_args.args[1] == "detailed_report"
    assert hermes_run.await_args.args[2] is True


@pytest.mark.asyncio
async def test_reef_source_question_after_hermes_report(monkeypatch: Any) -> None:
    """A follow-up source question uses stored Hermes metadata."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._record_reef_response("hermes", "detailed_report", "ask_hermes")

    handler._start_fast_apex_command("Did that come from Hermes?")
    assert handler._fast_apex_task is not None
    await handler._fast_apex_task

    spoken_prompt = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert "Yes. That came from Hermes' Reef Tank report." in spoken_prompt
    create.assert_awaited_once_with(reason="say", response={"tool_choice": "none"})


@pytest.mark.asyncio
async def test_reef_source_question_after_local_stats(monkeypatch: Any) -> None:
    """A follow-up source question uses stored Home Assistant metadata."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._record_reef_response("home_assistant", "current_stats", "local")

    handler._start_fast_apex_command("Did that come from Hermes?")
    assert handler._fast_apex_task is not None
    await handler._fast_apex_task

    spoken_prompt = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert "No. That came directly from the current reef tank data in Home Assistant." in spoken_prompt


@pytest.mark.asyncio
async def test_apex_fast_path_speaks_raw_llsato(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """The Apex fast-path must speak LLSATO 2.9 from the tool result, not a percentage."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    class _FakeApex:
        async def __call__(self, deps: object, **kwargs: object) -> dict[str, object]:
            del deps, kwargs
            return {
                "apex_status": {
                    "water_parameters": {"LLSATO": {"value": 2.9, "type": None}},
                    "equipment": {"ato": {"llsato": 2.9}},
                    "cached_at": "2026-08-31T03:22:00.869566Z",
                },
                "source": "apex_status_http",
            }

    monkeypatch.setattr(hf_mod, "Apex", _FakeApex)

    with caplog.at_level("INFO"):
        await handler._run_fast_apex_command("llsato")

    created = handler.connection.conversation.item.create
    created.assert_awaited_once()
    spoken_prompt = created.await_args.kwargs["item"]["content"][0]["text"]
    assert "LLSATO is 2.9." in spoken_prompt
    assert "85" not in spoken_prompt
    assert "%" not in spoken_prompt
    create.assert_awaited_once_with(reason="say", response={"tool_choice": "none"})
    assert handler._apex_spoke_turn == 1
    assert handler._suppress_unsolicited_response_turn == 1
    assert handler._last_reef_response_source == "home_assistant"
    assert handler._last_reef_response_type == "current_stats"
    assert handler._last_reef_response_route == "local"
    assert "[REEF_ROUTER] intent=current_stats" in caplog.text
    assert "[REEF_ROUTER] route=local" in caplog.text
    assert "[REEF_ROUTER] source=home_assistant" in caplog.text
    assert "[REEF_ROUTER] source_type=current_state" in caplog.text
    assert "[REEF_ROUTER] cache_used=false" in caplog.text
    assert "[REEF_ROUTER] response_owner=local" in caplog.text


@pytest.mark.asyncio
async def test_apex_fast_path_drops_unsolicited_followup_response(monkeypatch: Any) -> None:
    """After the Apex route claims the turn, the VAD auto-response must not play."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._turn_generation = 1
    monkeypatch.setattr(handler, "_run_fast_apex_command", AsyncMock())
    handler._start_fast_apex_command("What's the status of my reef tank?")

    captured_cancels: list[int] = []
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("response.created", response_id="resp-auto"),
            _FakeEvent("response.output_audio.delta", delta=_silent_pcm_delta(), response_id="resp-auto"),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript="Let me check the reef tank status for you.",
                response_id="resp-auto",
                item_id="item-auto",
                event_id="evt-auto",
            ),
            _FakeEvent("response.done"),
        ),
        captured_cancels=captured_cancels,
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    outputs = _drain_handler_outputs(handler)
    assert captured_cancels == [1]
    assert _assistant_transcripts(outputs) == []
    assert _audio_frame_count(outputs) == 0


@pytest.mark.asyncio
async def test_bus_fast_path_queues_one_speech_response(monkeypatch: Any) -> None:
    """One bus query should enqueue exactly one say() response.create."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler._turn_generation = 1
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)

    async def _query(*, preparation_threshold: int = 15) -> dict[str, object]:
        del preparation_threshold
        return {"spoken": "Your next 311 bus is arriving in 7 minutes at Macleay St."}

    monkeypatch.setattr(hf_mod.get_bus_monitor(), "query", _query)
    monkeypatch.setattr(hf_mod.get_bus_monitor(), "mark_query_spoken", MagicMock())

    await handler._run_fast_bus_command("query", 15)

    handler.connection.conversation.item.create.assert_awaited_once()
    create.assert_awaited_once_with(reason="say")
    assert handler._bus_spoke_turn == 1
    assert handler._suppress_unsolicited_response_turn == 1


@pytest.mark.asyncio
async def test_duplicate_transcript_event_is_delivered_once(monkeypatch: Any) -> None:
    """The same realtime transcript item must not be shown or logged twice."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    spoken = "Your next 311 bus is arriving in 7 minutes at Macleay St."

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._active_response_reason = "say"
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("response.created", response_id="resp-say"),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript=spoken,
                response_id="resp-say",
                item_id="item-say",
                output_index=0,
                event_id="evt-1",
            ),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript=spoken,
                response_id="resp-say",
                item_id="item-say",
                output_index=0,
                event_id="evt-2",
            ),
            _FakeEvent("response.done"),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert _assistant_transcripts(_drain_handler_outputs(handler)) == [spoken]


@pytest.mark.asyncio
async def test_bus_fast_path_drops_unsolicited_followup_response(monkeypatch: Any) -> None:
    """After the bus fast-path speaks, the VAD auto-response must not play."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler._turn_generation = 1
    monkeypatch.setattr(handler, "say", AsyncMock())

    async def _query(*, preparation_threshold: int = 15) -> dict[str, object]:
        del preparation_threshold
        return {"spoken": "Your next 311 bus is arriving in 7 minutes at Macleay St."}

    monkeypatch.setattr(hf_mod.get_bus_monitor(), "query", _query)
    monkeypatch.setattr(hf_mod.get_bus_monitor(), "mark_query_spoken", MagicMock())
    await handler._run_fast_bus_command("query", 15)

    captured_cancels: list[int] = []
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("response.created", response_id="resp-auto"),
            _FakeEvent("response.output_audio.delta", delta=_silent_pcm_delta(), response_id="resp-auto"),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript="I'll check the next Route 311 bus for you.",
                response_id="resp-auto",
                item_id="item-auto",
                event_id="evt-auto",
            ),
            _FakeEvent("response.done"),
        ),
        captured_cancels=captured_cancels,
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    outputs = _drain_handler_outputs(handler)
    assert captured_cancels == [1]
    assert _assistant_transcripts(outputs) == []
    assert _audio_frame_count(outputs) == 0


@pytest.mark.asyncio
async def test_distinct_response_ids_are_not_suppressed_by_identical_text(monkeypatch: Any) -> None:
    """Two real responses may speak the same words; identity is not the transcript text."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Aiden")
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    spoken = "Your next 311 bus is arriving in 7 minutes at Macleay St."

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        events=(
            _FakeEvent("response.created", response_id="resp-one"),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript=spoken,
                response_id="resp-one",
                item_id="item-one",
                output_index=0,
                event_id="evt-one",
            ),
            _FakeEvent("response.done"),
            _FakeEvent("response.created", response_id="resp-two"),
            _FakeEvent(
                "response.output_audio_transcript.done",
                transcript=spoken,
                response_id="resp-two",
                item_id="item-two",
                output_index=0,
                event_id="evt-two",
            ),
            _FakeEvent("response.done"),
        )
    )
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler._run_realtime_session()

    assert _assistant_transcripts(_drain_handler_outputs(handler)) == [spoken, spoken]


@pytest.mark.asyncio
async def test_home_assistant_query_skips_followup_when_bus_already_spoken(monkeypatch: Any) -> None:
    """If the fast path already announced the 311, the tool follow-up should not speak again."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._in_flight_tool_calls = {"call_1"}
    handler._bus_spoke_turn = handler._turn_generation

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"minutes": 22, "route": "311"},
        )
    )

    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_fast_ha_success_plays_emotion(monkeypatch: Any) -> None:
    """Fast-path device success plays the existing success emotion immediately."""

    class FakeHomeAssistant:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            return {"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.output_queue = asyncio.Queue()
    play = AsyncMock()
    monkeypatch.setattr(hf_mod, "HomeAssistant", FakeHomeAssistant)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)

    await handler._run_fast_ha_commands([{"action": "turn_switch_on", "entity_id": "switch.lamp_3"}])

    play.assert_awaited_once()


@pytest.mark.asyncio
async def test_fast_ha_failure_does_not_play_emotion(monkeypatch: Any) -> None:
    """Fast-path device failure must not play the success emotion."""

    class FakeHomeAssistant:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            return {"error": "Home Assistant could not find that entity or service."}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.output_queue = asyncio.Queue()
    play = AsyncMock()
    monkeypatch.setattr(hf_mod, "HomeAssistant", FakeHomeAssistant)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)

    await handler._run_fast_ha_commands([{"action": "turn_switch_on", "entity_id": "switch.missing"}])

    play.assert_not_awaited()


@pytest.mark.asyncio
async def test_fast_screen_up_success_plays_success_emotion_once(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Confirmed Screen Up activation plays the official success emotion once."""
    ha_calls: list[dict[str, Any]] = []

    class FakeHomeAssistant:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            ha_calls.append(kwargs)
            return {"status": "success", "service": "button.press", "entity_id": "button.screen_up"}

    play_calls: list[dict[str, Any]] = []

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            play_calls.append(kwargs)
            return {"status": "queued", "emotion": "success1"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(hf_mod, "HomeAssistant", FakeHomeAssistant)
    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)

    with caplog.at_level("INFO"):
        await handler._run_fast_ha_commands([{"action": "press_button", "entity_id": "button.screen_up"}])
        await handler._run_fast_ha_commands([{"action": "press_button", "entity_id": "button.screen_up"}])

    assert ha_calls == [
        {"action": "press_button", "entity_id": "button.screen_up"},
        {"action": "press_button", "entity_id": "button.screen_up"},
    ]
    assert play_calls == [{"emotion": "success", "allow_random": False}]
    assert "[SCREEN UP] User requested Screen Up" in caplog.text
    assert "[SCREEN UP] Home Assistant turn-on successful" in caplog.text
    assert "[SCREEN UP] Playing success emotion" in caplog.text
    assert "[SCREEN UP] success emotion completed" in caplog.text


@pytest.mark.asyncio
async def test_fast_screen_up_failure_does_not_play_success_emotion(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed Screen Up Home Assistant action must not play success."""

    class FakeHomeAssistant:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            return {"error": "Home Assistant could not find that entity or service."}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.output_queue = asyncio.Queue()
    play = AsyncMock()
    monkeypatch.setattr(hf_mod, "HomeAssistant", FakeHomeAssistant)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)

    with caplog.at_level("INFO"):
        await handler._run_fast_ha_commands([{"action": "press_button", "entity_id": "button.screen_up"}])

    play.assert_not_awaited()
    assert "[SCREEN UP] Home Assistant turn-on failed" in caplog.text
    assert "[SCREEN UP] success emotion skipped" in caplog.text
    assert "[SCREEN UP] Playing success emotion" not in caplog.text


@pytest.mark.asyncio
async def test_screen_up_tool_result_plays_success_after_confirmation(monkeypatch: Any) -> None:
    """LLM home_assistant Screen Up success plays the official success emotion."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    play = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)
    handler._in_flight_tool_calls = {"call_1"}
    handler._turn_screen_up_call_ids = {"call_1"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "success", "service": "button.press", "entity_id": "button.screen_up"},
        )
    )

    play.assert_awaited_once_with(screen_up=True)
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})


@pytest.mark.asyncio
async def test_screen_up_tool_result_failure_does_not_play_success(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed LLM Screen Up tool result keeps existing error handling and skips success."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    play = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", play)
    handler._in_flight_tool_calls = {"call_1"}
    handler._turn_screen_up_call_ids = {"call_1"}

    with caplog.at_level("INFO"):
        await handler._handle_tool_result(
            ToolNotification(
                id="call_1",
                tool_name="home_assistant",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result={"error": "Home Assistant is currently unavailable."},
            )
        )

    play.assert_not_awaited()
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})
    assert "[SCREEN UP] Home Assistant turn-on failed" in caplog.text
    assert "[SCREEN UP] success emotion skipped" in caplog.text


@pytest.mark.asyncio
async def test_screen_up_emotion_failure_does_not_fail_home_assistant(
    monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    """Screen Up stays successful if the official success emotion cannot play."""

    class FakeHomeAssistant:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            return {"status": "success", "service": "button.press", "entity_id": "button.screen_up"}

    class FakePlayEmotion:
        async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> dict[str, Any]:
            return {"error": "Emotion library not available"}

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(hf_mod, "HomeAssistant", FakeHomeAssistant)
    monkeypatch.setattr(hf_mod, "PlayEmotion", FakePlayEmotion)

    with caplog.at_level("INFO"):
        await handler._run_fast_ha_commands([{"action": "press_button", "entity_id": "button.screen_up"}])

    transcripts = _assistant_transcripts(_drain_handler_outputs(handler))
    assert json.loads(transcripts[0]) == {
        "status": "success",
        "service": "button.press",
        "entity_id": "button.screen_up",
    }
    assert "[SCREEN UP] Home Assistant turn-on successful" in caplog.text
    assert "[SCREEN UP] Unable to play success emotion: Emotion library not available" in caplog.text
    assert "[SCREEN UP] success emotion completed" not in caplog.text


@pytest.mark.asyncio
async def test_same_turn_hermes_result_is_spoken_once(monkeypatch: Any) -> None:
    """A successful same-turn Hermes result is submitted once and marked delivered after audio."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._turn_generation = 1
    handler._tool_call_generation = {"call_1": 1}
    handler._in_flight_tool_calls = {"call_1"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_1",
            tool_name="ask_hermes",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"reply": "Nitrate has been falling from 20 to 8."},
        )
    )

    handler.connection.conversation.item.create.assert_awaited_once()
    created_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert created_item["type"] == "function_call_output"
    assert created_item["call_id"] == "call_1"
    create.assert_awaited_once_with(reason="tool_result:ask_hermes", response={"tool_choice": "none"})
    assert handler._pending_hermes_result is not None
    assert handler._pending_hermes_result.status == "delivering"

    handler._active_response_reason = "tool_result:ask_hermes"
    handler._active_response_audio_delta_count = 4
    await handler._handle_hermes_speech_outcome()
    assert handler._pending_hermes_result.status == "delivered"
    create_count = create.await_count
    await handler._handle_hermes_speech_outcome()
    assert create.await_count == create_count


@pytest.mark.asyncio
async def test_stale_hermes_result_is_buffered(monkeypatch: Any) -> None:
    """A Hermes result from an earlier turn must not be injected into the newer turn."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._turn_generation = 2
    handler._tool_call_generation = {"call_old": 1}
    handler._in_flight_tool_calls = {"call_current"}

    await handler._handle_tool_result(
        ToolNotification(
            id="call_old",
            tool_name="ask_hermes",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"reply": "Mushroom corals are a good starter coral."},
        )
    )

    handler.connection.conversation.item.create.assert_not_awaited()
    assert create.await_count == 0
    assert handler._in_flight_tool_calls == {"call_current"}
    assert handler._pending_hermes_result is not None
    assert handler._pending_hermes_result.status == "buffered"
    assert handler._pending_hermes_result.request_id == "call_old"
    assert handler._pending_hermes_result.originating_turn_id == 1


@pytest.mark.asyncio
async def test_stale_hermes_result_does_not_contaminate_lamp_turn(monkeypatch: Any) -> None:
    """Lamp Home Assistant output stays clean while a late Hermes result is held for later."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    monkeypatch.setattr(handler, "_play_device_success_emotion", AsyncMock())
    handler._turn_generation = 2
    handler._tool_call_generation = {"call_old": 1, "call_lamp": 2}
    handler._in_flight_tool_calls = {"call_lamp"}
    handler._response_done_event.clear()

    await handler._handle_tool_result(
        ToolNotification(
            id="call_old",
            tool_name="ask_hermes",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"reply": "Nitrate has been falling from 20 to 8."},
        )
    )
    await handler._handle_tool_result(
        ToolNotification(
            id="call_lamp",
            tool_name="home_assistant",
            is_idle_tool_call=False,
            status=ToolState.COMPLETED,
            result={"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"},
        )
    )

    created_items = [call.kwargs["item"] for call in handler.connection.conversation.item.create.await_args_list]
    assert created_items == [
        {
            "type": "function_call_output",
            "call_id": "call_lamp",
            "output": json.dumps({"status": "success", "service": "switch.turn_on", "entity_id": "switch.lamp_3"}),
        }
    ]
    create.assert_awaited_once_with(reason="tool_result:home_assistant", response={"tool_choice": "none"})
    assert handler._pending_hermes_result is not None
    assert handler._pending_hermes_result.status == "buffered"

    handler._response_done_event.set()
    handler._active_response_reason = "tool_result:home_assistant"
    await handler._handle_hermes_speech_outcome()
    assert create.await_count == 2
    create.assert_awaited_with(reason="hermes_buffered_result", response={"tool_choice": "none"})
    deliver_item = handler.connection.conversation.item.create.await_args.kwargs["item"]
    assert deliver_item["type"] == "message"
    assert "Nitrate has been falling" in deliver_item["content"][0]["text"]


@pytest.mark.asyncio
async def test_zero_audio_hermes_followup_retries_once(monkeypatch: Any) -> None:
    """A silent Hermes follow-up is retried once through the existing speech path."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._pending_hermes_result = hf_mod._HermesPendingResult(
        "call_1", 1, "Nitrate has been falling from 20 to 8.", "delivering"
    )
    handler._active_response_reason = "tool_result:ask_hermes"
    handler._active_response_audio_delta_count = 0

    await handler._handle_hermes_speech_outcome()

    create.assert_awaited_once_with(reason="hermes_speech_retry", response={"tool_choice": "none"})
    assert handler._pending_hermes_result.speech_attempts == 1
    handler._active_response_reason = "hermes_speech_retry"
    handler._active_response_audio_delta_count = 2
    await handler._handle_hermes_speech_outcome()
    assert handler._pending_hermes_result.status == "delivered"
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_zero_audio_hermes_retry_uses_deterministic_fallback(monkeypatch: Any) -> None:
    """A second silent Hermes follow-up speaks a short fallback instead of dropping the result."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    pending = hf_mod._HermesPendingResult("call_1", 1, "Nitrate has been falling from 20 to 8.", "delivering")
    pending.speech_attempts = 1
    handler._pending_hermes_result = pending
    handler._active_response_reason = "hermes_speech_retry"
    handler._active_response_audio_delta_count = 0

    await handler._handle_hermes_speech_outcome()

    create.assert_awaited_once_with(reason="hermes_speech_fallback", response={"tool_choice": "none"})
    spoken = handler.connection.conversation.item.create.await_args.kwargs["item"]["content"][0]["text"]
    assert _HERMES_SPEECH_FALLBACK in spoken
    assert pending.status == "delivering"


@pytest.mark.asyncio
async def test_silent_ask_hermes_start_speaks_a_hold_line(monkeypatch: Any) -> None:
    """A long Hermes check with no spoken audio should get a short hold-on line."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._turn_generation = 1
    handler._in_flight_tool_calls = {"call_1"}
    handler._active_response_transcript_seen = False
    handler._active_response_audio_delta_count = 0

    await handler._acknowledge_long_running_tool("ask_hermes", "call_1", 1)

    handler.connection.conversation.item.create.assert_awaited_once()
    create.assert_awaited_once_with(reason="tool_hold:ask_hermes", response={"tool_choice": "none"})
    handler.connection.session.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_ask_hermes_hold_line_skips_when_model_already_spoke(monkeypatch: Any) -> None:
    """Do not inject a hold-on line if the function-call turn already had audio."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._turn_generation = 1
    handler._in_flight_tool_calls = {"call_1"}
    handler._active_response_transcript_seen = True

    await handler._acknowledge_long_running_tool("ask_hermes", "call_1", 1)

    handler.connection.conversation.item.create.assert_not_awaited()
    assert create.await_count == 0


@pytest.mark.asyncio
async def test_ask_hermes_hold_line_skips_when_tool_already_finished(monkeypatch: Any) -> None:
    """A fast already-running result should speak through the tool follow-up, not a hold line."""
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
    handler._turn_generation = 1
    handler._in_flight_tool_calls = set()

    await handler._acknowledge_long_running_tool("ask_hermes", "call_1", 1)

    handler.connection.conversation.item.create.assert_not_awaited()
    assert create.await_count == 0


def test_handler_uses_hf_startup_voice_at_startup(monkeypatch: Any) -> None:
    """Hugging Face startup should restore persisted HF voices."""
    handler = HuggingFaceRealtimeHandler(
        ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()),
        startup_voice="Aiden",
    )

    assert handler.get_current_voice() == "Aiden"


def test_handler_ignores_unsupported_hf_profile_voice(monkeypatch: Any) -> None:
    """Unsupported profile voices should not be sent to the Hugging Face backend."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "cedar")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == HF_DEFAULT_VOICE
    session = handler._get_session_config([])
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_handler_normalizes_hf_voice_case(monkeypatch: Any) -> None:
    """Lowercase Hugging Face speaker names should resolve to the curated UI value."""
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "serena")

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    assert handler.get_current_voice() == "Serena"


@pytest.mark.asyncio
async def test_local_response_create_uses_session_tools_only(monkeypatch: Any) -> None:
    """Local follow-ups should not resend the session's full tool registry."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = MagicMock()
    response_sent = asyncio.Event()

    async def acknowledge_response(**_kwargs: Any) -> None:
        handler._response_started_or_rejected_event.set()
        handler._response_done_event.set()
        response_sent.set()

    handler.connection.response.create = AsyncMock(side_effect=acknowledge_response)
    await handler._safe_response_create(reason="tool_result:reef_status")
    sender_task = asyncio.create_task(handler._response_sender_loop())

    await asyncio.wait_for(response_sent.wait(), timeout=1.0)
    sender_task.cancel()
    await sender_task

    handler.connection.response.create.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_tool_followup_passes_tool_choice_on_response_create(monkeypatch: Any) -> None:
    """Spoken tool follow-ups must not flip the session tool_choice."""
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = MagicMock()
    response_sent = asyncio.Event()

    async def acknowledge_response(**_kwargs: Any) -> None:
        handler._response_started_or_rejected_event.set()
        handler._response_done_event.set()
        response_sent.set()

    handler.connection.response.create = AsyncMock(side_effect=acknowledge_response)
    await handler._safe_response_create(reason="tool_result:home_assistant", response={"tool_choice": "none"})
    sender_task = asyncio.create_task(handler._response_sender_loop())

    await asyncio.wait_for(response_sent.wait(), timeout=1.0)
    sender_task.cancel()
    await sender_task

    handler.connection.response.create.assert_awaited_once_with(response={"tool_choice": "none"})
    handler.connection.session.update.assert_not_called()


@pytest.mark.asyncio
async def test_response_create_retries_when_server_does_not_acknowledge(monkeypatch: Any) -> None:
    """A dropped response.create acknowledgement should not stall the conversation."""
    monkeypatch.setattr(hf_mod, "_RESPONSE_STARTED_TIMEOUT", 0.01)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = MagicMock()
    response_sent = asyncio.Event()
    send_count = 0

    async def acknowledge_retry(**_kwargs: Any) -> None:
        nonlocal send_count
        send_count += 1
        if send_count == 2:
            handler._response_started_or_rejected_event.set()
            handler._response_done_event.set()
            response_sent.set()

    handler.connection.response.create = AsyncMock(side_effect=acknowledge_retry)
    await handler._safe_response_create(reason="tool_result:reef_status")
    sender_task = asyncio.create_task(handler._response_sender_loop())

    await asyncio.wait_for(response_sent.wait(), timeout=1.0)
    sender_task.cancel()
    await sender_task

    assert handler.connection.response.create.await_count == 2


@pytest.mark.asyncio
async def test_run_realtime_session_uses_default_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Use the backend default speaker when no profile voice is selected for the hf LB."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_update=captured_update)

    await handler._run_realtime_session()

    session = captured_update["session"]
    # HF at 16 kHz passes None so the backend uses its optimal default (16 kHz).
    assert session["audio"]["input"]["format"]["rate"] is None
    assert session["audio"]["output"]["format"]["rate"] is None
    assert session["audio"]["input"]["transcription"]["language"] == "en"
    assert session["audio"]["output"]["voice"] == HF_DEFAULT_VOICE


def test_huggingface_session_uses_configured_transcription_language(monkeypatch: Any) -> None:
    """Hugging Face realtime sessions should forward the configured transcription language."""
    monkeypatch.setattr(config, "REALTIME_TRANSCRIPTION_LANGUAGE", "zh")
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    session = handler._get_session_config([])

    assert session["audio"]["input"]["transcription"]["language"] == "zh"


@pytest.mark.asyncio
async def test_run_realtime_session_passes_allocated_session_query(monkeypatch: Any) -> None:
    """Hugging Face sessions must forward the allocated session token to the websocket connect call."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])

    captured_connect: dict[str, Any] = {}
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(captured_connect=captured_connect)
    handler._realtime_connect_query = {"session_token": "abc123"}

    await handler._run_realtime_session()

    assert "model" not in captured_connect
    assert captured_connect["extra_query"] == {"session_token": "abc123"}


@pytest.mark.parametrize(("hf_token", "expected_api_key"), [(None, "DUMMY"), ("hf-secret", "hf-secret")])
@pytest.mark.asyncio
async def test_build_realtime_client_local_uses_explicit_hf_token_only(
    monkeypatch: Any,
    hf_token: str | None,
    expected_api_key: str,
) -> None:
    """Local websocket mode must never forward cached Hugging Face credentials."""
    client_kwargs: dict[str, Any] = {}

    def _no_allocator(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("session allocator should not be called in direct websocket mode")

    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _no_allocator)
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "local")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: "hf-cached")
    monkeypatch.setattr(
        config,
        "HF_REALTIME_WS_URL",
        "ws://127.0.0.1:8765/v1/realtime?session_token=abc123&model=ignored-by-sdk",
    )

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "http://127.0.0.1:8765/v1"
    assert client_kwargs["websocket_base_url"] == "ws://127.0.0.1:8765/v1"
    assert handler._realtime_connect_query == {"session_token": "abc123"}


@pytest.mark.parametrize(
    (
        "hf_token",
        "cached_token",
        "hardware_id",
        "status_error",
        "expected_header",
        "expected_api_key",
        "expected_payload",
    ),
    [
        (
            "hf-secret",
            "hf-cached",
            "0123456789abcdef",
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-secret",
            },
            "hf-secret",
            {"hardware_id": "0123456789abcdef"},
        ),
        (
            None,
            "hf-cached",
            None,
            None,
            {
                "User-Agent": "reachy-mini-conversation-app",
                "X-Reachy-Mini-Authorization": "Bearer hf-cached",
            },
            "hf-cached",
            {},
        ),
        (None, None, None, None, {"User-Agent": "reachy-mini-conversation-app"}, "DUMMY", {}),
        (
            None,
            None,
            None,
            TimeoutError("status unavailable"),
            {"User-Agent": "reachy-mini-conversation-app"},
            "DUMMY",
            {},
        ),
    ],
)
@pytest.mark.asyncio
async def test_build_realtime_client_deployed_resolves_hf_token(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
    hf_token: str | None,
    cached_token: str | None,
    hardware_id: str | None,
    status_error: Exception | None,
    expected_header: dict[str, str],
    expected_api_key: str,
    expected_payload: dict[str, str],
) -> None:
    """Deployed allocation reports available credentials and robot identity."""
    client_kwargs: dict[str, Any] = {}
    posts: list[tuple[str, dict[str, str] | None, dict[str, str] | None]] = []
    connect_url = "wss://hf.example.test/v1/realtime?session_token=allocated"
    monkeypatch.setattr(hf_mod, "AsyncOpenAI", _fake_openai_client(client_kwargs))
    monkeypatch.setattr(hf_mod.httpx, "AsyncClient", _fake_allocator(connect_url, posts))
    monkeypatch.setattr(config, "HF_REALTIME_CONNECTION_MODE", "deployed")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")
    # A stale local URL must be ignored in deployed mode.
    monkeypatch.setattr(config, "HF_REALTIME_WS_URL", "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(config, "HF_TOKEN", hf_token)
    monkeypatch.setattr(hf_mod, "get_token", lambda: cached_token)

    reachy_mini = MagicMock()
    reachy_mini.client.get_status.return_value.hardware_id = hardware_id
    if status_error:
        reachy_mini.client.get_status.side_effect = status_error
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=reachy_mini, movement_manager=MagicMock()))

    client = await handler._build_realtime_client()

    assert client is not None
    assert posts == [("https://lb.example.test/session", expected_header, expected_payload)]
    reachy_mini.client.get_status.assert_called_once_with(wait=False)
    if status_error:
        assert "Daemon status unavailable for realtime session allocation" in caplog.text
    assert client_kwargs["api_key"] == expected_api_key
    assert client_kwargs["base_url"] == "https://hf.example.test/v1"
    assert client_kwargs["websocket_base_url"] == "wss://hf.example.test/v1"
    assert handler._realtime_connect_query == {"session_token": "allocated"}


@pytest.mark.asyncio
async def test_apply_personality_uses_selected_voice_for_lb_allocated_sessions(monkeypatch: Any) -> None:
    """Live personality updates should honor the selected Qwen CustomVoice speaker."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: "Serena")
    monkeypatch.setattr(config, "HF_REALTIME_SESSION_URL", "https://lb.example.test/session")

    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    monkeypatch.setattr(handler, "_restart_session", AsyncMock(return_value=None))

    result = await handler.apply_personality("mars_rover")

    assert "restarted realtime session" in result.lower()
    session = captured_update["session"]
    assert session["instructions"] == "new instructions"
    assert session["audio"]["output"]["voice"] == "Serena"


@pytest.mark.asyncio
async def test_apply_personality_restores_profile_when_tools_fail(monkeypatch: Any) -> None:
    """A failed tool reload should leave the previous profile selected."""
    selected_profiles: list[str | None] = []

    def select_profile(profile: str | None) -> None:
        selected_profiles.append(profile)
        config.REACHY_MINI_CUSTOM_PROFILE = profile

    def fail_tool_reload(*, force: bool = False) -> None:
        raise RuntimeError("tool reload failed")

    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr(hf_mod, "set_custom_profile", select_profile)
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "new instructions")
    monkeypatch.setattr(hf_mod.core_tools, "initialize_tools", fail_tool_reload)
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))

    result = await handler.apply_personality("broken")

    assert result == "Failed to apply personality: tool reload failed"
    assert config.REACHY_MINI_CUSTOM_PROFILE == "default"
    assert selected_profiles == ["broken", "default"]


@pytest.mark.asyncio
async def test_change_voice_updates_live_hf_session_without_restart(monkeypatch: Any) -> None:
    """Changing Hugging Face voice should update the active session in place."""
    captured_update: dict[str, Any] = {}

    class FakeSession:
        async def update(self, **kwargs: Any) -> None:
            captured_update.update(kwargs)

    class FakeConnection:
        session = FakeSession()

    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = FakeConnection()
    restart = AsyncMock(return_value=None)
    monkeypatch.setattr(handler, "_restart_session", restart)

    result = await handler.change_voice("Serena")

    assert result == "Voice changed to Serena."
    assert handler.get_current_voice() == "Serena"
    restart.assert_not_awaited()
    session = captured_update["session"]
    assert session["audio"]["output"]["voice"] == "Serena"


def test_session_limit_helpers() -> None:
    """Pool occupancy and 1008 close text must be recognized without a live server."""
    assert hf_mod._is_session_limit_error(RuntimeError("All session slots are in use"))
    assert hf_mod._is_session_limit_error(RuntimeError("error_type=session_limit_reached"))
    assert not hf_mod._is_session_limit_error(RuntimeError("no close frame received"))
    assert hf_mod._realtime_pool_has_idle_slot({"size": 1, "in_use": 0, "units": [{"state": "idle"}]})
    assert not hf_mod._realtime_pool_has_idle_slot({"size": 1, "in_use": 1, "units": [{"state": "active"}]})
    assert hf_mod._realtime_pool_is_stuck({"units": [{"state": "stuck"}]})
    assert not hf_mod._realtime_pool_is_stuck({"units": [{"state": "draining"}]})


@pytest.mark.asyncio
async def test_run_realtime_session_raises_slots_busy(monkeypatch: Any) -> None:
    """A local one-session pool rejection must not be logged as a generic session.update crash."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(
        update_errors=[RuntimeError("received 1008 (policy violation) All session slots are in use")]
    )

    with pytest.raises(RealtimeSessionSlotsBusy, match="session slots are in use"):
        await handler._run_realtime_session()


@pytest.mark.asyncio
async def test_start_up_waits_when_session_slots_are_full(monkeypatch: Any, caplog: pytest.LogCaptureFixture) -> None:
    """Startup must wait for `/v1/pool` instead of retrying a full slot every second."""
    monkeypatch.setattr(hf_mod, "get_session_instructions", lambda _instance_path=None: "test")
    monkeypatch.setattr(hf_mod, "get_session_voice", lambda default=HF_DEFAULT_VOICE: default)
    monkeypatch.setattr(hf_mod, "get_tool_specs", lambda: [])
    monkeypatch.setattr(hf_mod, "get_hf_direct_ws_url", lambda: "ws://127.0.0.1:8765/v1/realtime")
    monkeypatch.setattr(
        hf_mod.httpx,
        "AsyncClient",
        _fake_pool_client({"size": 1, "in_use": 0, "units": [{"index": 0, "state": "idle"}]}),
    )
    monkeypatch.setattr(hf_mod.asyncio, "sleep", AsyncMock())

    update_errors: list[BaseException] = [
        RuntimeError("received 1008 (policy violation) All session slots are in use")
    ]
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.client = _make_fake_realtime_client(update_errors=update_errors)
    monkeypatch.setattr(handler, "_build_realtime_client", AsyncMock(return_value=handler.client))
    monkeypatch.setattr(type(handler.tool_manager), "start_up", MagicMock())
    monkeypatch.setattr(type(handler.tool_manager), "shutdown", AsyncMock())

    await handler.start_up()

    assert update_errors == []
    assert "no free session slot" in caplog.text
