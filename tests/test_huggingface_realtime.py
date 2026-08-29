import json
import time
import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.conversation_handler as conv_mod
import reachy_mini_conversation_app.huggingface_realtime as hf_mod
from reachy_mini_conversation_app.config import config, get_default_voice
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.huggingface_realtime import (
    _HERMES_SPEECH_FALLBACK,
    RealtimeSessionSlotsBusy,
    HuggingFaceRealtimeHandler,
)
from reachy_mini_conversation_app.tools.home_assistant import HomeAssistant
from reachy_mini_conversation_app.tools.background_tool_manager import ToolState, ToolNotification


HF_DEFAULT_VOICE = get_default_voice()


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
            pass

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
async def test_home_assistant_control_skips_spoken_followup(monkeypatch: Any) -> None:
    """Successful local HA control should not start a slow spoken follow-up."""
    monkeypatch.setattr(hf_mod.core_tools, "get_tools", lambda: {"home_assistant": HomeAssistant()})
    handler = HuggingFaceRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock()))
    handler.connection = AsyncMock()
    handler.output_queue = asyncio.Queue()
    monkeypatch.setattr(handler, "_wait_for_response_done_before_tool_result", AsyncMock(return_value=True))
    create = AsyncMock()
    monkeypatch.setattr(handler, "_safe_response_create", create)
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

    handler.connection.conversation.item.create.assert_awaited_once()
    assert create.await_count == 0


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
    assert create.await_count == 0
    assert handler._pending_hermes_result is not None
    assert handler._pending_hermes_result.status == "buffered"

    handler._response_done_event.set()
    handler._active_response_reason = "tool_result:home_assistant"
    await handler._handle_hermes_speech_outcome()
    create.assert_awaited_once_with(reason="hermes_buffered_result", response={"tool_choice": "none"})
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
