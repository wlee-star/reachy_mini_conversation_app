import json
import time
import uuid
import base64
import random
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Final, Tuple, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from pydantic import Field, BaseModel
from numpy.typing import NDArray
from huggingface_hub import get_token
from typing_extensions import Literal, TypedDict
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeToolsConfigParam,
    RealtimeFunctionToolParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from websockets.exceptions import ConnectionClosedError
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.tools import core_tools
from reachy_mini_conversation_app.config import (
    HF_LOCAL_CONNECTION_MODE,
    config,
    get_default_voice,
    set_custom_profile,
    get_available_voices,
    get_hf_direct_ws_url,
    parse_hf_realtime_url,
    get_hf_connection_selection,
)
from reachy_mini_conversation_app.prompts import (
    get_session_voice,
    get_session_instructions,
    get_session_greeting_prompt,
)
from reachy_mini_conversation_app.streaming import AdditionalOutputs, audio_to_int16
from reachy_mini_conversation_app.tools.core_tools import (
    ToolSpec,
    ToolDependencies,
    get_tool_specs,
)
from reachy_mini_conversation_app.tools.play_emotion import PlayEmotion, is_success_emotion_request
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.home_assistant import (
    HomeAssistant,
    is_control_action,
    match_fast_ha_commands,
    is_device_control_success,
)
from reachy_mini_conversation_app.tools.tool_constants import ToolState
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


if TYPE_CHECKING:
    from openai.resources.realtime.realtime import AsyncRealtimeConnection


logger = logging.getLogger(__name__)

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0
_RESPONSE_STARTED_TIMEOUT: Final[float] = 3.0
_RESPONSE_REJECTION_RETRY_DELAY: Final[float] = 0.5
_SESSION_SLOT_WAIT_S: Final[float] = 15.0
_SESSION_SLOT_POLL_S: Final[float] = 0.5
_SESSION_LIMIT_MARKERS: Final[tuple[str, ...]] = (
    "session slots are in use",
    "session_limit_reached",
)
_LONG_RUNNING_TOOLS: Final[frozenset[str]] = frozenset({"ask_hermes"})
_LONG_TOOL_HOLD_PROMPT: Final[str] = (
    "The long check already started. Speak one short sentence that you are on it. This is not a new user request."
)
# Per-response, not session.update: a session tool_choice flip forces the local LLM to re-prefill.
_TOOL_FOLLOWUP_CREATE_KWARGS: Final[dict[str, dict[str, str]]] = {"response": {"tool_choice": "none"}}
_HERMES_SPEECH_FALLBACK: Final[str] = "I've got the reef result ready, but I couldn't play the response."
_HERMES_DELIVER_PROMPT: Final[str] = (
    "A previous check finished. Speak this result now in one or two short sentences, then stop. "
    "Do not mention tools, files, waiting, or that this was delayed: {text}"
)
_HERMES_RETRY_PROMPT: Final[str] = (
    "Speak the previous check result now in one or two short sentences. Do not mention tools or files."
)
_HERMES_SPEECH_REASONS: Final[frozenset[str]] = frozenset(
    {
        "tool_result:ask_hermes",
        "hermes_buffered_result",
        "hermes_speech_retry",
        "hermes_speech_fallback",
    }
)


class RealtimeSessionSlotsBusy(RuntimeError):
    """Speech-to-speech rejected the websocket because its pipeline pool is full."""


class _HermesPendingResult:
    """One Hermes tool result waiting to be spoken at most once."""

    def __init__(self, request_id: str, originating_turn_id: int, text: str, status: str) -> None:
        self.request_id = request_id
        self.originating_turn_id = originating_turn_id
        self.text = text
        self.completed_at = time.monotonic()
        self.status = status
        self.speech_attempts = 0


def _hermes_result_text(tool_result: object) -> str:
    """Return the spoken Hermes payload, or empty if there is nothing to deliver."""
    if isinstance(tool_result, dict):
        if tool_result.get("status") == "already_running":
            return ""
        reply = tool_result.get("reply")
        if isinstance(reply, str) and reply.strip():
            return reply.strip()
        error = tool_result.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
    if isinstance(tool_result, str) and tool_result.strip():
        return tool_result.strip()
    return ""


def _is_session_limit_error(exc: BaseException) -> bool:
    """Return whether a realtime close is the local one-session pool limit."""
    text = str(exc).lower()
    return any(marker in text for marker in _SESSION_LIMIT_MARKERS)


def _realtime_pool_has_idle_slot(payload: dict[str, Any]) -> bool:
    """Return whether speech-to-speech `/v1/pool` has a claimable unit."""
    units = payload.get("units")
    if isinstance(units, list) and units:
        return any(isinstance(unit, dict) and unit.get("state") == "idle" for unit in units)
    in_use = payload.get("in_use")
    size = payload.get("size")
    if isinstance(in_use, int) and isinstance(size, int):
        return in_use < size
    return False


def _realtime_pool_is_stuck(payload: dict[str, Any]) -> bool:
    """Return whether any pipeline unit is quarantined and unclaimable."""
    units = payload.get("units")
    if not isinstance(units, list):
        return False
    return any(isinstance(unit, dict) and unit.get("state") == "stuck" for unit in units)


class InputTranscriptChunksByItem(BaseModel):
    """Current item_id and its accumulated deltas. Only one item at a time."""

    item_id: str | None = None
    deltas: list[str] = Field(default_factory=list)


def to_realtime_tools_config(tool_specs: list[ToolSpec]) -> RealtimeToolsConfigParam:
    """Convert app tool specs to the OpenAI-compatible realtime session shape."""
    realtime_tools: RealtimeToolsConfigParam = []
    for spec in tool_specs:
        realtime_tools.append(
            RealtimeFunctionToolParam(
                type="function",
                name=spec["name"],
                description=spec["description"],
                parameters=spec["parameters"],
            )
        )
    return realtime_tools


class HFNativeRateAudioPCM(TypedDict):
    """Hugging Face extension for native-rate PCM audio."""

    type: Literal["audio/pcm"]
    rate: None


def _native_rate_audio_pcm() -> HFNativeRateAudioPCM:
    """Return the Hugging Face native-rate PCM config."""
    return {"type": "audio/pcm", "rate": None}


def _build_openai_compatible_client_from_realtime_url(
    realtime_url: str,
    bearer_token: str | None,
) -> tuple[AsyncOpenAI, dict[str, str]]:
    """Build an OpenAI-compatible realtime client from a direct websocket/base URL."""
    parsed = parse_hf_realtime_url(realtime_url)
    client = AsyncOpenAI(
        api_key=bearer_token or "DUMMY",
        base_url=parsed.base_url,
        websocket_base_url=parsed.websocket_base_url,
    )
    return client, parsed.connect_query


class HuggingFaceRealtimeHandler(ConversationHandler):
    """Realtime stream handler for the Hugging Face OpenAI-compatible endpoint."""

    SAMPLE_RATE = 16000

    def __init__(
        self,
        deps: ToolDependencies,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        super().__init__()

        self.deps = deps

        self.client: AsyncOpenAI
        self.connection: "AsyncRealtimeConnection | None" = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.instance_path = instance_path
        self._voice_override: str | None = self._normalize_startup_voice(startup_voice)
        self._realtime_connect_query: dict[str, str] = {}

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_debounce_delay = 0.5  # seconds
        self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._active_response_reason: str | None = None
        self._active_response_started_at: float | None = None
        self._active_response_audio_delta_count = 0
        self._active_response_audio_bytes = 0
        self._active_response_transcript_seen = False
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None
        self._turn_speech_started_at: float | None = None
        self._turn_speech_stopped_at: float | None = None
        self._turn_tool_received_at: float | None = None
        self._turn_generation = 0
        self._tool_call_generation: dict[str, int] = {}
        self._startup_greeting_sent = False
        self._in_flight_tool_calls: set[str] = set()
        self._tool_batch_needs_response = False
        self._tool_followup_tools_disabled = False
        self._fast_ha_task: asyncio.Task[None] | None = None
        self._user_speech_in_progress = False
        self._pending_hermes_result: _HermesPendingResult | None = None
        self._turn_device_control_call_ids: set[str] = set()
        self._device_success_emotion_played = False

    @staticmethod
    def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Remove bulky transport-only fields before echoing tool output back to the model."""
        if tool_name == "camera" and "b64_im" in tool_result:
            sanitized = dict(tool_result)
            sanitized.pop("b64_im", None)
            sanitized["image_attached"] = True
            return sanitized
        return tool_result

    def _normalize_startup_voice(self, voice: str | None) -> str | None:
        """Return a valid persisted startup voice, or None."""
        return self._resolve_backend_voice(voice, source="persisted startup voice")

    async def _wait_for_response_done_before_tool_result(self) -> bool:
        """Return whether the function-call response finished before sending tool output."""
        if self._response_done_event.is_set():
            return True

        try:
            await asyncio.wait_for(
                self._response_done_event.wait(),
                timeout=_RESPONSE_DONE_TIMEOUT,
            )
            return True
        except asyncio.TimeoutError:
            return False

    def _reset_active_response_audio_state(self) -> None:
        """Clear leftover audio/transcript flags so a silent tool turn is detectable."""
        self._active_response_audio_delta_count = 0
        self._active_response_audio_bytes = 0
        self._active_response_transcript_seen = False

    def _start_fast_ha_command(self, transcript: str) -> None:
        commands = match_fast_ha_commands(transcript)
        if not commands:
            return
        self._fast_ha_task = asyncio.create_task(
            self._run_fast_ha_commands(commands),
            name="ha-fast-path",
        )

    async def _run_fast_ha_commands(self, commands: list[dict[str, Any]]) -> None:
        for command in commands:
            logger.info("[HA] fast-path executing %s", command)
            started = time.perf_counter()
            try:
                result = await HomeAssistant()(self.deps, **command)
            except Exception as exc:
                logger.warning("Fast-path Home Assistant command failed: %s", exc)
                continue
            logger.info(
                "[HA] fast-path finished in %.0f ms: %s",
                (time.perf_counter() - started) * 1000,
                result,
            )
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": json.dumps(result)}))
            if is_device_control_success(result if isinstance(result, dict) else None, None):
                await self._play_device_success_emotion()

    def _reset_device_success_turn_state(self) -> None:
        """Clear per-turn device-control success tracking."""
        self._turn_device_control_call_ids.clear()
        self._device_success_emotion_played = False

    def _value_from_tool_args(self, args_json: str, key: str) -> object:
        try:
            parsed = json.loads(args_json)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        return parsed.get(key)

    async def _play_device_success_emotion(self) -> None:
        """Queue the existing success emotion once after a confirmed device action."""
        if self._device_success_emotion_played:
            return
        self._device_success_emotion_played = True
        try:
            result = await PlayEmotion()(self.deps, emotion="success", allow_random=False)
        except Exception as exc:
            logger.warning("Device success emotion failed: %s", exc)
            return
        if result.get("error"):
            logger.warning("Device success emotion failed: %s", result["error"])
            return
        logger.info("Played device success emotion: %s", result.get("emotion"))

    async def _complete_skipped_success_emotion(self, call_id: str) -> None:
        """Finish a play_emotion(success) call without queuing a second animation."""
        await self._handle_tool_result(
            ToolNotification(
                id=call_id,
                tool_name="play_emotion",
                is_idle_tool_call=False,
                status=ToolState.COMPLETED,
                result={"status": "skipped", "emotion": "success"},
            )
        )

    async def _start_or_skip_success_emotion_tool(
        self,
        call_id: str,
        args_json_str: str,
        turn_generation: int,
    ) -> None:
        """Play success only when this turn has no device-control tool owning the emotion."""
        await self._wait_for_response_done_before_tool_result()
        if self._turn_generation != turn_generation:
            await self._complete_skipped_success_emotion(call_id)
            return
        if self._device_success_emotion_played or self._turn_device_control_call_ids:
            logger.info("Skipping play_emotion success; device-control result owns this turn")
            await self._complete_skipped_success_emotion(call_id)
            return
        background_tool = await self.tool_manager.start_tool(
            call_id=call_id,
            tool_call_routine=ToolCallRoutine(
                tool_name="play_emotion",
                args_json_str=args_json_str,
                deps=self.deps,
            ),
            is_idle_tool_call=False,
        )
        logger.info(
            "Started background tool: play_emotion (id=%s, call_id=%s)",
            background_tool.tool_id,
            call_id,
        )

    async def _acknowledge_long_running_tool(self, tool_name: str, call_id: str, turn_generation: int) -> None:
        """Speak a short hold-on line when a long tool starts with no audio."""
        if tool_name not in _LONG_RUNNING_TOOLS:
            return
        await self._wait_for_response_done_before_tool_result()
        if not self.connection:
            return
        if self._turn_generation != turn_generation:
            return
        if self._user_speech_in_progress:
            return
        if call_id not in self._in_flight_tool_calls:
            return
        if self._active_response_transcript_seen or self._active_response_audio_delta_count > 0:
            return
        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": _LONG_TOOL_HOLD_PROMPT}],
                },
            )
            await self._safe_response_create(reason=f"tool_hold:{tool_name}", **_TOOL_FOLLOWUP_CREATE_KWARGS)
        except ConnectionClosedError:
            logger.warning("Connection closed while acknowledging long-running tool %s", tool_name)
        except Exception as exc:
            logger.warning("Failed to acknowledge long-running tool %s: %s", tool_name, exc)

    def _conversation_idle_for_buffered_hermes(self) -> bool:
        """Return whether a buffered Hermes result can be spoken without interrupting the user."""
        if self._user_speech_in_progress:
            return False
        if not self._response_done_event.is_set():
            return False
        if self._in_flight_tool_calls:
            return False
        return True

    def _track_hermes_result(self, request_id: str, originating_turn_id: int, text: str, status: str) -> None:
        """Remember one Hermes result so it can be spoken once, including after a newer turn."""
        existing = self._pending_hermes_result
        if (
            existing is not None
            and existing.request_id == request_id
            and existing.status in {"delivering", "delivered"}
        ):
            logger.info(
                "Hermes result already %s request_id=%s; skipping duplicate",
                existing.status,
                request_id,
            )
            return
        if (
            existing is not None
            and existing.request_id != request_id
            and existing.status in {"buffered", "delivering"}
        ):
            logger.info(
                "Keeping in-flight Hermes result request_id=%s; not replacing with request_id=%s",
                existing.request_id,
                request_id,
            )
            return
        self._pending_hermes_result = _HermesPendingResult(request_id, originating_turn_id, text, status)
        logger.info(
            "Hermes request completed request_id=%s originating_turn=%s status=%s",
            request_id,
            originating_turn_id,
            status,
        )

    def _buffer_hermes_result(self, request_id: str, originating_turn_id: int, text: str) -> None:
        """Hold a completed Hermes result until the newer user turn is idle."""
        self._track_hermes_result(request_id, originating_turn_id, text, "buffered")
        logger.info(
            "Hermes request buffered because newer turn is active request_id=%s originating_turn=%s current_turn=%s",
            request_id,
            originating_turn_id,
            self._turn_generation,
        )

    async def _queue_hermes_prompt(self, text: str, reason: str) -> None:
        """Ask the existing realtime session to speak one Hermes follow-up."""
        pending = self._pending_hermes_result
        if pending is not None and pending.status == "delivered":
            return
        if not self.connection:
            return
        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": text}],
                },
            )
            await self._safe_response_create(reason=reason, **_TOOL_FOLLOWUP_CREATE_KWARGS)
        except ConnectionClosedError:
            logger.warning("Connection closed while queuing Hermes speech reason=%s", reason)
        except Exception as exc:
            logger.warning("Failed to queue Hermes speech reason=%s: %s", reason, exc)

    async def _deliver_buffered_hermes_result(self) -> None:
        """Speak a buffered Hermes result after the interrupting turn has finished."""
        pending = self._pending_hermes_result
        if pending is None or pending.status != "buffered":
            return
        if not self._conversation_idle_for_buffered_hermes():
            return
        pending.status = "delivering"
        logger.info(
            "Hermes result delivered after user interaction became idle request_id=%s originating_turn=%s",
            pending.request_id,
            pending.originating_turn_id,
        )
        await self._queue_hermes_prompt(
            _HERMES_DELIVER_PROMPT.format(text=pending.text),
            "hermes_buffered_result",
        )

    async def _handle_hermes_speech_outcome(self) -> None:
        """Retry once, then fall back, when a Hermes follow-up produced no audio."""
        pending = self._pending_hermes_result
        reason = self._active_response_reason
        if pending is None:
            return
        if reason not in _HERMES_SPEECH_REASONS:
            if pending.status == "buffered":
                await self._deliver_buffered_hermes_result()
            return
        if self._active_response_audio_delta_count > 0:
            logger.info(
                "Hermes speech response produced audio request_id=%s turn=%s reason=%s deltas=%d",
                pending.request_id,
                pending.originating_turn_id,
                reason,
                self._active_response_audio_delta_count,
            )
            pending.status = "delivered"
            logger.info(
                "Hermes result delivered request_id=%s originating_turn=%s",
                pending.request_id,
                pending.originating_turn_id,
            )
            return
        logger.warning(
            "Hermes speech response produced zero audio request_id=%s reason=%s transcript_seen=%s",
            pending.request_id,
            reason,
            self._active_response_transcript_seen,
        )
        if reason == "hermes_speech_fallback":
            pending.status = "delivered"
            logger.info(
                "Hermes speech fallback finished without audio request_id=%s",
                pending.request_id,
            )
            return
        if pending.speech_attempts == 0:
            pending.speech_attempts = 1
            pending.status = "delivering"
            logger.info("Hermes speech retry request_id=%s", pending.request_id)
            await self._queue_hermes_prompt(_HERMES_RETRY_PROMPT, "hermes_speech_retry")
            return
        pending.speech_attempts = 2
        pending.status = "delivering"
        logger.info("Hermes speech fallback request_id=%s", pending.request_id)
        await self._queue_hermes_prompt(
            f"Say exactly this, then stop: {_HERMES_SPEECH_FALLBACK}",
            "hermes_speech_fallback",
        )

    async def _set_tool_followup_choice(self, tool_choice: Literal["auto", "none"]) -> bool:
        if self.connection is None:
            return False
        try:
            await self.connection.session.update(
                session=RealtimeSessionCreateRequestParam(
                    type="realtime",
                    tool_choice=tool_choice,
                )
            )
        except Exception as exc:
            logger.warning("Failed to set tool follow-up choice to %s: %s", tool_choice, exc)
            return False
        self._tool_followup_tools_disabled = tool_choice == "none"
        return True

    def _resolve_backend_voice(
        self,
        voice: str | None,
        *,
        source: str,
        fallback: str | None = None,
    ) -> str | None:
        """Return a backend-supported voice, optionally falling back when unsupported."""
        available_voices = get_available_voices()
        voice_value = (voice or "").strip()
        if not voice_value:
            return fallback

        voice_by_lowercase = {candidate.lower(): candidate for candidate in available_voices}
        normalized_voice = voice_by_lowercase.get(voice_value.lower())
        if normalized_voice is not None:
            return normalized_voice

        if voice:
            logger.warning(
                "Ignoring unsupported %s %r; expected one of %s",
                source,
                voice,
                available_voices,
            )
        return fallback

    def _get_session_config(self, tool_specs: list[ToolSpec]) -> RealtimeSessionCreateRequestParam:
        """Return the Hugging Face OpenAI-compatible session config."""
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=get_session_instructions(self.instance_path),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    # The OpenAI SDK type only includes 24 kHz PCM, but the HF
                    # compatible server uses rate=None for native 16 kHz mode.
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    transcription=AudioTranscriptionParam(
                        model="gpt-4o-transcribe",
                        language=config.REALTIME_TRANSCRIPTION_LANGUAGE,
                    ),
                    turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=_native_rate_audio_pcm(),  # type: ignore[typeddict-item]
                    voice=self.get_current_voice(),
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    def _is_connected(self) -> bool:
        """Return whether the realtime connection is open."""
        return self.connection is not None

    def _idle_behavior_ready(self) -> bool:
        """Hold idle behavior while a model response is still active."""
        return self._response_done_event.is_set()

    async def _cancel_partial_transcript_task(self) -> None:
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

    async def change_voice(self, voice: str) -> str:
        """Change only the voice, updating the active session when possible."""
        default_voice = get_default_voice()
        resolved_voice = (
            self._resolve_backend_voice(voice, source="requested voice", fallback=default_voice) or default_voice
        )
        self._voice_override = resolved_voice
        if self.connection is not None:
            try:
                await self.connection.session.update(
                    session=RealtimeSessionCreateRequestParam(
                        type="realtime",
                        audio=RealtimeAudioConfigParam(
                            output=RealtimeAudioConfigOutputParam(
                                voice=resolved_voice,
                            ),
                        ),
                    ),
                )
                return f"Voice changed to {resolved_voice}."
            except Exception as e:
                logger.warning("Failed to update live session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        default_voice = get_default_voice()
        voice = self._voice_override or get_session_voice(default=default_voice)
        return self._resolve_backend_voice(voice, source="session voice", fallback=default_voice) or default_voice

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality to the active or next realtime connection."""
        previous_profile = config.REACHY_MINI_CUSTOM_PROFILE
        set_custom_profile(profile)
        try:
            instructions = get_session_instructions(self.instance_path)
            voice = self.get_current_voice()
            core_tools.initialize_tools(force=True)
        except Exception as exc:
            set_custom_profile(previous_profile)
            logger.error("Failed to resolve personality %r: %s", profile, exc)
            return f"Failed to apply personality: {exc}"

        if self.connection is not None:
            try:
                await self.connection.session.update(
                    session=RealtimeSessionCreateRequestParam(
                        type="realtime",
                        instructions=instructions,
                        audio=RealtimeAudioConfigParam(
                            output=RealtimeAudioConfigOutputParam(
                                voice=voice,
                            ),
                        ),
                    ),
                )
                logger.info("Applied personality via live update: %s", profile or "default")
            except Exception as exc:
                logger.warning("Live update failed; will restart session: %s", exc)

            try:
                await self._restart_session()
                return "Applied personality and restarted realtime session."
            except Exception as exc:
                logger.warning("Failed to restart session after apply: %s", exc)
                return "Applied personality. Will take effect on next connection."

        logger.info(
            "Applied personality recorded: %s (no live connection; will apply on next session)",
            profile or "default",
        )
        return "Applied personality. Will take effect on next connection."

    async def _emit_debounced_partial(self, transcript: str, item_id: str, sequence_counter: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)

            input_transcript = self.input_transcript_chunks_by_item
            if input_transcript.item_id == item_id and len(input_transcript.deltas) - 1 == sequence_counter:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a Hugging Face partial transcript snapshot."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        self.client = await self._build_realtime_client()

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except RealtimeSessionSlotsBusy:
                logger.warning(
                    "Realtime websocket rejected (attempt %d/%d): speech-to-speech has no free session slot",
                    attempt,
                    max_attempts,
                )
                if attempt < max_attempts:
                    freed = await self._wait_for_idle_realtime_slot()
                    if not freed:
                        logger.warning(
                            "Speech-to-speech session slot still occupied. "
                            "Stop extra conversation clients or restart speech-to-speech."
                        )
                    self.client = await self._build_realtime_client()
                    continue
                raise
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    self.client = await self._build_realtime_client()
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception as exc:
                    logger.debug("Failed to clear connected event after session exit: %s", exc)

    async def _restart_session(self) -> None:
        """Close the current websocket so the startup loop reconnects once."""
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception as exc:
                logger.warning("Failed to close realtime session for restart: %s", exc)
            finally:
                self.connection = None
        try:
            self._connected_event.clear()
        except Exception as exc:
            logger.debug("Failed to clear connected event: %s", exc)

    async def _safe_response_create(self, *, reason: str = "response", **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        logger.info("Queued response.create reason=%s", reason)
        await self._pending_responses.put((reason, kwargs))

    async def say(self, text: str) -> None:
        """Inject ``text`` as a turn and have the model voice it now.

        Mirrors the startup-greeting path: create a user message item, then
        queue a ``response.create`` through the serial sender. Not verbatim TTS
        (speech-to-speech may rephrase). Raises if the session is closed.
        """
        text = (text or "").strip()
        if not text:
            raise ValueError("say: empty text")
        if not self.connection:
            raise RuntimeError("say: no active session")
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        )
        self._mark_activity("say")
        await self._safe_response_create(reason="say")

    async def _send_startup_greeting_prompt(self) -> None:
        """Prompt the model to open the conversation once the session is ready."""
        if self._startup_greeting_sent or not self.connection:
            return

        greeting_prompt = get_session_greeting_prompt().strip()
        if not greeting_prompt:
            self._startup_greeting_sent = True
            return

        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": greeting_prompt,
                        },
                    ],
                },
            )
            self._startup_greeting_sent = True
            self._mark_activity("startup_greeting_prompt")
            await self._safe_response_create(reason="startup_greeting")
            logger.info("Queued startup greeting prompt")
        except Exception as e:
            logger.warning("Failed to queue startup greeting prompt: %s", e)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits until the receiver observes response.created or a rejection.
        4. Waits for the response cycle to complete (response.done).
        5. If the server rejected with active_response, retries from step 1.
        """
        while self.connection:
            try:
                reason, kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return

            # Parallel tool calls enqueue duplicate empty requests; coalesce to one.
            while not kwargs and not self._pending_responses.empty():
                try:
                    reason, kwargs = self._pending_responses.get_nowait()
                except asyncio.QueueEmpty:
                    break

            sent = False
            max_retries = 5
            attempts = 0
            while not sent and self.connection and attempts < max_retries:
                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break

                self._last_response_rejected = False
                self._response_started_or_rejected_event.clear()
                self._response_done_event.clear()
                self._active_response_reason = reason
                self._active_response_started_at = time.perf_counter()
                self._reset_active_response_audio_state()
                try:
                    logger.info("Sending response.create reason=%s", reason)
                    await self.connection.response.create(**kwargs)
                except Exception as e:
                    logger.debug("_response_sender_loop: send failed reason=%s: %s", reason, e)
                    self._active_response_reason = None
                    self._response_done_event.set()
                    break

                try:
                    await asyncio.wait_for(
                        self._response_started_or_rejected_event.wait(),
                        timeout=_RESPONSE_STARTED_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    attempts += 1
                    logger.warning(
                        "No acknowledgement for response.create; retrying (%d/%d) reason=%s",
                        attempts,
                        max_retries,
                        reason,
                    )
                    self._response_done_event.set()
                    continue

                # Check if the receiver loop observed an asynchronous rejection.
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.debug("response.create rejected %d times; giving up reason=%s", attempts, reason)
                        self._active_response_reason = None
                        break
                    logger.debug(
                        "response.create was rejected; retrying (%d/%d) reason=%s", attempts, max_retries, reason
                    )
                    await asyncio.sleep(_RESPONSE_REJECTION_RETRY_DELAY)
                    continue

                try:
                    await asyncio.wait_for(
                        self._response_done_event.wait(),
                        timeout=_RESPONSE_DONE_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.done; assuming response completed reason=%s", reason)
                    self._response_done_event.set()
                    break

                sent = True

    async def _handle_tool_result(self, completed_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        if completed_tool.error is not None:
            logger.error(
                "Tool '%s' (id=%s) failed with error: %s",
                completed_tool.tool_name,
                completed_tool.id,
                completed_tool.error,
            )
            tool_result = {"error": completed_tool.error}
            tool_result_for_model = tool_result
        elif completed_tool.result is not None:
            tool_result = completed_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(completed_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else tool_result
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                completed_tool.tool_name,
                completed_tool.id,
            )
            logger.debug("Tool '%s' model-visible result: %s", completed_tool.tool_name, tool_result_for_model)
        else:
            logger.warning(
                "Tool '%s' (id=%s) returned no result and no error", completed_tool.tool_name, completed_tool.id
            )
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                completed_tool.tool_name,
                completed_tool.id,
            )
            return

        try:
            send_result_to_model = not completed_tool.is_idle_tool_call
            if send_result_to_model:
                self._mark_activity("tool_result_ready")
            if self._turn_tool_received_at is not None:
                logger.info(
                    "Turn latency: tool %s finished %.0f ms after function call",
                    completed_tool.tool_name,
                    (time.perf_counter() - self._turn_tool_received_at) * 1000,
                )
            model_result_submitted = False
            tool_generation = (
                self._tool_call_generation.pop(completed_tool.id, None) if isinstance(completed_tool.id, str) else None
            )
            result_is_stale = tool_generation is not None and tool_generation != self._turn_generation
            hermes_text = (
                _hermes_result_text(tool_result_for_model) if completed_tool.tool_name == "ask_hermes" else ""
            )
            hermes_request_id = completed_tool.id if isinstance(completed_tool.id, str) else str(uuid.uuid4())
            hermes_originating_turn = tool_generation if tool_generation is not None else self._turn_generation
            hermes_should_buffer = (
                send_result_to_model
                and completed_tool.tool_name == "ask_hermes"
                and bool(hermes_text)
                and (result_is_stale or self._user_speech_in_progress)
            )
            if send_result_to_model and result_is_stale and not hermes_should_buffer:
                logger.warning(
                    "Ignoring stale tool result for '%s' (id=%s); a newer turn is active",
                    completed_tool.tool_name,
                    completed_tool.id,
                )
                send_result_to_model = False
            elif hermes_should_buffer:
                self._buffer_hermes_result(hermes_request_id, hermes_originating_turn, hermes_text)
                send_result_to_model = False
            elif send_result_to_model and completed_tool.tool_name == "ask_hermes" and hermes_text:
                self._track_hermes_result(hermes_request_id, hermes_originating_turn, hermes_text, "delivering")
            if send_result_to_model and isinstance(completed_tool.id, str):
                if not await self._wait_for_response_done_before_tool_result():
                    logger.warning(
                        "response.done missing for tool '%s' (id=%s); sending result anyway",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                if not self.connection:
                    logger.warning(
                        "Connection closed before sending tool '%s' (id=%s) result back",
                        completed_tool.tool_name,
                        completed_tool.id,
                    )
                    return
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": completed_tool.id,
                        "output": json.dumps(tool_result_for_model),
                    },
                )
                model_result_submitted = True

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result_for_model),
                    },
                ),
            )

            if model_result_submitted and completed_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                image_width = tool_result.get("image_width")
                image_height = tool_result.get("image_height")
                jpeg_bytes_value = tool_result.get("jpeg_bytes")
                jpeg_bytes = jpeg_bytes_value if isinstance(jpeg_bytes_value, int) else (len(b64_im) * 3) // 4
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                if isinstance(image_width, int) and isinstance(image_height, int):
                    logger.info(
                        "Added camera image to conversation frame=%sx%s jpeg_bytes=%s",
                        image_width,
                        image_height,
                        jpeg_bytes,
                    )
                else:
                    logger.info(
                        "Added camera image to conversation jpeg_bytes=%s",
                        jpeg_bytes,
                    )

            if isinstance(completed_tool.id, str):
                self._in_flight_tool_calls.discard(completed_tool.id)
                self._turn_device_control_call_ids.discard(completed_tool.id)

            if (
                not result_is_stale
                and completed_tool.tool_name == "home_assistant"
                and is_device_control_success(completed_tool.result, completed_tool.error)
            ):
                await self._play_device_success_emotion()

            tool = core_tools.get_tools().get(completed_tool.tool_name)

            # Always surface errors, skip the spoken follow-up for tools that opt out.
            if model_result_submitted and (
                tool is None or tool.wants_spoken_followup(completed_tool.result, completed_tool.error)
            ):
                self._tool_batch_needs_response = True

            # Parallel tool calls in one turn: respond once every result is in, not per tool.
            if self._tool_batch_needs_response and not self._in_flight_tool_calls:
                self._tool_batch_needs_response = False
                await self._safe_response_create(
                    reason=f"tool_result:{completed_tool.tool_name}",
                    **_TOOL_FOLLOWUP_CREATE_KWARGS,
                )

            if self._pending_hermes_result is not None and self._pending_hermes_result.status == "buffered":
                await self._deliver_buffered_hermes_result()

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        tool_specs = get_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )
        connect_kwargs: dict[str, Any] = {}
        if self._realtime_connect_query:
            connect_kwargs["extra_query"] = self._realtime_connect_query
        async with self.client.realtime.connect(**connect_kwargs) as conn:
            try:
                session_config = self._get_session_config(tool_specs)
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
            except Exception as exc:
                if _is_session_limit_error(exc):
                    raise RealtimeSessionSlotsBusy(str(exc)) from exc
                logger.exception("Realtime session.update failed; aborting startup")
                raise

            logger.info("Realtime session updated successfully")

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Manage events received from the realtime server.
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start the response sender worker
                response_sender_task = asyncio.create_task(self._response_sender_loop(), name="response-sender")
                await self._send_startup_greeting_prompt()

                async for event in self.connection:
                    logger.debug("Realtime event: %s", event.type)
                    if event.type == "input_audio_buffer.speech_started":
                        self._mark_activity("user_speech_started")
                        self._user_speech_in_progress = True
                        self._turn_speech_started_at = time.perf_counter()
                        self._turn_speech_stopped_at = None
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._turn_tool_received_at = None
                        if self._clear_queue and (
                            not self._response_done_event.is_set() or not self.output_queue.empty()
                        ):
                            self._clear_queue()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._mark_activity("user_speech_stopped")
                        self._turn_speech_stopped_at = time.perf_counter()
                        if self._turn_speech_started_at is not None:
                            logger.info(
                                "Turn latency: VAD %.0f ms",
                                (self._turn_speech_stopped_at - self._turn_speech_started_at) * 1000,
                            )
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        self.deps.movement_manager.set_speaking(False)
                        logger.info(
                            "Response audio done reason=%s audio_deltas=%d audio_bytes=%d",
                            self._active_response_reason,
                            self._active_response_audio_delta_count,
                            self._active_response_audio_bytes,
                        )

                    if event.type == "response.output_text.delta":
                        logger.debug("response text delta")

                    if event.type == "response.output_text.done":
                        logger.debug("response text done: %s", event.text)

                    if event.type == "response.created":
                        self._mark_activity("response_created")
                        self.deps.movement_manager.set_speaking(True)
                        self._response_done_event.clear()
                        self._response_started_or_rejected_event.set()
                        if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                            self._turn_response_created_at = time.perf_counter()
                            delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                        logger.info("Response created reason=%s", self._active_response_reason)

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        # Resume tracking for responses that emit no audio (text-only / tool-only).
                        self.deps.movement_manager.set_speaking(False)
                        self._response_done_event.set()
                        self._response_started_or_rejected_event.set()
                        elapsed_ms = (
                            (time.perf_counter() - self._active_response_started_at) * 1000
                            if self._active_response_started_at is not None
                            else None
                        )
                        logger.info(
                            "Response done reason=%s elapsed_ms=%s audio_deltas=%d transcript_seen=%s",
                            self._active_response_reason,
                            f"{elapsed_ms:.0f}" if elapsed_ms is not None else "unknown",
                            self._active_response_audio_delta_count,
                            self._active_response_transcript_seen,
                        )
                        if (
                            self._active_response_reason is not None
                            and self._active_response_reason.startswith("tool_result:")
                            and self._active_response_audio_delta_count == 0
                        ):
                            logger.warning(
                                "Tool follow-up response completed without audio deltas reason=%s transcript_seen=%s",
                                self._active_response_reason,
                                self._active_response_transcript_seen,
                            )
                        await self._handle_hermes_speech_outcome()
                        if self._tool_followup_tools_disabled:
                            await self._set_tool_followup_choice("auto")
                        self._active_response_reason = None
                        self._active_response_started_at = None

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        self._mark_activity("user_transcription_delta")
                        logger.debug(f"User partial transcript: {event.delta}")

                        item_id = event.item_id
                        delta = event.delta or ""

                        input_transcript = self.input_transcript_chunks_by_item
                        self._record_partial_transcript_delta(input_transcript, item_id, delta)

                        current_partial = "".join(input_transcript.deltas)
                        sequence_counter = len(input_transcript.deltas) - 1

                        await self._cancel_partial_transcript_task()

                        # Start new debounce timer with the last delta
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(current_partial, item_id, sequence_counter)
                        )

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        self._mark_activity("user_transcription_completed")
                        raw_transcript = event.transcript or ""
                        transcript = raw_transcript.strip()
                        logger.debug("User transcript: %s", raw_transcript)
                        self.deps.movement_manager.set_listening(False)

                        await self._cancel_partial_transcript_task()

                        if not transcript:
                            logger.debug("Ignoring empty user transcript")
                            self._user_speech_in_progress = False
                            continue

                        self._user_speech_in_progress = False
                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        self._turn_tool_received_at = None
                        if self._turn_speech_stopped_at is not None:
                            logger.info(
                                "Turn latency: STT %.0f ms after speech_stopped",
                                (self._turn_user_done_at - self._turn_speech_stopped_at) * 1000,
                            )
                        self._turn_generation += 1
                        self._in_flight_tool_calls.clear()
                        self._tool_batch_needs_response = False
                        self._reset_device_success_turn_state()
                        self._reset_active_response_audio_state()
                        self._start_fast_ha_command(transcript)

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))
                        self._emit_transcript("user", transcript, True)

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.done":
                        self._mark_activity("assistant_transcript_done")
                        self._active_response_transcript_seen = bool(event.transcript)
                        logger.info(
                            "Assistant transcript done reason=%s transcript=%s",
                            self._active_response_reason,
                            event.transcript,
                        )
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event.transcript})
                        )
                        self._emit_transcript("assistant", event.transcript or "", True)

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        decoded_pcm_bytes = base64.b64decode(event.delta)
                        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
                        self._mark_activity("assistant_audio_delta")
                        self._active_response_audio_delta_count += 1
                        self._active_response_audio_bytes += len(decoded_pcm_bytes)
                        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                            self._turn_first_audio_at = time.perf_counter()
                            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
                        await self.output_queue.put(
                            (
                                self.SAMPLE_RATE,
                                decoded_pcm,
                            ),
                        )
                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, args=%s",
                            tool_name,
                            call_id,
                            args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                args_json_str,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            if self.connection:
                                await self.connection.conversation.item.create(
                                    item={
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": json.dumps({"error": "invalid tool call"}),
                                    },
                                )
                                await self._safe_response_create(reason="invalid_tool_call")
                            continue

                        self._turn_tool_received_at = time.perf_counter()
                        self._tool_call_generation[call_id] = self._turn_generation
                        self._in_flight_tool_calls.add(call_id)
                        if tool_name == "home_assistant" and is_control_action(
                            self._value_from_tool_args(args_json_str, "action")
                        ):
                            self._turn_device_control_call_ids.add(call_id)
                        if tool_name == "play_emotion" and is_success_emotion_request(
                            self._value_from_tool_args(args_json_str, "emotion")
                        ):
                            asyncio.create_task(
                                self._start_or_skip_success_emotion_tool(
                                    call_id, args_json_str, self._tool_call_generation[call_id]
                                ),
                                name=f"success-emotion-{call_id}",
                            )
                            await self.output_queue.put(
                                AdditionalOutputs(
                                    {
                                        "role": "assistant",
                                        "content": (
                                            f"🛠️ Used tool {tool_name} with args {args_json_str}. "
                                            f"The tool is now running. Tool ID: {call_id}"
                                        ),
                                    },
                                ),
                            )
                            continue
                        background_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                            is_idle_tool_call=False,
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {background_tool.tool_id}",
                                },
                            ),
                        )
                        logger.info(
                            "Started background tool: %s (id=%s, call_id=%s)",
                            tool_name,
                            background_tool.tool_id,
                            call_id,
                        )
                        if tool_name == "ask_hermes":
                            logger.info(
                                "Hermes request started request_id=%s turn=%s",
                                call_id,
                                self._turn_generation,
                            )
                        if tool_name in _LONG_RUNNING_TOOLS:
                            asyncio.create_task(
                                self._acknowledge_long_running_tool(
                                    tool_name, call_id, self._tool_call_generation[call_id]
                                ),
                                name=f"tool-hold-{call_id}",
                            )

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "") or getattr(err, "type", "")

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            self._last_response_rejected = True
                            self._response_started_or_rejected_event.set()
                            logger.debug("response.create rejected; worker will retry after active response finishes")
                        else:
                            self._response_started_or_rejected_event.set()
                            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                        if code == "input_audio_buffer_commit_empty":
                            self.deps.movement_manager.set_listening(False)

                        # Only show user-facing errors, not internal state errors.
                        if code not in (
                            "input_audio_buffer_commit_empty",
                            "conversation_already_has_active_response",
                        ):
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                            )
            finally:
                # Stop the response sender worker.
                if response_sender_task is not None:
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass

                # Stop background tool manager tasks (listener + cleanup) in all paths.
                await self.tool_manager.shutdown()

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the realtime server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for the realtime API.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        _, audio_frame = frame
        if audio_frame.size == 0:
            return

        # Reshape if needed
        if audio_frame.ndim == 2:
            # channels-last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to the realtime input buffer (guard against races during reconnect).
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def _wait_for_idle_realtime_slot(self) -> bool:
        """Poll speech-to-speech `/v1/pool` until a unit is idle, or give up."""
        direct_realtime_url = get_hf_direct_ws_url()
        if not direct_realtime_url:
            await asyncio.sleep(_SESSION_SLOT_WAIT_S)
            return False
        pool_url = f"{parse_hf_realtime_url(direct_realtime_url).base_url.rstrip('/')}/pool"
        deadline = time.monotonic() + _SESSION_SLOT_WAIT_S
        logged_wait = False
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=2.0) as http_client:
                    response = await http_client.get(pool_url)
                    response.raise_for_status()
                    payload = response.json()
            except Exception as exc:
                logger.debug("Could not read realtime pool status: %s", exc)
                await asyncio.sleep(_SESSION_SLOT_POLL_S)
                continue
            if not isinstance(payload, dict):
                await asyncio.sleep(_SESSION_SLOT_POLL_S)
                continue
            if _realtime_pool_is_stuck(payload):
                logger.warning("Speech-to-speech realtime slot is stuck; restart speech-to-speech to free it")
                return False
            if _realtime_pool_has_idle_slot(payload):
                return True
            if not logged_wait:
                logger.info("Waiting for a free speech-to-speech realtime session slot")
                logged_wait = True
            await asyncio.sleep(_SESSION_SLOT_POLL_S)
        return False

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Unblock the response sender worker so it can exit
        self._response_done_event.set()

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        await self._cancel_partial_transcript_task()

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def get_available_voices(self) -> list[str]:
        """Return the available Hugging Face voices."""
        return get_available_voices()

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the Hugging Face OpenAI-compatible realtime client."""
        configured_bearer_token = (config.HF_TOKEN or "").strip()
        connection_selection = get_hf_connection_selection()
        direct_realtime_url = get_hf_direct_ws_url()
        if connection_selection.mode == HF_LOCAL_CONNECTION_MODE:
            if not direct_realtime_url:
                raise RuntimeError("HF_REALTIME_WS_URL must be set when HF_REALTIME_CONNECTION_MODE=local")
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                configured_bearer_token,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct Hugging Face realtime endpoint %s", direct_realtime_url)
            return client

        session_url = connection_selection.session_url
        if not session_url:
            raise RuntimeError("Built-in Hugging Face session proxy URL is unavailable")
        if direct_realtime_url:
            logger.info("HF_REALTIME_CONNECTION_MODE=deployed; ignoring HF_REALTIME_WS_URL.")

        bearer_token = configured_bearer_token or (get_token() or "").strip()
        allocator_headers = {"User-Agent": "reachy-mini-conversation-app"}
        if bearer_token:
            allocator_headers["X-Reachy-Mini-Authorization"] = f"Bearer {bearer_token}"
        allocator_payload: dict[str, str] = {}
        try:
            hardware_id = self.deps.reachy_mini.client.get_status(wait=False).hardware_id
        except (AssertionError, ConnectionError, TimeoutError) as e:
            logger.warning("Daemon status unavailable for realtime session allocation: %s", e)
        else:
            if hardware_id:
                allocator_payload["hardware_id"] = hardware_id

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(session_url, headers=allocator_headers, json=allocator_payload)
            response.raise_for_status()
            payload = response.json()

        connect_url = payload.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError(f"Session allocator response did not contain a valid connect_url: {payload!r}")

        parsed_connect_url = parse_hf_realtime_url(connect_url)
        if not parsed_connect_url.has_realtime_path:
            raise ValueError(f"Expected realtime connect URL ending with /realtime, got: {connect_url}")

        logger.info("Allocated realtime session %s", payload.get("session_id") or "<unknown>")
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            connect_url,
            bearer_token,
        )
        self._realtime_connect_query = connect_query
        return client
