"""Tests for the Hermes Gateway HTTP client."""

import asyncio

import httpx
import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HERMES_CHAT_MODEL,
    HERMES_SESSION_HEADER,
    HERMES_VOICE_SYSTEM_PROMPT,
    HERMES_REEF_TREND_INSTRUCTION,
    HermesRequestError,
    HermesTimeoutError,
    HermesNotConfiguredError,
    hermes_is_busy,
    is_trend_query,
    send_to_hermes,
    is_process_narration,
    get_hermes_session_id,
)


GATEWAY_URL = "http://127.0.0.1:8642/v1/chat/completions"
API_KEY = "test-hermes-key"


@pytest.fixture(autouse=True)
def _fresh_hermes_lock() -> None:
    """Each test gets its own lock so pytest-asyncio's per-test loops do not collide."""
    hermes_client._HERMES_REQUEST_LOCK = asyncio.Lock()


def _configure_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hermes_client.config, "HERMES_GATEWAY_URL", GATEWAY_URL)
    monkeypatch.setattr(hermes_client.config, "HERMES_API_KEY", API_KEY)


def _fake_async_client(
    *,
    response: httpx.Response | None = None,
    error: BaseException | None = None,
    posts: list[tuple[str, dict[str, str] | None, object | None]] | None = None,
) -> type:
    class FakeAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> httpx.Response:
            if posts is not None:
                posts.append((url, headers, json))
            if error is not None:
                raise error
            assert response is not None
            return response

    return FakeAsyncClient


def _json_response(status_code: int, payload: object | None = None, text: str | None = None) -> httpx.Response:
    request = httpx.Request("POST", GATEWAY_URL)
    if text is not None:
        return httpx.Response(status_code, text=text, request=request)
    return httpx.Response(status_code, json=payload, request=request)


def _completion_payload(content: str) -> dict[str, object]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": HERMES_CHAT_MODEL,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
    }


@pytest.mark.asyncio
async def test_send_to_hermes_posts_chat_completion_and_returns_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful chat.completion yields assistant content and uses bearer plus session headers."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(
            response=_json_response(200, _completion_payload("Reef temperature is 26.1 C.")), posts=posts
        ),
    )

    reply = await send_to_hermes("what's the reef temperature", "session-abc")

    assert reply == "Reef temperature is 26.1 C."
    assert posts == [
        (
            GATEWAY_URL,
            {
                "Authorization": f"Bearer {API_KEY}",
                "Accept": "application/json",
                HERMES_SESSION_HEADER: "session-abc",
            },
            {
                "model": HERMES_CHAT_MODEL,
                "messages": [
                    {"role": "system", "content": HERMES_VOICE_SYSTEM_PROMPT},
                    {"role": "user", "content": "what's the reef temperature"},
                ],
            },
        )
    ]


@pytest.mark.asyncio
async def test_send_to_hermes_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hung gateway becomes HermesTimeoutError instead of blocking."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(error=httpx.TimeoutException("timed out")),
    )

    with pytest.raises(HermesTimeoutError, match="timed out"):
        await send_to_hermes("when's the next 311", "session-abc")


@pytest.mark.asyncio
async def test_send_to_hermes_queues_overlapping_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second Hermes task waits for the in-flight request instead of reporting busy."""
    _configure_gateway(monkeypatch)
    first_started = asyncio.Event()
    posts = 0

    class SerialAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "SerialAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> httpx.Response:
            nonlocal posts
            posts += 1
            first_started.set()
            await asyncio.sleep(0.05)
            return _json_response(200, _completion_payload("ok"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", SerialAsyncClient)

    first = asyncio.create_task(send_to_hermes("reef tank trends", "session-abc"))
    await first_started.wait()
    second = asyncio.create_task(send_to_hermes("next 311 bus", "session-abc"))
    assert await first == "ok"
    assert await second == "ok"
    assert posts == 2


@pytest.mark.asyncio
async def test_send_to_hermes_queues_duplicate_in_flight_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second identical query waits for the first instead of failing busy."""
    _configure_gateway(monkeypatch)
    first_started = asyncio.Event()
    posts = 0

    class SlowAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "SlowAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> httpx.Response:
            nonlocal posts
            posts += 1
            first_started.set()
            await asyncio.sleep(0.05)
            return _json_response(200, _completion_payload("ok"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", SlowAsyncClient)

    first = asyncio.create_task(send_to_hermes("reef tank trends", "session-abc"))
    await first_started.wait()
    second = asyncio.create_task(send_to_hermes("reef tank trends", "session-abc"))
    assert await first == "ok"
    assert await second == "ok"
    assert posts == 2


@pytest.mark.asyncio
async def test_send_to_hermes_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-success HTTP status codes are surfaced as HermesRequestError."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(503, {"error": "unavailable"})),
    )

    with pytest.raises(HermesRequestError, match="HTTP 503"):
        await send_to_hermes("turn on the tank lights", "session-abc")


@pytest.mark.asyncio
async def test_send_to_hermes_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body is a request error, not a crash."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, text="not-json")),
    )

    with pytest.raises(HermesRequestError, match="malformed JSON"):
        await send_to_hermes("reef status", "session-abc")


@pytest.mark.asyncio
async def test_send_to_hermes_missing_reply_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """JSON without assistant content is rejected."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, {"message": "oops"})),
    )

    with pytest.raises(HermesRequestError, match="missing choices"):
        await send_to_hermes("reef status", "session-abc")


@pytest.mark.asyncio
async def test_send_to_hermes_requires_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing URL or key fails before any network call."""
    monkeypatch.setattr(hermes_client.config, "HERMES_GATEWAY_URL", "")
    monkeypatch.setattr(hermes_client.config, "HERMES_API_KEY", "")
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", _fake_async_client(posts=posts))

    with pytest.raises(HermesNotConfiguredError):
        await send_to_hermes("hello", "session-abc")
    assert posts == []


def test_get_hermes_session_id_is_stable_for_the_process() -> None:
    """Hermes session context stays on one id for the app lifetime."""
    first = get_hermes_session_id()
    second = get_hermes_session_id()
    assert first == second
    assert first


@pytest.mark.asyncio
async def test_hermes_is_busy_tracks_the_request_lock() -> None:
    """Busy is true only while the gateway lock is held."""
    assert hermes_is_busy() is False
    async with hermes_client._HERMES_REQUEST_LOCK:
        assert hermes_is_busy() is True
    assert hermes_is_busy() is False


def test_is_process_narration_detects_file_access_failures() -> None:
    """Hermes file/tool narration is rejected before it can be spoken as a trend."""
    assert is_process_narration("It seems there is an issue with accessing the file content.") is True
    assert is_process_narration("I need to inspect the file before I can answer.") is True
    assert is_process_narration("I couldn't access the file") is True
    assert is_process_narration("Nitrate has been falling from 20 to 8, while temperature stayed near 26.") is False


def test_is_trend_query_matches_reef_history_phrases() -> None:
    """Trend routing stays on Hermes; live snapshot phrases do not match."""
    assert is_trend_query("how is my reef tank trending?") is True
    assert is_trend_query("parameter history") is True
    assert is_trend_query("what's the reef temperature") is False


@pytest.mark.asyncio
async def test_send_to_hermes_appends_trend_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reef-trend questions ask Hermes for an answer, not process narration."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, _completion_payload("Nitrate is falling.")), posts=posts),
    )

    reply = await send_to_hermes("how is my reef tank trending?", "session-abc")

    assert reply == "Nitrate is falling."
    body = posts[0][2]
    assert isinstance(body, dict)
    messages = body["messages"]
    assert isinstance(messages, list)
    user_message = messages[1]
    assert isinstance(user_message, dict)
    assert "how is my reef tank trending?" in str(user_message["content"])
    assert HERMES_REEF_TREND_INSTRUCTION in str(user_message["content"])
    assert "Do not narrate files" in HERMES_REEF_TREND_INSTRUCTION
