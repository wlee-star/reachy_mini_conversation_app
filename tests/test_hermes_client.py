"""Tests for the Hermes Gateway HTTP client."""

import asyncio
from pathlib import Path

import httpx
import pytest

from reachy_mini_conversation_app import hermes_client
from reachy_mini_conversation_app.hermes_client import (
    HERMES_CHAT_MODEL,
    HERMES_SESSION_HEADER,
    HERMES_VOICE_SYSTEM_PROMPT,
    HERMES_REEF_TREND_INSTRUCTION,
    HERMES_REEF_VOICE_SYSTEM_PROMPT,
    HermesRequestError,
    HermesTimeoutError,
    HermesCircuitOpenError,
    HermesNotConfiguredError,
    hermes_is_busy,
    is_trend_query,
    send_to_hermes,
    chat_completions_url,
    hermes_circuit_state,
    is_process_narration,
    parse_reef_timestamp,
    reset_hermes_circuit,
    get_hermes_session_id,
    reef_cache_age_seconds,
    load_latest_reef_thread,
    hermes_request_timeout_s,
    reef_cache_status_for_age,
    hermes_in_flight_request_id,
)


GATEWAY_URL = "http://127.0.0.1:8642/v1/chat/completions"
API_KEY = "test-hermes-key"


@pytest.fixture(autouse=True)
def _fresh_hermes_lock() -> None:
    """Each test gets its own lock so pytest-asyncio's per-test loops do not collide."""
    hermes_client._HERMES_REQUEST_LOCK = asyncio.Lock()
    hermes_client._HERMES_IN_FLIGHT_REQUEST_ID = None
    reset_hermes_circuit()


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
async def test_send_to_hermes_hard_timeout_releases_lock(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A hung POST that never raises httpx.TimeoutException is still cancelled."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 0.05)

    class HungAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "HungAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> httpx.Response:
            del url, headers, json
            await asyncio.sleep(5)
            return _json_response(200, _completion_payload("too late"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", HungAsyncClient)

    with caplog.at_level("INFO"):
        with pytest.raises(HermesTimeoutError, match="timed out"):
            await send_to_hermes("reef tank trends", "session-abc")
    assert hermes_is_busy() is False
    assert hermes_in_flight_request_id() is None
    assert "cancelling timed-out request" in caplog.text
    assert "request cleanup complete" in caplog.text
    assert "Reef request timed out" in caplog.text


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
    assert is_process_narration("I'm having trouble accessing the reef data. Let me try again!") is True
    assert is_process_narration("Nitrate has been falling from 20 to 8, while temperature stayed near 26.") is False


def test_is_trend_query_matches_reef_history_phrases() -> None:
    """Trend routing stays on Hermes; live snapshot phrases do not match."""
    assert is_trend_query("how is my reef tank trending?") is True
    assert is_trend_query("parameter history") is True
    assert is_trend_query("How has my reef tank changed over the last 6 hours?") is True
    assert is_trend_query("How is my ATO trending?") is True
    assert is_trend_query("How much ATO have I been using?") is True
    assert is_trend_query("Give me a reef tank report.") is True
    assert is_trend_query("Give me a report on my reef tank") is True
    assert is_trend_query("Analyse my reef tank.") is True
    assert is_trend_query("How has my reef tank been doing?") is True
    assert is_trend_query("Are my reef parameters improving?") is True
    assert is_trend_query("Give me a trending report") is True
    assert is_trend_query("Can you give me a reef trending report?") is True
    assert is_trend_query("what's the reef temperature") is False
    assert is_trend_query("What is the current pH?") is False
    assert is_trend_query("Give me a weather report") is False


def test_chat_completions_url_normalizes_host_port() -> None:
    """A host:port gateway URL is posted to the chat-completions path, not /."""
    assert chat_completions_url("http://127.0.0.1:8642") == "http://127.0.0.1:8642/v1/chat/completions"
    assert chat_completions_url("http://127.0.0.1:8642/") == "http://127.0.0.1:8642/v1/chat/completions"
    assert (
        chat_completions_url("http://127.0.0.1:8642/v1/chat/completions")
        == "http://127.0.0.1:8642/v1/chat/completions"
    )


def test_load_latest_reef_thread_reads_summary_and_slopes(tmp_path: Path) -> None:
    """The Reefy thread cache is the history source Hermes should report from."""
    thread = tmp_path / "reef_thread.jsonl"
    thread.write_text(
        '{"type":"run","ts":"2026-08-31T04:00:04Z","cache_ts":"2026-08-31T03:59:03Z",'
        '"summary":"Reef stable - temp 24.0C (-0.012/6h); ATO 2.9 (~204h until refill).",'
        '"trends":{"Tmp":{"trend_6h":-0.012,"trend_str":"-0.012/6h"},'
        '"LLSATO":{"trend_6h":-0.071,"trend_str":"-0.071/6h"}},'
        '"handoff":{"for_reachy":{"ask_first":true,"source":"hermes"}},'
        '"ato_hours_until_low":204.0}\n',
        encoding="utf-8",
    )
    snapshot = load_latest_reef_thread(str(thread))
    assert snapshot is not None
    assert snapshot["report"].startswith("Reef stable")
    assert snapshot["ato_hours_until_low"] == 204.0
    assert snapshot["source"] == "hermes"
    assert isinstance(snapshot["cache_age_seconds"], float)
    handoff = snapshot["handoff"]
    assert isinstance(handoff, dict)
    for_reachy = handoff["for_reachy"]
    assert isinstance(for_reachy, dict)
    assert for_reachy["source"] == "hermes"
    assert for_reachy["ask_first"] is True
    trends = snapshot["trends"]
    assert isinstance(trends, dict)
    assert trends["Tmp"]["trend_str"] == "-0.012/6h"
    assert trends["LLSATO"]["trend_6h"] == -0.071


@pytest.mark.asyncio
async def test_send_to_hermes_posts_to_chat_completions_when_url_is_host_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ask_hermes must not POST to the gateway root, which returns HTTP 404."""
    monkeypatch.setattr(hermes_client.config, "HERMES_GATEWAY_URL", "http://127.0.0.1:8642")
    monkeypatch.setattr(hermes_client.config, "HERMES_API_KEY", API_KEY)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, _completion_payload("ok")), posts=posts),
    )

    reply = await send_to_hermes("hello", "session-abc")

    assert reply == "ok"
    assert posts[0][0] == "http://127.0.0.1:8642/v1/chat/completions"


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
    system_message = messages[0]
    user_message = messages[1]
    assert isinstance(system_message, dict)
    assert isinstance(user_message, dict)
    assert system_message["content"] == HERMES_REEF_VOICE_SYSTEM_PROMPT
    assert "how is my reef tank trending?" in str(user_message["content"])
    assert HERMES_REEF_TREND_INSTRUCTION in str(user_message["content"])
    assert "Do not narrate files" in HERMES_REEF_TREND_INSTRUCTION


def test_parse_reef_timestamp_accepts_z_suffix() -> None:
    """Reefy JSONL timestamps use a trailing Z."""
    parsed = parse_reef_timestamp("2026-08-31T04:00:04Z")
    assert parsed is not None
    assert parsed.year == 2026
    assert parsed.month == 8
    assert parse_reef_timestamp("not-a-date") is None


def test_reef_cache_age_prefers_report_timestamp(tmp_path: Path) -> None:
    """Cache age uses the report timestamp before file mtime."""
    thread = tmp_path / "reef_thread.jsonl"
    thread.write_text("x", encoding="utf-8")
    age = reef_cache_age_seconds(generated_at="2026-09-02T10:00:00Z", path=str(thread))
    assert age is not None
    assert age >= 0


def test_reef_cache_status_for_age_never_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache fallbacks are degraded or stale, never success."""
    monkeypatch.setattr(hermes_client.config, "REEF_CACHE_MAX_AGE_SECONDS", 3600)
    assert reef_cache_status_for_age(120.0) == "degraded"
    assert reef_cache_status_for_age(7200.0) == "stale"
    assert reef_cache_status_for_age(None) == "stale"


def test_hermes_request_timeout_is_shorter_for_reef(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive Reef waits use the 15s cap, not the 180s delegated-task timeout."""
    monkeypatch.setattr(hermes_client.config, "HERMES_REQUEST_TIMEOUT_SECONDS", 180)
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 15)
    assert hermes_request_timeout_s(history_request=True) == 15
    assert hermes_request_timeout_s(history_request=False) == 180


@pytest.mark.asyncio
async def test_send_to_hermes_uses_reef_timeout_for_trend_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reef trend POSTs use the interactive timeout on the HTTP client."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 15)
    seen: list[object] = []

    class RecordingAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            seen.append(kwargs.get("timeout"))

        async def __aenter__(self) -> "RecordingAsyncClient":
            return self

        async def __aexit__(self, *_args: object) -> bool:
            return False

        async def post(
            self,
            url: str,
            headers: dict[str, str] | None = None,
            json: object | None = None,
        ) -> httpx.Response:
            del url, headers, json
            return _json_response(200, _completion_payload("ok"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", RecordingAsyncClient)
    await send_to_hermes("Can you tell me what my reef tank is trending at?", "session-abc")
    assert seen == [15]


@pytest.mark.asyncio
async def test_send_to_hermes_timeout_allows_a_following_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a timed-out Reef request, the lock is clear and the next call can run."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(hermes_client.config, "HERMES_REEF_REQUEST_TIMEOUT_SECONDS", 0.05)
    posts = 0

    class FirstHungThenOkClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "FirstHungThenOkClient":
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
            del url, headers, json
            if posts == 1:
                await asyncio.sleep(5)
            return _json_response(200, _completion_payload("second succeeded"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", FirstHungThenOkClient)

    with pytest.raises(HermesTimeoutError):
        await send_to_hermes("reef tank trends", "session-abc")
    assert hermes_is_busy() is False
    assert hermes_in_flight_request_id() is None

    reply = await send_to_hermes("next 311 bus", "session-abc")
    assert reply == "second succeeded"
    assert posts == 2
    assert hermes_is_busy() is False


@pytest.mark.asyncio
async def test_send_to_hermes_retries_http_5xx_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 5xx is retried once, then classified as HERMES_HTTP_ERROR."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(503, {"error": "unavailable"}), posts=posts),
    )

    with pytest.raises(HermesRequestError, match="HTTP 503") as exc_info:
        await send_to_hermes("turn on the tank lights", "session-abc")
    assert exc_info.value.category == "HERMES_HTTP_ERROR"
    assert len(posts) == 2


@pytest.mark.asyncio
async def test_send_to_hermes_does_not_retry_http_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 4xx fails immediately without a second POST."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(400, {"error": "bad request"}), posts=posts),
    )

    with pytest.raises(HermesRequestError, match="HTTP 400") as exc_info:
        await send_to_hermes("turn on the tank lights", "session-abc")
    assert exc_info.value.category == "HERMES_HTTP_ERROR"
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_send_to_hermes_retries_connection_error_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection failure is retried once, then the second attempt can succeed."""
    _configure_gateway(monkeypatch)
    posts = 0

    class FlakyAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "FlakyAsyncClient":
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
            del url, headers, json
            if posts == 1:
                raise httpx.ConnectError("connection refused", request=httpx.Request("POST", GATEWAY_URL))
            return _json_response(200, _completion_payload("recovered"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", FlakyAsyncClient)

    reply = await send_to_hermes("next 311 bus", "session-abc")
    assert reply == "recovered"
    assert posts == 2


@pytest.mark.asyncio
async def test_send_to_hermes_empty_reply_is_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP 200 with an empty assistant string is HERMES_INVALID_RESPONSE."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, _completion_payload("   "))),
    )

    with pytest.raises(HermesRequestError, match="missing a non-empty reply") as exc_info:
        await send_to_hermes("reef status", "session-abc")
    assert exc_info.value.category == "HERMES_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_send_to_hermes_does_not_retry_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout is not retried; that would double the live wait."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(error=httpx.TimeoutException("timed out"), posts=posts),
    )

    with pytest.raises(HermesTimeoutError) as exc_info:
        await send_to_hermes("when's the next 311", "session-abc")
    assert exc_info.value.category == "HERMES_TIMEOUT"
    assert len(posts) == 1


@pytest.mark.asyncio
async def test_hermes_circuit_opens_after_repeated_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two consecutive timeouts open the circuit so the next call fails fast."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(hermes_client.config, "HERMES_CIRCUIT_FAILURE_THRESHOLD", 2)
    monkeypatch.setattr(hermes_client.config, "HERMES_CIRCUIT_COOLDOWN_SECONDS", 60)
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(error=httpx.TimeoutException("timed out")),
    )

    with pytest.raises(HermesTimeoutError):
        await send_to_hermes("reef tank trends", "session-abc")
    with pytest.raises(HermesTimeoutError):
        await send_to_hermes("reef tank trends", "session-abc")
    assert hermes_circuit_state() == "open"

    with pytest.raises(HermesCircuitOpenError) as exc_info:
        await send_to_hermes("reef tank trends", "session-abc")
    assert exc_info.value.category == "HERMES_CIRCUIT_OPEN"
    assert hermes_is_busy() is False


@pytest.mark.asyncio
async def test_hermes_circuit_recovers_after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """After cooldown, one successful probe closes the circuit."""
    _configure_gateway(monkeypatch)
    monkeypatch.setattr(hermes_client.config, "HERMES_CIRCUIT_FAILURE_THRESHOLD", 1)
    monkeypatch.setattr(hermes_client.config, "HERMES_CIRCUIT_COOLDOWN_SECONDS", 0)
    posts = 0

    class RecoveringAsyncClient:
        def __init__(self, **kwargs: object) -> None:
            self.timeout = kwargs.get("timeout")

        async def __aenter__(self) -> "RecoveringAsyncClient":
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
            del url, headers, json
            if posts == 1:
                raise httpx.TimeoutException("timed out")
            return _json_response(200, _completion_payload("back online"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", RecoveringAsyncClient)

    with pytest.raises(HermesTimeoutError):
        await send_to_hermes("reef tank trends", "session-abc")
    assert hermes_circuit_state() == "open"

    reply = await send_to_hermes("reef tank trends", "session-abc")
    assert reply == "back online"
    assert hermes_circuit_state() == "closed"
    assert posts == 2


@pytest.mark.asyncio
async def test_five_consecutive_hermes_successes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five successful Hermes calls stay closed and return content."""
    _configure_gateway(monkeypatch)
    posts: list[tuple[str, dict[str, str] | None, object | None]] = []
    monkeypatch.setattr(
        hermes_client.httpx,
        "AsyncClient",
        _fake_async_client(response=_json_response(200, _completion_payload("ok")), posts=posts),
    )

    for index in range(5):
        reply = await send_to_hermes(f"reef check {index}", "session-abc")
        assert reply == "ok"
    assert len(posts) == 5
    assert hermes_circuit_state() == "closed"


@pytest.mark.asyncio
async def test_send_to_hermes_cancelled_releases_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling an in-flight request cleans up and does not retry."""
    _configure_gateway(monkeypatch)
    started = asyncio.Event()
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
            del url, headers, json
            started.set()
            await asyncio.sleep(5)
            return _json_response(200, _completion_payload("too late"))

    monkeypatch.setattr(hermes_client.httpx, "AsyncClient", SlowAsyncClient)

    task = asyncio.create_task(send_to_hermes("next 311 bus", "session-abc"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert posts == 1
    assert hermes_is_busy() is False
    assert hermes_in_flight_request_id() is None
    assert hermes_circuit_state() == "closed"
