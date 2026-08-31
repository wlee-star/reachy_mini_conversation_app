import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.tools import reef_status as reef_status_mod
from reachy_mini_conversation_app.tools.apex import (
    Apex,
    match_apex_intent,
    spoken_apex_update,
    classify_reef_intent,
    spoken_reef_source_answer,
    match_reef_source_question,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


def _write_cache(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "probes": {
                    "temperature": {"value": 26.1, "type": "temperature", "status": "ok"},
                    "ph": {"value": 8.1, "type": "ph", "status": "ok"},
                },
                "ato": {"level": "normal"},
                "alarms": {"leak": "off"},
                "alerts": ["skimmer cup soon"],
                "controller": "apex",
                "cached_at": "2026-08-24T10:00:00Z",
                "age_seconds": 12,
                "stale": False,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(autouse=True)
def _disable_live_apex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reef_status_mod.config, "APEX_STATUS_URL", None)


@pytest.mark.asyncio
async def test_apex_status_reads_existing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Apex status returns the current reef cache snapshot."""
    cache_path = tmp_path / "reef_cache.json"
    _write_cache(cache_path)
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(cache_path))

    result = await Apex()(_deps(), action="get_apex_status")

    assert result["source"] == "reef_cache_direct"
    assert result["apex_status"]["controller"] == "apex"
    assert result["apex_status"]["water_parameters"]["temperature"]["value"] == 26.1
    assert result["apex_status"]["equipment"]["ato"] == {"level": "normal"}


@pytest.mark.asyncio
async def test_water_parameters_can_be_filtered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Water parameter reads can be narrowed to requested probes."""
    cache_path = tmp_path / "reef_cache.json"
    _write_cache(cache_path)
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(cache_path))

    result = await Apex()(_deps(), action="get_water_parameters", include=["ph"])

    assert result["water_parameters"] == {"ph": {"value": 8.1, "type": "ph", "status": "ok"}}
    assert result["age_seconds"] == 12


@pytest.mark.asyncio
async def test_equipment_status_returns_cache_equipment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Equipment status exposes local cache equipment fields."""
    cache_path = tmp_path / "reef_cache.json"
    _write_cache(cache_path)
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(cache_path))

    result = await Apex()(_deps(), action="get_equipment_status")

    assert result["equipment_status"] == {
        "ato": {"level": "normal"},
        "controller": "apex",
        "outlets": [],
    }
    assert result["source"] == "reef_cache_direct"


@pytest.mark.asyncio
async def test_alerts_returns_alerts_and_alarms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Alert reads return both alert and alarm fields."""
    cache_path = tmp_path / "reef_cache.json"
    _write_cache(cache_path)
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(cache_path))

    result = await Apex()(_deps(), action="get_alerts")

    assert result["alerts"] == ["skimmer cup soon"]
    assert result["alarms"] == {"leak": "off"}


@pytest.mark.asyncio
async def test_apex_reports_missing_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing reef cache is a concise tool error."""
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(tmp_path / "missing.json"))

    result = await Apex()(_deps(), action="get_apex_status")

    assert result == {"error": "Reef cache not found. Ensure reef_cache.py cron is running."}


@pytest.mark.asyncio
async def test_apex_reports_malformed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed reef cache is a concise tool error."""
    cache_path = tmp_path / "reef_cache.json"
    cache_path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(cache_path))

    result = await Apex()(_deps(), action="get_apex_status")

    assert result == {"error": "Apex reef cache could not be read."}


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "http://apex.local/status")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> object:
        return self._payload


class _FakeAsyncClient:
    requested_urls: list[str] = []
    response = _FakeResponse(200, {})

    def __init__(self, *args: object, **kwargs: object) -> None:
        return None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def get(self, url: str) -> _FakeResponse:
        self.__class__.requested_urls.append(url)
        return self.__class__.response


@pytest.mark.asyncio
async def test_apex_reads_live_status_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Apex prefers the live /status JSON when APEX_STATUS_URL is set."""
    monkeypatch.setattr(reef_status_mod.config, "APEX_STATUS_URL", "http://192.168.0.143:8080/status")
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(tmp_path / "missing.json"))
    _FakeAsyncClient.requested_urls = []
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "controller": {"hostname": "Cade_S3_1200_P", "date": "08/24/2026 05:24:01"},
            "probes": [{"name": "Tmp", "value": 24.2, "type": "Temp"}],
            "outlets": [{"name": "VarSpd1_I1", "state": "PF1"}],
        },
    )
    monkeypatch.setattr(reef_status_mod.httpx, "AsyncClient", _FakeAsyncClient)

    result = await Apex()(_deps(), action="get_apex_status")

    assert _FakeAsyncClient.requested_urls == ["http://192.168.0.143:8080/status"]
    assert result["source"] == "apex_status_http"
    assert result["apex_status"]["controller"] == "Cade_S3_1200_P"
    assert result["apex_status"]["water_parameters"]["Tmp"]["value"] == 24.2
    assert result["apex_status"]["equipment"]["outlets"][0]["name"] == "VarSpd1_I1"


def _live_user_payload() -> dict[str, object]:
    return {
        "probes": [
            {"name": "Tmp", "value": 24.0, "type": "Temp"},
            {"name": "pH", "value": 5.94, "type": "pH"},
            {"name": "ORP", "value": 39.0, "type": "ORP"},
            {"name": "FS100", "value": 2207.0, "type": None},
            {"name": "LLSATO", "value": 2.9, "type": None},
        ],
        "fetched_at": "2026-08-31T03:22:00.869566Z",
    }


@pytest.mark.asyncio
async def test_apex_keeps_raw_llsato_from_live_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Live LLSATO 2.9 is returned as 2.9, not converted to a percentage."""
    monkeypatch.setattr(reef_status_mod.config, "APEX_STATUS_URL", "http://192.168.0.143:8080/status")
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(tmp_path / "missing.json"))
    _FakeAsyncClient.requested_urls = []
    _FakeAsyncClient.response = _FakeResponse(200, _live_user_payload())
    monkeypatch.setattr(reef_status_mod.httpx, "AsyncClient", _FakeAsyncClient)

    result = await Apex()(_deps(), action="get_apex_status")

    assert _FakeAsyncClient.requested_urls == ["http://192.168.0.143:8080/status"]
    assert result["source"] == "apex_status_http"
    assert result["apex_status"]["cached_at"] == "2026-08-31T03:22:00.869566Z"
    assert result["apex_status"]["water_parameters"]["LLSATO"]["value"] == 2.9
    assert result["apex_status"]["equipment"]["ato"]["llsato"] == 2.9
    spoken = spoken_apex_update(result, "llsato")
    assert spoken == "LLSATO is 2.9."
    assert "85" not in spoken
    assert "%" not in spoken
    assert "salinity" not in spoken.lower()


@pytest.mark.parametrize(
    ("transcript", "metric"),
    [
        ("What is my reef tank temperature?", "tmp"),
        ("What is my reef tank pH?", "ph"),
        ("What is my LLSATO value?", "llsato"),
        ("what's the L L S A T O level value?", "llsato"),
        ("What is my ATO level?", "ato"),
        ("How much ATO do I have?", "ato"),
        ("What's the status of my reef tank?", "status"),
        ("What's the temperature?", "tmp"),
        ("What's the pH?", "ph"),
    ],
)
def test_match_apex_intent_routes_live_reef_questions(transcript: str, metric: str) -> None:
    """Live reef questions are intercepted before the LLM can invent numbers."""
    intent = match_apex_intent(transcript)
    assert intent is not None
    assert intent.metric == metric


def test_match_apex_intent_leaves_trends_to_hermes() -> None:
    """Historical trend questions are not handled by the live Apex fast-path."""
    assert match_apex_intent("how is my reef tank trending?") is None
    assert match_apex_intent("ATO history") is None
    assert match_apex_intent("How has my reef tank changed over the last 6 hours?") is None
    assert match_apex_intent("How is my ATO trending?") is None
    assert match_apex_intent("How much ATO have I been using?") is None
    assert match_apex_intent("Give me a reef tank report.") is None
    assert match_apex_intent("Give me a report on the status of my reef tank.") is None
    assert match_apex_intent("Analyse my reef tank.") is None
    assert match_apex_intent("How has my reef tank been doing?") is None
    assert match_apex_intent("Are my reef parameters improving or getting worse?") is None


@pytest.mark.parametrize(
    ("transcript", "intent", "route", "metric"),
    [
        ("What are my reef tank stats?", "current_stats", "local", "status"),
        ("What is my reef tank status?", "current_stats", "local", "status"),
        ("What's the status of my reef tank?", "current_stats", "local", "status"),
        ("What is my pH?", "current_stats", "local", "ph"),
        ("What is my pH right now?", "current_stats", "local", "ph"),
        ("What's my current pH?", "current_stats", "local", "ph"),
        ("What is my temperature?", "current_stats", "local", "tmp"),
        ("What's the temperature of my tank?", "current_stats", "local", "tmp"),
        ("What is my ORP?", "current_stats", "local", "orp"),
        ("What's my ATO?", "current_stats", "local", "ato"),
        ("What's my current ATO?", "current_stats", "local", "ato"),
        ("How much ATO do I have?", "current_stats", "local", "ato"),
        ("What are the current reef readings?", "current_stats", "local", "status"),
        ("Give me my current reef readings.", "current_stats", "local", "status"),
        ("What are my current reef parameters?", "current_stats", "local", "status"),
        ("Check my reef tank", "current_stats", "local", "status"),
        ("What's my reef looking like right now?", "current_stats", "local", "status"),
        ("Give me a reef tank report.", "detailed_report", "ask_hermes", None),
        ("What is my reef tank report?", "detailed_report", "ask_hermes", None),
        ("Can you give me a reef tank report?", "detailed_report", "ask_hermes", None),
        ("Give me a report on my reef tank", "detailed_report", "ask_hermes", None),
        ("Give me a report on the status of my reef tank.", "detailed_report", "ask_hermes", None),
        ("Can you give me a detailed reef report?", "detailed_report", "ask_hermes", None),
        ("Give me a detailed reef tank report.", "detailed_report", "ask_hermes", None),
        ("What's my latest reef report?", "detailed_report", "ask_hermes", None),
        ("Give me the latest reef report.", "detailed_report", "ask_hermes", None),
        ("Give me a full report on my reef tank.", "detailed_report", "ask_hermes", None),
        ("Can you report on my reef tank?", "detailed_report", "ask_hermes", None),
        ("What's the report on my reef tank?", "detailed_report", "ask_hermes", None),
        ("Give me a detailed report on the reef.", "detailed_report", "ask_hermes", None),
        ("Give me my reef report.", "detailed_report", "ask_hermes", None),
        ("reef tank repot", "detailed_report", "ask_hermes", None),
        ("reef tank repo", "detailed_report", "ask_hermes", None),
        ("reef report please", "detailed_report", "ask_hermes", None),
        ("Analyse my reef tank.", "detailed_report", "ask_hermes", None),
        ("Analyze my reef tank", "detailed_report", "ask_hermes", None),
        ("Give me a reef tank analysis", "detailed_report", "ask_hermes", None),
        ("How is my reef tank trending?", "trends", "ask_hermes", None),
        ("What are my reef tank trends?", "trends", "ask_hermes", None),
        ("What are my reef trends?", "trends", "ask_hermes", None),
        ("Can you give me a reef trending report?", "trends", "ask_hermes", None),
        ("Richie, can you give me a trending report?", "trends", "ask_hermes", None),
        ("give me a trending report", "trends", "ask_hermes", None),
        ("trending report", "trends", "ask_hermes", None),
        ("trend report", "trends", "ask_hermes", None),
        ("reef trending report", "trends", "ask_hermes", None),
        ("reef trends", "trends", "ask_hermes", None),
        ("reef history", "trends", "ask_hermes", None),
        ("What are the trends in my reef?", "trends", "ask_hermes", None),
        ("Show me my reef trends", "trends", "ask_hermes", None),
        ("Give me my reef trend report.", "trends", "ask_hermes", None),
        ("How is my pH trending?", "trends", "ask_hermes", None),
        ("How is my ORP trending?", "trends", "ask_hermes", None),
        ("How has my temperature been trending?", "trends", "ask_hermes", None),
        ("How has my ATO been trending?", "trends", "ask_hermes", None),
        ("How has my ORP been trending?", "trends", "ask_hermes", None),
        ("What has changed in my reef tank?", "trends", "ask_hermes", None),
        ("How has my reef tank changed?", "trends", "ask_hermes", None),
        ("What's happening with my reef parameters?", "trends", "ask_hermes", None),
        ("How has my reef tank been doing?", "trends", "ask_hermes", None),
        ("How has my reef changed?", "trends", "ask_hermes", None),
        ("how have my reef parameters changed", "trends", "ask_hermes", None),
        ("Are my reef parameters improving or getting worse?", "trends", "ask_hermes", None),
        ("What trends are you seeing in my reef?", "trends", "ask_hermes", None),
        ("Ask Hermes what my reef tank report is.", "detailed_report", "ask_hermes", None),
        ("Ask Hermes for my reef tank report.", "detailed_report", "ask_hermes", None),
        ("Ask Hermes what my reef trends are.", "trends", "ask_hermes", None),
        ("Can you ask Hermes how my pH is trending?", "trends", "ask_hermes", None),
        ("Ask Hermes about my reef tank.", "detailed_report", "ask_hermes", None),
        ("Reachy, ask Hermes what my Reef Tank report is.", "detailed_report", "ask_hermes", None),
    ],
)
def test_classify_reef_intent_routes_status_and_reports(
    transcript: str,
    intent: str,
    route: str,
    metric: str | None,
) -> None:
    """Current stats stay on Apex; report/trend/analysis belong to Hermes."""
    classified = classify_reef_intent(transcript)
    assert classified is not None
    assert classified.intent == intent
    assert classified.route == route
    assert classified.metric == metric


def test_classify_reef_intent_ignores_unrelated_speech() -> None:
    """Non-reef speech is not claimed by the reef router."""
    assert classify_reef_intent("what's the weather") is None
    assert classify_reef_intent("Give me a weather report") is None
    assert classify_reef_intent("Ask Hermes about the weather") is None


def test_explicit_ask_hermes_overrides_current_stats() -> None:
    """An explicit Hermes request must not be redirected to local current stats."""
    classified = classify_reef_intent("Ask Hermes about my reef tank.")
    assert classified is not None
    assert classified.route == "ask_hermes"
    assert classified.explicit_hermes_request is True
    classified = classify_reef_intent("Ask Hermes what my reef tank report is.")
    assert classified is not None
    assert classified.intent == "detailed_report"
    assert classified.explicit_hermes_request is True
    classified = classify_reef_intent("Ask Hermes what my reef trends are.")
    assert classified is not None
    assert classified.intent == "trends"
    assert classified.explicit_hermes_request is True


def test_reef_source_question_and_answers() -> None:
    """Source follow-ups are classified from stored metadata, not guessed."""
    assert match_reef_source_question("Did that come from Hermes?")
    assert match_reef_source_question("Rishi, did that come from Hermes?")
    assert match_reef_source_question("Was that from Hermes?")
    assert match_reef_source_question("Did Hermes give you that?")
    assert match_reef_source_question("Did you get that from Hermes?")
    assert not match_reef_source_question("What is my reef tank report?")
    assert spoken_reef_source_answer("hermes", "detailed_report") == "Yes. That came from Hermes' Reef Tank report."
    assert spoken_reef_source_answer("hermes", "trends") == "Yes. That came from Hermes' Reef Tank trend data."
    assert (
        spoken_reef_source_answer("home_assistant", "current_stats")
        == "No. That came directly from the current reef tank data in Home Assistant."
    )


def test_spoken_apex_update_does_not_invent_salinity() -> None:
    """A salinity question with no salinity probe does not invent a percentage."""
    result = {
        "apex_status": {
            "water_parameters": {"LLSATO": {"value": 2.9}, "Tmp": {"value": 24.0}},
            "equipment": {"ato": {"llsato": 2.9}},
        },
        "source": "apex_status_http",
    }
    spoken = spoken_apex_update(result, "salinity")
    assert spoken == "There is no salinity reading in the current Apex status."
    assert "85" not in spoken
    assert "balanced" not in spoken.lower()


def test_spoken_ato_uses_raw_llsato() -> None:
    """ATO questions report the LLSATO probe, not a calibrated percentage."""
    result = {
        "apex_status": {
            "water_parameters": {"LLSATO": {"value": 2.9}},
            "equipment": {"ato": {"llsato": 2.9}},
        }
    }
    assert spoken_apex_update(result, "ato") == "LLSATO is 2.9."
