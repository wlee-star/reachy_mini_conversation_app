import json
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest

from reachy_mini_conversation_app.tools import reef_status as reef_status_mod
from reachy_mini_conversation_app.tools.apex import Apex
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
