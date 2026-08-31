from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_apex import _FakeResponse, _FakeAsyncClient

from reachy_mini_conversation_app.tools import reef_status as reef_status_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.reef_status import ReefStatus


def _deps() -> ToolDependencies:
    return ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())


@pytest.mark.asyncio
async def test_reef_status_reads_live_status_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reef_status uses the live Apex /status URL when configured."""
    monkeypatch.setattr(reef_status_mod.config, "APEX_STATUS_URL", "http://192.168.0.143:8080/status")
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(tmp_path / "missing.json"))
    _FakeAsyncClient.requested_urls = []
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "controller": {"hostname": "Cade_S3_1200_P"},
            "probes": [
                {"name": "Tmp", "value": 24.2, "type": "Temp"},
                {"name": "LLSATO", "value": 12.3, "type": None},
            ],
        },
    )
    monkeypatch.setattr(reef_status_mod.httpx, "AsyncClient", _FakeAsyncClient)

    result = await ReefStatus()(_deps())

    assert result["source"] == "apex_status_http"
    assert result["reef_status"]["probes"]["Tmp"]["value"] == 24.2
    assert result["reef_status"]["ato"]["llsato"] == 12.3
    assert result["reef_status"]["controller"] == "Cade_S3_1200_P"


@pytest.mark.asyncio
async def test_reef_status_keeps_raw_llsato_from_live_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """reef_status fallback also preserves LLSATO 2.9 and fetched_at."""
    monkeypatch.setattr(reef_status_mod.config, "APEX_STATUS_URL", "http://192.168.0.143:8080/status")
    monkeypatch.setattr(reef_status_mod, "CACHE_PATH", str(tmp_path / "missing.json"))
    _FakeAsyncClient.requested_urls = []
    _FakeAsyncClient.response = _FakeResponse(
        200,
        {
            "probes": [
                {"name": "Tmp", "value": 24.0, "type": "Temp"},
                {"name": "LLSATO", "value": 2.9, "type": None},
            ],
            "fetched_at": "2026-08-31T03:22:00.869566Z",
        },
    )
    monkeypatch.setattr(reef_status_mod.httpx, "AsyncClient", _FakeAsyncClient)

    result = await ReefStatus()(_deps())

    assert result["source"] == "apex_status_http"
    assert result["reef_status"]["probes"]["LLSATO"]["value"] == 2.9
    assert result["reef_status"]["ato"]["llsato"] == 2.9
    assert result["reef_status"]["cached_at"] == "2026-08-31T03:22:00.869566Z"
    assert "85" not in str(result)
