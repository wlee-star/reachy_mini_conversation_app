from __future__ import annotations
from pathlib import Path

import pytest

from reachy_mini_conversation_app.local_mcp import (
    InstalledLocalMcpTool,
    InstalledLocalMcpServer,
    InstalledLocalMcpManifest,
    sync_hermes_mcp_catalog,
    read_installed_local_mcp,
    write_installed_local_mcp,
    iter_enabled_local_mcp_tools,
)
from reachy_mini_conversation_app.personality import available_tool_catalog
from reachy_mini_conversation_app.tools.core_tools import get_tools, initialize_tools


def _time_tool() -> InstalledLocalMcpTool:
    return InstalledLocalMcpTool(
        local_name="time__get_time",
        client_tool_name="time__get_time",
        remote_name="get_time",
        description="Local timezone lookup.",
        parameters_schema={"type": "object", "properties": {"timezone": {"type": "string"}}, "required": []},
    )


def test_read_installed_local_mcp_missing_file_is_empty(tmp_path: Path) -> None:
    """A missing manifest is an empty catalog, not seeded cloud Spaces."""
    manifest = read_installed_local_mcp(tmp_path)
    assert manifest.servers == []


def test_write_and_read_local_mcp_manifest(tmp_path: Path) -> None:
    """Hand-written local MCP servers round-trip through the manifest."""
    server = InstalledLocalMcpServer(
        alias="time",
        url="http://127.0.0.1:8760/mcp",
        source="manual",
        tools=[_time_tool()],
    )
    write_installed_local_mcp(InstalledLocalMcpManifest(servers=[server]), tmp_path)

    loaded = read_installed_local_mcp(tmp_path)
    assert loaded.servers[0].alias == "time"
    assert loaded.servers[0].tools[0].local_name == "time__get_time"


def test_iter_enabled_local_mcp_tools_filters_by_profile(tmp_path: Path) -> None:
    """Only profile-enabled local MCP tools are returned."""
    write_installed_local_mcp(
        InstalledLocalMcpManifest(
            servers=[
                InstalledLocalMcpServer(
                    alias="time",
                    url="http://192.168.1.9:8760/mcp",
                    source="manual",
                    tools=[_time_tool()],
                )
            ]
        ),
        tmp_path,
    )

    enabled = iter_enabled_local_mcp_tools(["time__get_time"], tmp_path)
    ignored = iter_enabled_local_mcp_tools(["dance"], tmp_path)

    assert len(enabled) == 1
    assert enabled[0][1].local_name == "time__get_time"
    assert ignored == []


def test_initialize_tools_registers_local_mcp_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local MCP tools join the registry through the existing RemoteMcpTool adapter."""
    write_installed_local_mcp(
        InstalledLocalMcpManifest(
            servers=[
                InstalledLocalMcpServer(
                    alias="time",
                    url="http://127.0.0.1:8760/mcp",
                    source="manual",
                    tools=[_time_tool()],
                )
            ]
        ),
        tmp_path,
    )
    (tmp_path / "profile_toolsets.json").write_text(
        '{"version": 1, "profiles": {"default": ["time__get_time"]}}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("reachy_mini_conversation_app.config.config.REACHY_MINI_CUSTOM_PROFILE", "default")
    monkeypatch.setattr("reachy_mini_conversation_app.tools.core_tools.config.REACHY_MINI_CUSTOM_PROFILE", "default")

    initialize_tools(tmp_path, force=True)
    tools = get_tools()
    assert "time__get_time" in tools
    assert tools["time__get_time"].name == "time__get_time"


def test_iter_enabled_skips_hermes_mcp_when_ask_hermes_is_enabled(tmp_path: Path) -> None:
    """Hermes-synced MCP tools must not also dispatch while ask_hermes is enabled."""
    write_installed_local_mcp(
        InstalledLocalMcpManifest(
            servers=[
                InstalledLocalMcpServer(
                    alias="apex",
                    url="http://10.0.0.8:8751/mcp",
                    source="hermes",
                    tools=[
                        InstalledLocalMcpTool(
                            local_name="apex__get_temperature",
                            client_tool_name="apex__get_temperature",
                            remote_name="get_temperature",
                            description="Reef temperature.",
                            parameters_schema={"type": "object", "properties": {}, "required": []},
                        )
                    ],
                ),
                InstalledLocalMcpServer(
                    alias="time",
                    url="http://127.0.0.1:8760/mcp",
                    source="manual",
                    tools=[_time_tool()],
                ),
            ]
        ),
        tmp_path,
    )

    with_hermes = iter_enabled_local_mcp_tools(["ask_hermes", "apex__get_temperature", "time__get_time"], tmp_path)
    without_hermes = iter_enabled_local_mcp_tools(["apex__get_temperature", "time__get_time"], tmp_path)

    assert [tool.local_name for _server, tool, _client in with_hermes] == ["time__get_time"]
    assert {tool.local_name for _server, tool, _client in without_hermes} == {
        "apex__get_temperature",
        "time__get_time",
    }


def test_available_tool_catalog_includes_local_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool access lists local MCP tools beside built-in and Space tools."""
    write_installed_local_mcp(
        InstalledLocalMcpManifest(
            servers=[
                InstalledLocalMcpServer(
                    alias="apex",
                    url="http://10.0.0.8:8751/mcp",
                    source="hermes",
                    tools=[
                        InstalledLocalMcpTool(
                            local_name="apex__get_temperature",
                            client_tool_name="apex__get_temperature",
                            remote_name="get_temperature",
                            description="Reef temperature.",
                            parameters_schema={"type": "object", "properties": {}, "required": []},
                        )
                    ],
                )
            ]
        ),
        tmp_path,
    )
    monkeypatch.setattr("reachy_mini_conversation_app.config.config.INSTANCE_PATH", tmp_path)

    catalog = {tool["id"]: tool for tool in available_tool_catalog()}
    assert catalog["apex__get_temperature"]["kind"] == "local_mcp"
    assert catalog["apex__get_temperature"]["source"] == "local:apex"


def test_sync_hermes_mcp_catalog_imports_http_and_skips_stdio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hermes HTTP servers are imported; stdio servers and manual entries stay put."""
    hermes_config = tmp_path / "hermes.yaml"
    hermes_config.write_text(
        """
mcp_servers:
  apex:
    url: http://192.168.1.40:8751/mcp
    timeout: 20
    tools:
      include: [get_temperature, get_ph]
  files:
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem"]
  home_assistant:
    url: http://127.0.0.1:8123/mcp
    enabled: false
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_CONFIG_PATH", str(hermes_config))
    write_installed_local_mcp(
        InstalledLocalMcpManifest(
            servers=[
                InstalledLocalMcpServer(
                    alias="time",
                    url="http://127.0.0.1:8760/mcp",
                    source="manual",
                    tools=[_time_tool()],
                )
            ]
        ),
        tmp_path,
    )

    manifest = sync_hermes_mcp_catalog(tmp_path)
    aliases = {server.alias: server for server in manifest.servers}
    assert "time" in aliases
    assert aliases["time"].source == "manual"
    assert aliases["apex"].source == "hermes"
    assert aliases["apex"].url == "http://192.168.1.40:8751/mcp"
    assert [tool.remote_name for tool in aliases["apex"].tools] == ["get_temperature", "get_ph"]
    assert "files" not in aliases
    assert "home_assistant" not in aliases
