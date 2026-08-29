"""Register local or LAN MCP servers next to Hugging Face Tool Spaces.

Reachy motor, camera, and memory tools stay in-process. This module only
adapts HTTP MCP servers (Hermes-owned or hand-written) into RemoteMcpTool.
"""

import os
import json
import logging
import threading
from typing import Any
from pathlib import Path
from dataclasses import field, asdict, dataclass

from reachy_mini_conversation_app.mcp_client import (
    RemoteToolSpec,
    RemoteMcpToolClient,
    RemoteMcpServerConfig,
    validate_http_mcp_url,
    apply_name_normalization,
    build_namespaced_tool_name,
)
from reachy_mini_conversation_app.tool_spaces import TERMINAL_EXTERNAL_CONTENT_DIRECTORY


logger = logging.getLogger(__name__)

INSTALLED_LOCAL_MCP_FILENAME = "installed_local_mcp.json"
INSTALLED_LOCAL_MCP_VERSION = 1
HERMES_CONFIG_PATH_ENV = "HERMES_CONFIG_PATH"
_MANIFEST_LOCK = threading.RLock()
_DEFAULT_HERMES_CONFIG_PATHS = (
    Path.home() / ".hermes" / "config.yaml",
    Path.home() / ".hermes" / "mcp_servers.yaml",
)


@dataclass(frozen=True)
class InstalledLocalMcpTool:
    """Cached metadata for one tool on a local or LAN MCP server."""

    local_name: str
    client_tool_name: str
    remote_name: str
    description: str
    parameters_schema: dict[str, Any]


@dataclass(frozen=True)
class InstalledLocalMcpServer:
    """One HTTP MCP server registered for the conversation app."""

    alias: str
    url: str
    source: str = "manual"
    bearer_token: str | None = None
    timeout_s: float = 30.0
    enabled_tools: list[str] = field(default_factory=list)
    tools: list[InstalledLocalMcpTool] = field(default_factory=list)


@dataclass(frozen=True)
class InstalledLocalMcpManifest:
    """Persisted local/LAN MCP servers."""

    version: int = INSTALLED_LOCAL_MCP_VERSION
    servers: list[InstalledLocalMcpServer] = field(default_factory=list)


def get_installed_local_mcp_path(instance_path: str | Path | None) -> Path:
    """Return the local MCP manifest path for the current mode."""
    if instance_path is not None:
        return Path(instance_path) / INSTALLED_LOCAL_MCP_FILENAME
    return TERMINAL_EXTERNAL_CONTENT_DIRECTORY / INSTALLED_LOCAL_MCP_FILENAME


def _normalize_alias(alias: str) -> str:
    normalized = apply_name_normalization(alias)
    if not normalized:
        raise ValueError(f"MCP server alias {alias!r} cannot be normalized.")
    return normalized


def _tool_from_raw(alias: str, raw_tool: dict[str, Any]) -> InstalledLocalMcpTool:
    remote_name = str(raw_tool.get("remote_name") or "").strip()
    if not remote_name:
        raise ValueError(f"Local MCP server '{alias}' has a tool without remote_name.")
    local_name = str(raw_tool.get("local_name") or "").strip() or build_namespaced_tool_name(alias, remote_name)
    client_tool_name = str(raw_tool.get("client_tool_name") or "").strip() or local_name
    parameters_schema = raw_tool.get("parameters_schema")
    if not isinstance(parameters_schema, dict):
        parameters_schema = {"type": "object", "properties": {}, "required": []}
    return InstalledLocalMcpTool(
        local_name=local_name,
        client_tool_name=client_tool_name,
        remote_name=remote_name,
        description=str(raw_tool.get("description") or "").strip(),
        parameters_schema=parameters_schema,
    )


def _server_from_raw(raw_server: dict[str, Any]) -> InstalledLocalMcpServer:
    alias = _normalize_alias(str(raw_server.get("alias") or ""))
    url = validate_http_mcp_url(str(raw_server.get("url") or "").strip())
    source = str(raw_server.get("source") or "manual").strip() or "manual"
    bearer = raw_server.get("bearer_token")
    bearer_token = str(bearer).strip() if isinstance(bearer, str) and bearer.strip() else None
    timeout_s = float(raw_server.get("timeout_s") or 30.0)
    if timeout_s <= 0:
        raise ValueError(f"Local MCP server '{alias}' timeout_s must be greater than zero.")
    raw_enabled = raw_server.get("enabled_tools") or []
    if not isinstance(raw_enabled, list) or not all(isinstance(name, str) for name in raw_enabled):
        raise ValueError(f"Local MCP server '{alias}' enabled_tools must be a list of strings.")
    raw_tools = raw_server.get("tools") or []
    if not isinstance(raw_tools, list):
        raise ValueError(f"Local MCP server '{alias}' tools must be a list.")
    return InstalledLocalMcpServer(
        alias=alias,
        url=url,
        source=source,
        bearer_token=bearer_token,
        timeout_s=timeout_s,
        enabled_tools=[name.strip() for name in raw_enabled if name.strip()],
        tools=[_tool_from_raw(alias, tool) for tool in raw_tools if isinstance(tool, dict)],
    )


def _placeholder_tool(alias: str, remote_name: str) -> InstalledLocalMcpTool:
    local_name = build_namespaced_tool_name(alias, remote_name)
    return InstalledLocalMcpTool(
        local_name=local_name,
        client_tool_name=local_name,
        remote_name=remote_name,
        description=f"Local MCP tool '{remote_name}' from '{alias}'.",
        parameters_schema={"type": "object", "properties": {}, "required": []},
    )


def read_installed_local_mcp(instance_path: str | Path | None) -> InstalledLocalMcpManifest:
    """Read the local MCP manifest. Missing files yield an empty catalog."""
    with _MANIFEST_LOCK:
        manifest_path = get_installed_local_mcp_path(instance_path)
        if not manifest_path.exists():
            return InstalledLocalMcpManifest()
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read local MCP manifest {manifest_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Local MCP manifest {manifest_path} must be a JSON object.")
        raw_servers = raw.get("servers") or []
        if not isinstance(raw_servers, list):
            raise ValueError(f"Local MCP manifest {manifest_path} servers must be a list.")
        servers = [_server_from_raw(server) for server in raw_servers if isinstance(server, dict)]
        return InstalledLocalMcpManifest(
            version=int(raw.get("version") or INSTALLED_LOCAL_MCP_VERSION), servers=servers
        )


def write_installed_local_mcp(
    manifest: InstalledLocalMcpManifest,
    instance_path: str | Path | None,
) -> Path:
    """Persist the local MCP manifest."""
    with _MANIFEST_LOCK:
        manifest_path = get_installed_local_mcp_path(instance_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INSTALLED_LOCAL_MCP_VERSION,
            "servers": [
                {
                    **asdict(server),
                    "tools": [asdict(tool) for tool in server.tools],
                }
                for server in manifest.servers
            ],
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return manifest_path


def build_local_mcp_client(server: InstalledLocalMcpServer) -> RemoteMcpToolClient:
    """Build an MCP client for one local or LAN server."""
    headers = {"Authorization": f"Bearer {server.bearer_token}"} if server.bearer_token else {}
    return RemoteMcpToolClient(
        RemoteMcpServerConfig(
            alias=server.alias,
            url=server.url,
            headers=headers,
            request_timeout_s=min(10.0, server.timeout_s),
            tool_timeout_s=server.timeout_s,
        ),
        known_tools=[
            RemoteToolSpec(
                server_alias=server.alias,
                remote_name=tool.remote_name,
                namespaced_name=tool.client_tool_name,
                description=tool.description,
                parameters_schema=tool.parameters_schema,
            )
            for tool in server.tools
            if tool.remote_name
        ],
    )


def iter_enabled_local_mcp_tools(
    tool_names: list[str],
    instance_path: str | Path | None,
) -> list[tuple[InstalledLocalMcpServer, InstalledLocalMcpTool, RemoteMcpToolClient]]:
    """Return enabled local/LAN MCP tools and shared clients, without network calls."""
    skip_hermes_owned = "ask_hermes" in tool_names
    enabled: list[tuple[InstalledLocalMcpServer, InstalledLocalMcpTool, RemoteMcpToolClient]] = []
    for server in read_installed_local_mcp(instance_path).servers:
        if skip_hermes_owned and server.source == "hermes":
            logger.info(
                "Skipping Hermes-owned MCP server '%s'; ask_hermes is the enabled Hermes gateway.",
                server.alias,
            )
            continue
        enabled_tool_names = {name for name in tool_names if name.startswith(f"{server.alias}__")}
        if server.enabled_tools:
            allowed_local_names = {build_namespaced_tool_name(server.alias, item) for item in server.enabled_tools}
            enabled_tool_names.update(name for name in tool_names if name in allowed_local_names)
        if not enabled_tool_names:
            continue
        if not server.tools:
            logger.warning(
                "Local MCP server '%s' has no cached tools; enable it after a Hermes sync or manifest refresh.",
                server.alias,
            )
            continue
        client = build_local_mcp_client(server)
        for tool in server.tools:
            if tool.local_name not in enabled_tool_names:
                continue
            enabled.append((server, tool, client))
    return enabled


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Hermes MCP sync requires PyYAML to read config.yaml.") from exc
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Hermes config {path} must be a mapping.")
    return raw


def _hermes_config_paths() -> list[Path]:
    configured = (os.getenv(HERMES_CONFIG_PATH_ENV) or "").strip()
    if configured:
        return [Path(configured).expanduser()]
    return list(_DEFAULT_HERMES_CONFIG_PATHS)


def _mcp_servers_from_hermes_config(raw: dict[str, Any]) -> dict[str, Any]:
    servers = raw.get("mcp_servers")
    if servers is None:
        return {}
    if not isinstance(servers, dict):
        raise ValueError("Hermes mcp_servers must be a mapping of alias to server config.")
    return servers


def _server_from_hermes_entry(alias: str, raw_server: dict[str, Any]) -> InstalledLocalMcpServer | None:
    if raw_server.get("enabled") is False:
        return None
    url = raw_server.get("url")
    if not url:
        logger.info("Skipping Hermes MCP server '%s': stdio servers are not imported.", alias)
        return None
    headers = raw_server.get("headers") or {}
    bearer_token = None
    if isinstance(headers, dict):
        authorization = str(headers.get("Authorization") or headers.get("authorization") or "")
        prefix = "Bearer "
        if authorization.startswith(prefix):
            bearer_token = authorization[len(prefix) :].strip() or None
    tools_policy = raw_server.get("tools") or {}
    include: list[str] = []
    if isinstance(tools_policy, dict):
        raw_include = tools_policy.get("include") or []
        if isinstance(raw_include, str):
            include = [raw_include]
        elif isinstance(raw_include, list):
            include = [str(name) for name in raw_include if str(name).strip()]
    timeout_s = float(raw_server.get("timeout") or raw_server.get("timeout_s") or 30.0)
    normalized_alias = _normalize_alias(alias)
    tools = [_placeholder_tool(normalized_alias, name) for name in include]
    return InstalledLocalMcpServer(
        alias=normalized_alias,
        url=validate_http_mcp_url(str(url).strip()),
        source="hermes",
        bearer_token=bearer_token,
        timeout_s=timeout_s,
        enabled_tools=include,
        tools=tools,
    )


def sync_hermes_mcp_catalog(instance_path: str | Path | None = None) -> InstalledLocalMcpManifest:
    """Import HTTP MCP servers from Hermes without replacing manual entries."""
    existing = {server.alias: server for server in read_installed_local_mcp(instance_path).servers}
    imported: dict[str, InstalledLocalMcpServer] = {}
    for path in _hermes_config_paths():
        if not path.is_file():
            continue
        try:
            raw = (
                _load_yaml_mapping(path)
                if path.suffix.lower() in {".yaml", ".yml"}
                else json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read Hermes config %s: %s", path, exc)
            continue
        if not isinstance(raw, dict):
            logger.warning("Ignoring Hermes config %s: expected a mapping.", path)
            continue
        try:
            hermes_servers = _mcp_servers_from_hermes_config(raw)
        except ValueError as exc:
            logger.warning("Ignoring Hermes config %s: %s", path, exc)
            continue
        for alias, raw_server in hermes_servers.items():
            if not isinstance(raw_server, dict):
                continue
            try:
                server = _server_from_hermes_entry(str(alias), raw_server)
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping Hermes MCP server %r: %s", alias, exc)
                continue
            if server is not None:
                imported[server.alias] = server

    merged: list[InstalledLocalMcpServer] = []
    seen: set[str] = set()
    for alias, server in existing.items():
        replacement = imported.get(alias)
        if replacement is not None and server.source == "hermes":
            if server.url == replacement.url and server.tools and not replacement.tools:
                replacement = InstalledLocalMcpServer(
                    alias=replacement.alias,
                    url=replacement.url,
                    source="hermes",
                    bearer_token=replacement.bearer_token,
                    timeout_s=replacement.timeout_s,
                    enabled_tools=replacement.enabled_tools or server.enabled_tools,
                    tools=server.tools,
                )
            merged.append(replacement)
        else:
            merged.append(server)
        seen.add(alias)
    for alias, server in imported.items():
        if alias not in seen:
            merged.append(server)

    manifest = InstalledLocalMcpManifest(servers=merged)
    write_installed_local_mcp(manifest, instance_path)
    logger.info("Hermes MCP sync wrote %d local server(s).", len(merged))
    return manifest
