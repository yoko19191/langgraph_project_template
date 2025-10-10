"""MCP configuration loading utilities with smart file discovery."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


class MCPServerConfig(BaseModel):
    """MCP server configuration model.

    A server can be configured either as:
    1. HTTP/SSE server with a URL
    2. Command-based server with command and args
    """

    url: str | None = Field(None, description="HTTP/SSE endpoint URL")
    command: str | None = Field(None, description="Executable command")
    args: List[str] | None = Field(None, description="Command arguments")
    transport: str | None = Field(
        None, description="Transport type (http, sse, streamable_http)"
    )

    @field_validator("url", "command")
    @classmethod
    def check_at_least_one_endpoint(cls, v: str | None, info) -> str | None:
        """Validate that either url or command is provided."""
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that at least one of url or command is provided."""
        if not self.url and not self.command:
            raise ValueError(
                "Either 'url' or 'command' must be provided for MCP server configuration"
            )
        if self.url and self.command:
            logger.warning(
                "Both 'url' and 'command' provided. URL-based configuration will take precedence."
            )


class MCPConfig(BaseModel):
    """Root MCP configuration model."""

    mcpServers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="Dictionary of MCP server configurations"
    )


def find_nearest_config_file(
    config_name: str = "mcp.json", start_path: Path | None = None
) -> Path | None:
    """Find nearest config file by traversing up the directory tree.

    Args:
        config_name: Name of config file to find (default: "mcp.json")
        start_path: Starting directory for search (default: __file__ location)

    Returns:
        Path to config file if found, None otherwise

    Example:
        >>> config_path = find_nearest_config_file("mcp.json")
        >>> config_path = find_nearest_config_file("custom-mcp.json")
    """
    current = start_path or Path(__file__).resolve().parent
    root = Path("/")

    while current != root:
        config_path = current / config_name
        if config_path.exists() and config_path.is_file():
            logger.debug(f"Found config file at: {config_path}")
            return config_path
        current = current.parent

    # Check root as well
    config_path = root / config_name
    if config_path.exists() and config_path.is_file():
        logger.debug(f"Found config file at: {config_path}")
        return config_path

    return None


def load_mcp_config(
    config_name: str = "mcp.json", config_path: Path | None = None
) -> Dict[str, Dict[str, Any]]:
    """Load and validate MCP server configurations.

    Args:
        config_name: Name of config file (default: "mcp.json")
        config_path: Explicit path to config file. If None, searches upward
                    from current location for nearest config file.

    Returns:
        Dictionary of MCP server configurations compatible with
        langchain-mcp-adapters MultiServerMCPClient.

    Raises:
        FileNotFoundError: If config file is not found.
        ValueError: If configuration is invalid.

    Example:
        >>> # Auto-discovery
        >>> config = load_mcp_config()
        >>>
        >>> # Custom name
        >>> config = load_mcp_config("custom-mcp.json")
        >>>
        >>> # Explicit path
        >>> config = load_mcp_config(config_path=Path("/path/to/mcp.json"))
    """
    # Determine config file path
    if config_path is None:
        config_path = find_nearest_config_file(config_name)
        if config_path is None:
            raise FileNotFoundError(
                f"MCP configuration file '{config_name}' not found. "
                f"Searched upward from {Path(__file__).resolve().parent}"
            )

    # Check if file exists (when explicit path provided)
    if not config_path.exists():
        raise FileNotFoundError(
            f"MCP configuration file not found: {config_path}\n"
            f"Please create {config_name} in an accessible directory."
        )

    # Load and parse JSON
    try:
        with open(config_path, encoding="utf-8") as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {config_path}: {e}") from e

    # Validate using pydantic model
    try:
        mcp_config = MCPConfig(**raw_config)
    except ValidationError as e:
        raise ValueError(f"Invalid MCP configuration in {config_path}:\n{e}") from e

    # Convert to format expected by langchain-mcp-adapters
    servers_config: Dict[str, Dict[str, Any]] = {}

    for server_name, server_config in mcp_config.mcpServers.items():
        config_dict = server_config.model_dump(exclude_none=True)

        # Infer transport type if not specified
        if "transport" not in config_dict:
            if config_dict.get("url"):
                url = config_dict["url"]
                if "/sse" in url:
                    config_dict["transport"] = "sse"
                elif "mcp.deepwiki.com" in url:
                    config_dict["transport"] = "streamable_http"
                else:
                    config_dict["transport"] = "http"

        servers_config[server_name] = config_dict

    logger.info(f"Loaded {len(servers_config)} MCP server(s) from {config_path}")
    for server_name in servers_config:
        logger.debug(f"  - {server_name}")

    return servers_config
