"""MCP Client setup and management for LangChain/LangGraph Agent."""

import asyncio
import logging
from typing import Any, Callable, Dict, List, cast

from langchain_mcp_adapters.client import (  # type: ignore[import-untyped]
    MultiServerMCPClient,
)

from src.app.common.utils.mcp import load_mcp_config

# Global MCP client and caches
_mcp_client: MultiServerMCPClient | None = None
_mcp_tools_cache: Dict[str, List[Callable[..., Any]]] = {}
_mcp_prompts_cache: Dict[str, List[Dict[str, Any]]] = {}
_mcp_resources_cache: Dict[str, List[Dict[str, Any]]] = {}

logger = logging.getLogger(__name__)

# MCP Server configurations - loaded from config file
_MCP_SERVERS: Dict[str, Dict[str, Any]] | None = None


def _get_mcp_servers() -> Dict[str, Dict[str, Any]]:
    """Get MCP server configurations, loading from config if needed.

    Returns:
        Dictionary of MCP server configurations.
    """
    global _MCP_SERVERS

    if _MCP_SERVERS is None:
        try:
            _MCP_SERVERS = load_mcp_config()
        except FileNotFoundError:
            logger.warning(
                "MCP configuration file not found. No MCP servers will be available. "
                "Create mcp.json to configure MCP servers."
            )
            _MCP_SERVERS = {}
        except Exception as e:
            logger.error(f"Failed to load MCP configuration: {e}")
            _MCP_SERVERS = {}

    return _MCP_SERVERS


async def aget_mcp_client(
    server_configs: Dict[str, Any] | None = None,
) -> MultiServerMCPClient | None:
    """Get or initialize MCP client with given server configurations (async).

    If server_configs is provided, creates a new client for those specific servers.
    If no server_configs provided, uses the global client with all configured servers.

    Args:
        server_configs: Optional dict of server configurations to use

    Returns:
        MultiServerMCPClient instance or None if initialization fails
    """
    global _mcp_client

    # If specific server configs provided, create a dedicated client for them
    if server_configs is not None:
        try:
            client = MultiServerMCPClient(server_configs)  # pyright: ignore[reportArgumentType]
            logger.info(
                f"Created MCP client with {len(server_configs)} server(s): {list(server_configs.keys())}"
            )
            return client
        except Exception as e:
            logger.error(f"Failed to create MCP client: {e}")
            return None

    # Otherwise, use global client for all servers (backward compatibility)
    if _mcp_client is None:
        mcp_servers = _get_mcp_servers()
        if not mcp_servers:
            logger.warning("No MCP servers configured")
            return None

        try:
            _mcp_client = MultiServerMCPClient(mcp_servers)  # pyright: ignore[reportArgumentType]
            logger.info(
                f"Initialized global MCP client with {len(mcp_servers)} server(s): {list(mcp_servers.keys())}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize global MCP client: {e}")
            return None
    return _mcp_client


def get_mcp_client(
    server_configs: Dict[str, Any] | None = None,
) -> MultiServerMCPClient | None:
    """Get or initialize MCP client with given server configurations (sync).

    Synchronous wrapper for aget_mcp_client.

    Args:
        server_configs: Optional dict of server configurations to use

    Returns:
        MultiServerMCPClient instance or None if initialization fails
    """
    return asyncio.run(aget_mcp_client(server_configs))


async def aget_mcp_tools(server_name: str) -> List[Callable[..., Any]]:
    """Get MCP tools for a specific server, initializing client if needed (async).

    Args:
        server_name: Name of the MCP server to get tools from

    Returns:
        List of callable tools from the specified server
    """
    global _mcp_tools_cache

    # Return cached tools if available
    if server_name in _mcp_tools_cache:
        logger.debug(f"Returning cached tools for MCP server '{server_name}'")
        return _mcp_tools_cache[server_name]

    # Check if server exists in configuration
    mcp_servers = _get_mcp_servers()
    if server_name not in mcp_servers:
        logger.warning(f"MCP server '{server_name}' not found in configuration")
        _mcp_tools_cache[server_name] = []
        return []

    try:
        # Create a client with only this specific server to get its tools
        server_config = {server_name: mcp_servers[server_name]}
        client = MultiServerMCPClient(server_config)  # pyright: ignore[reportArgumentType]

        # Get tools from this server-specific client
        tools = await client.get_tools()
        tools_list = cast(List[Callable[..., Any]], tools)

        _mcp_tools_cache[server_name] = tools_list
        logger.info(f"Loaded {len(tools_list)} tool(s) from MCP server '{server_name}'")
        return tools_list
    except Exception as e:
        logger.warning(
            f"Failed to load tools from MCP server '{server_name}': {e}",
            exc_info=True,
        )
        _mcp_tools_cache[server_name] = []
        return []


def get_mcp_tools(server_name: str) -> List[Callable[..., Any]]:
    """Get MCP tools for a specific server, initializing client if needed (sync).

    Synchronous wrapper for aget_mcp_tools.

    Args:
        server_name: Name of the MCP server to get tools from

    Returns:
        List of callable tools from the specified server
    """
    return asyncio.run(aget_mcp_tools(server_name))


async def aget_all_mcp_tools() -> List[Callable[..., Any]]:
    """Get all tools from all configured MCP servers (async).

    Returns:
        List of all callable tools from all configured servers
    """
    mcp_servers = _get_mcp_servers()
    all_tools = []

    logger.info(f"Loading tools from {len(mcp_servers)} MCP server(s)")

    for server_name in mcp_servers.keys():
        tools = await aget_mcp_tools(server_name)
        all_tools.extend(tools)

    logger.info(f"Total loaded: {len(all_tools)} tool(s) from all MCP servers")
    return all_tools


def get_all_mcp_tools() -> List[Callable[..., Any]]:
    """Get all tools from all configured MCP servers (sync).

    Synchronous wrapper for aget_all_mcp_tools.

    Returns:
        List of all callable tools from all configured servers
    """
    return asyncio.run(aget_all_mcp_tools())


async def aget_mcp_prompts(server_name: str) -> List[Dict[str, Any]]:
    """Get MCP prompts for a specific server (async).

    Args:
        server_name: Name of the MCP server to get prompts from

    Returns:
        List of prompt definitions from the specified server
    """
    global _mcp_prompts_cache

    # Return cached prompts if available
    if server_name in _mcp_prompts_cache:
        logger.debug(f"Returning cached prompts for MCP server '{server_name}'")
        return _mcp_prompts_cache[server_name]

    # Check if server exists in configuration
    mcp_servers = _get_mcp_servers()
    if server_name not in mcp_servers:
        logger.warning(f"MCP server '{server_name}' not found in configuration")
        _mcp_prompts_cache[server_name] = []
        return []

    try:
        # Create a client with only this specific server
        server_config = {server_name: mcp_servers[server_name]}
        client = MultiServerMCPClient(server_config)  # pyright: ignore[reportArgumentType]

        # Get prompts from this server-specific client
        prompts = await client.list_prompts()
        prompts_list = cast(List[Dict[str, Any]], prompts)

        _mcp_prompts_cache[server_name] = prompts_list
        logger.info(
            f"Loaded {len(prompts_list)} prompt(s) from MCP server '{server_name}'"
        )
        return prompts_list
    except Exception as e:
        logger.warning(
            f"Failed to load prompts from MCP server '{server_name}': {e}",
            exc_info=True,
        )
        _mcp_prompts_cache[server_name] = []
        return []


def get_mcp_prompts(server_name: str) -> List[Dict[str, Any]]:
    """Get MCP prompts for a specific server (sync).

    Synchronous wrapper for aget_mcp_prompts.

    Args:
        server_name: Name of the MCP server to get prompts from

    Returns:
        List of prompt definitions from the specified server
    """
    return asyncio.run(aget_mcp_prompts(server_name))


async def aget_all_mcp_prompts() -> List[Dict[str, Any]]:
    """Get all prompts from all configured MCP servers (async).

    Returns:
        List of all prompt definitions from all configured servers
    """
    mcp_servers = _get_mcp_servers()
    all_prompts = []

    logger.info(f"Loading prompts from {len(mcp_servers)} MCP server(s)")

    for server_name in mcp_servers.keys():
        prompts = await aget_mcp_prompts(server_name)
        all_prompts.extend(prompts)

    logger.info(f"Total loaded: {len(all_prompts)} prompt(s) from all MCP servers")
    return all_prompts


def get_all_mcp_prompts() -> List[Dict[str, Any]]:
    """Get all prompts from all configured MCP servers (sync).

    Synchronous wrapper for aget_all_mcp_prompts.

    Returns:
        List of all prompt definitions from all configured servers
    """
    return asyncio.run(aget_all_mcp_prompts())


async def aget_mcp_resources(server_name: str) -> List[Dict[str, Any]]:
    """Get MCP resources for a specific server (async).

    Args:
        server_name: Name of the MCP server to get resources from

    Returns:
        List of resource definitions from the specified server
    """
    global _mcp_resources_cache

    # Return cached resources if available
    if server_name in _mcp_resources_cache:
        logger.debug(f"Returning cached resources for MCP server '{server_name}'")
        return _mcp_resources_cache[server_name]

    # Check if server exists in configuration
    mcp_servers = _get_mcp_servers()
    if server_name not in mcp_servers:
        logger.warning(f"MCP server '{server_name}' not found in configuration")
        _mcp_resources_cache[server_name] = []
        return []

    try:
        # Create a client with only this specific server
        server_config = {server_name: mcp_servers[server_name]}
        client = MultiServerMCPClient(server_config)  # pyright: ignore[reportArgumentType]

        # Get resources from this server-specific client
        resources = await client.list_resources()
        resources_list = cast(List[Dict[str, Any]], resources)

        _mcp_resources_cache[server_name] = resources_list
        logger.info(
            f"Loaded {len(resources_list)} resource(s) from MCP server '{server_name}'"
        )
        return resources_list
    except Exception as e:
        logger.warning(
            f"Failed to load resources from MCP server '{server_name}': {e}",
            exc_info=True,
        )
        _mcp_resources_cache[server_name] = []
        return []


def get_mcp_resources(server_name: str) -> List[Dict[str, Any]]:
    """Get MCP resources for a specific server (sync).

    Synchronous wrapper for aget_mcp_resources.

    Args:
        server_name: Name of the MCP server to get resources from

    Returns:
        List of resource definitions from the specified server
    """
    return asyncio.run(aget_mcp_resources(server_name))


async def aget_all_mcp_resources() -> List[Dict[str, Any]]:
    """Get all resources from all configured MCP servers (async).

    Returns:
        List of all resource definitions from all configured servers
    """
    mcp_servers = _get_mcp_servers()
    all_resources = []

    logger.info(f"Loading resources from {len(mcp_servers)} MCP server(s)")

    for server_name in mcp_servers.keys():
        resources = await aget_mcp_resources(server_name)
        all_resources.extend(resources)

    logger.info(f"Total loaded: {len(all_resources)} resource(s) from all MCP servers")
    return all_resources


def get_all_mcp_resources() -> List[Dict[str, Any]]:
    """Get all resources from all configured MCP servers (sync).

    Synchronous wrapper for aget_all_mcp_resources.

    Returns:
        List of all resource definitions from all configured servers
    """
    return asyncio.run(aget_all_mcp_resources())


def add_mcp_server(name: str, config: Dict[str, Any]) -> None:
    """Add a new MCP server configuration.

    This modifies the runtime configuration only. To persist changes,
    update the config file directly.

    Args:
        name: Server name
        config: Server configuration dict
    """
    global _MCP_SERVERS
    mcp_servers = _get_mcp_servers()
    mcp_servers[name] = config
    logger.info(f"Added MCP server '{name}' to runtime configuration")
    clear_mcp_cache()


def remove_mcp_server(name: str) -> None:
    """Remove an MCP server configuration.

    This modifies the runtime configuration only. To persist changes,
    update the config file directly.

    Args:
        name: Server name to remove
    """
    global _MCP_SERVERS
    mcp_servers = _get_mcp_servers()
    if name in mcp_servers:
        del mcp_servers[name]
        logger.info(f"Removed MCP server '{name}' from runtime configuration")
        clear_mcp_cache()
    else:
        logger.warning(f"MCP server '{name}' not found in configuration")


def clear_mcp_cache() -> None:
    """Clear the MCP client and all caches (useful for testing)."""
    global \
        _mcp_client, \
        _mcp_tools_cache, \
        _mcp_prompts_cache, \
        _mcp_resources_cache, \
        _MCP_SERVERS
    _mcp_client = None
    _mcp_tools_cache = {}
    _mcp_prompts_cache = {}
    _mcp_resources_cache = {}
    _MCP_SERVERS = None  # Force reload from config on next access
    logger.debug("Cleared all MCP caches")
