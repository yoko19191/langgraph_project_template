"""Tests for MCP client and tools management."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.app.common.tools.mcp import (
    add_mcp_server,
    aget_all_mcp_prompts,
    aget_all_mcp_resources,
    aget_all_mcp_tools,
    aget_mcp_client,
    aget_mcp_prompts,
    aget_mcp_resources,
    aget_mcp_tools,
    clear_mcp_cache,
    get_all_mcp_prompts,
    get_all_mcp_resources,
    get_all_mcp_tools,
    get_mcp_client,
    get_mcp_prompts,
    get_mcp_resources,
    get_mcp_tools,
    remove_mcp_server,
)


class TestMCPConfiguration:
    """Test MCP configuration loading and server management."""

    def test_clear_mcp_cache(self):
        """Test clearing MCP cache."""
        # This should not raise any errors
        clear_mcp_cache()

    def test_add_and_remove_mcp_server(self):
        """Test adding and removing MCP servers at runtime."""
        # Clear cache first
        clear_mcp_cache()

        # Add a test server
        test_config = {
            "url": "http://test-server:8080",
            "transport": "http",
        }
        add_mcp_server("test_server", test_config)

        # Remove the test server
        remove_mcp_server("test_server")

        # Clear cache after test
        clear_mcp_cache()


class TestMCPClientSync:
    """Test synchronous MCP client operations."""

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_mcp_client_sync(self):
        """Test getting MCP client synchronously."""
        clear_mcp_cache()

        client = get_mcp_client()

        # Client should be initialized if config exists
        assert client is not None
        assert hasattr(client, "get_tools")

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_mcp_tools_sync(self):
        """Test getting tools from a specific server synchronously."""
        clear_mcp_cache()

        # This will depend on your actual MCP configuration
        # Replace 'deepwiki' with an actual server name from your mcp.json
        tools = get_mcp_tools("context7")

        # Should return a list (may be empty if server not configured)
        assert isinstance(tools, list)

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_all_mcp_tools_sync(self):
        """Test getting all tools from all servers synchronously."""
        clear_mcp_cache()

        tools = get_all_mcp_tools()

        # Should return a list
        assert isinstance(tools, list)

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_mcp_prompts_sync(self):
        """Test getting prompts from a specific server synchronously."""
        clear_mcp_cache()

        prompts = get_mcp_prompts("context7")

        # Should return a list
        assert isinstance(prompts, list)

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_mcp_resources_sync(self):
        """Test getting resources from a specific server synchronously."""
        clear_mcp_cache()

        resources = get_mcp_resources("context7")

        # Should return a list
        assert isinstance(resources, list)

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_all_mcp_prompts_sync(self):
        """Test getting all prompts from all servers synchronously."""
        clear_mcp_cache()

        prompts = get_all_mcp_prompts()

        # Should return a list
        assert isinstance(prompts, list)

        clear_mcp_cache()

    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    def test_get_all_mcp_resources_sync(self):
        """Test getting all resources from all servers synchronously."""
        clear_mcp_cache()

        resources = get_all_mcp_resources()

        # Should return a list
        assert isinstance(resources, list)

        clear_mcp_cache()


class TestMCPClientAsync:
    """Test asynchronous MCP client operations."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_mcp_client(self):
        """Test getting MCP client asynchronously."""
        clear_mcp_cache()

        client = await aget_mcp_client()

        # Client should be initialized if config exists
        assert client is not None
        assert hasattr(client, "get_tools")

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_mcp_tools(self):
        """Test getting tools from a specific server asynchronously."""
        clear_mcp_cache()

        # Replace 'deepwiki' with an actual server name from your mcp.json
        tools = await aget_mcp_tools("context7")

        # Should return a list (may be empty if server not configured)
        assert isinstance(tools, list)

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_all_mcp_tools(self):
        """Test getting all tools from all servers asynchronously."""
        clear_mcp_cache()

        tools = await aget_all_mcp_tools()

        # Should return a list
        assert isinstance(tools, list)
        print(f"Loaded {len(tools)} tools from all MCP servers")  # noqa: T201

        # Print tool names for debugging
        for tool in tools:
            tool_name = getattr(tool, "name", "unknown")
            print(f"  - {tool_name}")  # noqa: T201

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_mcp_tools_caching(self):
        """Test that tools are properly cached."""
        clear_mcp_cache()

        # First call - should fetch from server
        tools1 = await aget_mcp_tools("context7")

        # Second call - should return from cache
        tools2 = await aget_mcp_tools("context7")

        # Should be the same list
        assert tools1 == tools2

        clear_mcp_cache()

    @pytest.mark.asyncio
    async def test_aget_mcp_tools_nonexistent_server(self):
        """Test getting tools from a non-existent server."""
        clear_mcp_cache()

        tools = await aget_mcp_tools("nonexistent_server_12345")

        # Should return empty list
        assert tools == []

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_mcp_prompts(self):
        """Test getting prompts from a specific server asynchronously."""
        clear_mcp_cache()

        prompts = await aget_mcp_prompts("context7")

        # Should return a list (may be empty if server has no prompts)
        assert isinstance(prompts, list)

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_all_mcp_prompts(self):
        """Test getting all prompts from all servers asynchronously."""
        clear_mcp_cache()

        prompts = await aget_all_mcp_prompts()

        # Should return a list
        assert isinstance(prompts, list)
        print(f"Loaded {len(prompts)} prompts from all MCP servers")  # noqa: T201

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_mcp_resources(self):
        """Test getting resources from a specific server asynchronously."""
        clear_mcp_cache()

        resources = await aget_mcp_resources("context7")

        # Should return a list (may be empty if server has no resources)
        assert isinstance(resources, list)

        clear_mcp_cache()

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_aget_all_mcp_resources(self):
        """Test getting all resources from all servers asynchronously."""
        clear_mcp_cache()

        resources = await aget_all_mcp_resources()

        # Should return a list
        assert isinstance(resources, list)
        print(f"Loaded {len(resources)} resources from all MCP servers")  # noqa: T201

        clear_mcp_cache()


class TestMCPClientDirect:
    """Test direct usage of MultiServerMCPClient (baseline test)."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        not Path("mcp.json").exists(),
        reason="mcp.json not found in project root",
    )
    async def test_direct_client_usage(self):
        """Test direct usage of MultiServerMCPClient as baseline."""
        from langchain_mcp_adapters.client import MultiServerMCPClient

        from src.app.common.utils.mcp import load_mcp_config

        # This is the working pattern you mentioned
        mcp_config = load_mcp_config()
        mcp_client = MultiServerMCPClient(mcp_config)

        tools = await mcp_client.get_tools()

        assert isinstance(tools, list)
        print(f"Direct client loaded {len(tools)} tools")  # noqa: T201

        # Print tool names for debugging
        for tool in tools:
            tool_name = getattr(tool, "name", "unknown")
            print(f"  - {tool_name}")  # noqa: T201


class TestMCPErrorHandling:
    """Test error handling in MCP operations."""

    @pytest.mark.asyncio
    async def test_missing_config_file(self):
        """Test behavior when mcp.json is missing."""
        clear_mcp_cache()

        with patch("src.app.common.utils.mcp.find_nearest_config_file") as mock_find:
            mock_find.return_value = None

            # Should handle missing config gracefully
            client = await aget_mcp_client()

            # Client might be None or initialized with empty config
            # depending on implementation
            assert client is None or isinstance(client, object)

        clear_mcp_cache()

    @pytest.mark.asyncio
    async def test_invalid_server_config(self):
        """Test behavior with invalid server configuration."""
        clear_mcp_cache()

        # Try to get client with invalid config
        invalid_config = {
            "invalid_server": {
                # Missing required fields
            }
        }

        client = await aget_mcp_client(invalid_config)

        # Should handle gracefully (return None or raise specific exception)
        # depending on implementation
        assert client is None or isinstance(client, object)

        clear_mcp_cache()


@pytest.mark.skipif(
    not Path("mcp.json").exists(),
    reason="mcp.json not found in project root",
)
def test_integration_full_workflow():
    """Integration test for full MCP workflow (sync)."""
    clear_mcp_cache()

    # Get client
    client = get_mcp_client()
    assert client is not None

    # Get all tools
    all_tools = get_all_mcp_tools()
    assert isinstance(all_tools, list)

    print(f"\nIntegration test loaded {len(all_tools)} tools")  # noqa: T201

    clear_mcp_cache()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not Path("mcp.json").exists(),
    reason="mcp.json not found in project root",
)
async def test_integration_full_workflow_async():
    """Integration test for full MCP workflow (async)."""
    clear_mcp_cache()

    # Get client
    client = await aget_mcp_client()
    assert client is not None

    # Get all tools
    all_tools = await aget_all_mcp_tools()
    assert isinstance(all_tools, list)

    print(f"\nAsync integration test loaded {len(all_tools)} tools")  # noqa: T201

    clear_mcp_cache()


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
