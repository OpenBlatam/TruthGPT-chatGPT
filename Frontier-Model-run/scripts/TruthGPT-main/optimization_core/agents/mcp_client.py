"""
OpenClaw MCP Client.
Support for Anthropic's Model Context Protocol (MCP).
Allows agents to discover and use tools/resources from external MCP servers.
"""

import json
import logging
import httpx
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for interacting with MCP (Model Context Protocol) servers.
    Provides methods to list tools and call them in a standardized way.
    """
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Fetch available tools from the MCP server."""
        try:
            response = await self.client.post(
                f"{self.server_url}/tools/list",
                json={}
            )
            response.raise_for_status()
            return response.json().get("tools", [])
        except Exception as e:
            logger.error(f"Failed to list MCP tools from {self.server_url}: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute an MCP tool."""
        try:
            response = await self.client.post(
                f"{self.server_url}/tools/call",
                json={"name": tool_name, "arguments": arguments}
            )
            response.raise_for_status()
            return response.json().get("content", "No output.")
        except Exception as e:
            logger.error(f"Failed to call MCP tool '{tool_name}': {e}")
            return f"Error: {str(e)}"

    async def list_resources(self) -> List[Dict[str, Any]]:
        """Fetch available resources from the MCP server."""
        try:
            response = await self.client.get(f"{self.server_url}/resources/list")
            response.raise_for_status()
            return response.json().get("resources", [])
        except Exception as e:
            logger.error(f"Failed to list MCP resources: {e}")
            return []

    async def close(self):
        """Clean up the httpx client."""
        await self.client.aclose()

