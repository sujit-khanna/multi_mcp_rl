#!/usr/bin/env python3
"""
Real MCP Tool Manager
====================

Real tool manager that properly integrates with MCP servers.
Based on the working implementation from mini_agent_trajectories.py

Author: SkyRL Tool Agent Team
Date: 2024-01-29
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv, dotenv_values

# Configure logging
logger = logging.getLogger(__name__)

# Load environment
env_path = "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env"
load_dotenv(env_path, override=True)

# MCP Server Configuration
MCP_SERVERS = {
    "slack": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/slack_limited_server.py",
    "tavily": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/tavily_limited_server.py",
    "polygon": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/polygon_limited_server.py", 
    "fmp": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/fmp_limited_server.py",
    "python": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/python_execution_server.py"
}


class RealMCPToolManager:
    """
    Real MCP tool manager based on the working implementation from mini_agent_trajectories.py
    """
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.failed_servers: Set[str] = set()
        
    async def initialize(self):
        """Initialize tools from all MCP servers"""
        logger.info("Initializing real MCP tools...")
        
        for server_name, server_script in MCP_SERVERS.items():
            server_path = os.path.join(os.getcwd(), server_script)
            try:
                await self._initialize_single_server(server_name, server_path)
            except Exception as e:
                logger.error(f"Failed to init {server_name}: {e}")
                self.failed_servers.add(server_name)
        
        # Log availability summary
        total_servers = len(MCP_SERVERS)
        available_servers = total_servers - len(self.failed_servers)
        logger.info(f"Loaded {len(self.tools)} tools from {available_servers}/{total_servers} servers")
        if self.failed_servers:
            logger.warning(f"Failed servers: {', '.join(sorted(self.failed_servers))}")

    async def _initialize_single_server(self, server_name: str, server_path: str):
        """Initialize a single server and get its tools"""
        env_vars = dict(os.environ)
        env_vars.update(dotenv_values(env_path))
        
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        
        async with stdio_client(server_params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                for tool in (await session.list_tools()).tools:
                    self.tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                        "server": server_name
                    }
                logger.info(f"Initialized {server_name} with tools: {[t.name for t in (await session.list_tools()).tools]}")

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return list(self.tools.values())
    
    def _get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server alias for a given tool name"""
        # Direct mapping based on tool name prefixes and patterns
        for server_alias in MCP_SERVERS.keys():
            if tool_name.startswith(server_alias) or server_alias in tool_name.lower():
                return server_alias
        
        # Check the stored tool info
        tool_info = self.tools.get(tool_name)
        if tool_info and 'server' in tool_info:
            return tool_info['server']
            
        return None

    def _is_tool_from_failed_server(self, tool_name: str) -> bool:
        """Check if a tool belongs to a failed server"""
        server_alias = self._get_server_for_tool(tool_name)
        return server_alias in self.failed_servers if server_alias else False

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute a tool using real MCP server"""
        logger.debug(f"Looking up tool: {tool_name}")
        
        # Check if tool belongs to a failed server
        if self._is_tool_from_failed_server(tool_name):
            server_alias = self._get_server_for_tool(tool_name)
            logger.warning(f"Tool '{tool_name}' is disabled due to server '{server_alias}' failure")
            return False, {"error": f"Tool '{tool_name}' is disabled (server '{server_alias}' failed)", "retryable": False}
        
        info = self.tools.get(tool_name)
        if not info:
            logger.error(f"Tool '{tool_name}' not found in available tools")
            return False, {"error": f"Tool '{tool_name}' not found"}
        
        # Find which server this tool belongs to
        server_alias = self._get_server_for_tool(tool_name)
        if not server_alias:
            logger.error(f"Could not determine server for tool '{tool_name}'")
            return False, {"error": f"Tool '{tool_name}' has no matching server", "retryable": False}
        
        server_script = MCP_SERVERS.get(server_alias)
        if not server_script:
            logger.error(f"Server script not found for alias '{server_alias}'")
            return False, {"error": f"Server '{server_alias}' not configured", "retryable": False}
        
        logger.debug(f"Using server script: {server_script} for server alias: {server_alias}")
        server_path = os.path.join(os.getcwd(), server_script)
        
        env_vars = dict(os.environ)
        env_vars.update(dotenv_values(env_path))
        
        params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        
        try:
            logger.debug(f"Connecting to server at {server_path}")
            async with stdio_client(params) as (r, w):
                logger.debug("Connected to server, initializing session")
                async with ClientSession(r, w) as session:
                    logger.debug("Session initialized, calling tool")
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments), timeout=30.0
                    )
                    logger.debug("Tool call completed successfully")
                    text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                    return True, {"content": text}
                    
        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} timed out after 30 seconds")
            return False, {"error": "timeout", "retryable": True}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed with error: {e}")
            return False, {"error": str(e), "retryable": False}
    
    def execute_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronous wrapper for tool execution"""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use run_coroutine_threadsafe
                import concurrent.futures
                import threading
                
                # Create a new thread to run the async code
                future = concurrent.futures.Future()
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        success, result = new_loop.run_until_complete(self.execute_tool(tool_name, arguments))
                        future.set_result((success, result))
                    except Exception as e:
                        future.set_exception(e)
                    finally:
                        new_loop.close()
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=30)
                
                if not future.done():
                    raise TimeoutError(f"Tool {tool_name} timed out")
                
                success, result = future.result()
                
            except RuntimeError:
                # No event loop running - create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                success, result = loop.run_until_complete(self.execute_tool(tool_name, arguments))
                loop.close()
            
            if success:
                return result.get("content", "")
            else:
                raise RuntimeError(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Sync execution failed for {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup - nothing to do since we create sessions on demand"""
        logger.info("Cleanup complete (sessions are created on demand)")


# Alias for compatibility
MCPToolManager = RealMCPToolManager