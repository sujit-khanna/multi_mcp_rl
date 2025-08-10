#!/usr/bin/env python3
"""
Simple Shared MCP Tool Manager
==============================

A simpler, more robust implementation that focuses on connection reuse
without complex pooling that causes MCP client issues.

Author: SkyRL Tool Agent Team
Date: 2024-01-31
"""

import os
import sys
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, ClassVar
from pathlib import Path
import threading

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import dotenv_values

try:
    from .retry_utils import RetryableToolExecutor, RetryConfig
except ImportError:
    # Fallback for when module is imported directly
    from retry_utils import RetryableToolExecutor, RetryConfig

# Configure logging
logger = logging.getLogger(__name__)

# Load environment path - use relative path resolution
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
ENV_PATH = PROJECT_ROOT / ".env"
if not ENV_PATH.exists():
    # Try parent directory
    ENV_PATH = PROJECT_ROOT.parent / ".env"

# MCP Server Configuration - use relative paths
MCP_TOOLS_DIR = PROJECT_ROOT / "mcp_tools" / "limited"
MCP_SERVERS = {
    "slack": str(MCP_TOOLS_DIR / "slack_limited_server.py"),
    "tavily": str(MCP_TOOLS_DIR / "tavily_limited_server.py"),
    "polygon": str(MCP_TOOLS_DIR / "polygon_limited_server.py"),
    "fmp": str(MCP_TOOLS_DIR / "fmp_limited_server.py"),
    "python": str(MCP_TOOLS_DIR / "python_execution_server.py")
}


class SimpleSharedMCPToolManager:
    """
    Simple shared tool manager that pre-initializes tool discovery
    but uses fresh connections for each tool call to avoid MCP client issues.
    
    The performance gain comes from:
    1. Shared tool discovery (no re-initialization of servers)
    2. Retry logic with exponential backoff
    3. Single manager instance across all evaluations
    """
    
    # Singleton instance
    _instance: ClassVar[Optional['SimpleSharedMCPToolManager']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, retry_config: Optional[RetryConfig] = None):
        """Initialize shared tool manager (only runs once due to singleton)"""
        # Skip if already initialized
        if hasattr(self, '_initialized'):
            return
            
        self.retry_config = retry_config or RetryConfig()
        self.retry_executor = RetryableToolExecutor(self.retry_config)
        
        # Tool registry
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.failed_servers: set = set()
        
        # Performance metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retried_calls": 0,
            "initialization_reuses": 0
        }
        
        self._initialized = False
        self._init_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the shared tool manager"""
        async with self._init_lock:
            if self._initialized:
                logger.debug("Simple shared tool manager already initialized")
                self.metrics["initialization_reuses"] += 1
                return
            
            logger.info("Initializing simple shared MCP tool manager")
            
            # Discover tools from all servers (one time setup)
            await self._discover_tools()
            
            self._initialized = True
            
            # Log initialization summary
            logger.info(f"Simple shared tool manager initialized:")
            logger.info(f"  - Tools: {len(self.available_tools)}")
            logger.info(f"  - Failed servers: {list(self.failed_servers)}")
    
    async def _discover_tools(self):
        """Discover tools from all servers (one-time setup)"""
        for server_name, server_path in MCP_SERVERS.items():
            try:
                env_vars = dict(os.environ)
                env_vars.update(dotenv_values(str(ENV_PATH)))
                
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[os.path.basename(server_path)],
                    env=env_vars,
                    cwd=os.path.dirname(server_path)
                )
                
                # Use the same pattern as working RealMCPToolManager
                async with stdio_client(params) as (r, w):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        
                        for tool in tools.tools:
                            self.available_tools[tool.name] = {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.inputSchema,
                                "server": server_name
                            }
                        
                        logger.info(f"Discovered {len(tools.tools)} tools from {server_name}")
                        
            except Exception as e:
                logger.error(f"Failed to discover tools from {server_name}: {e}")
                self.failed_servers.add(server_name)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return list(self.available_tools.values())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute a tool with retry logic"""
        if not self._initialized:
            await self.initialize()
        
        self.metrics["total_calls"] += 1
        
        # Execute with retry
        success, result = await self.retry_executor.execute_with_retry(
            self._execute_tool_fresh_connection,
            tool_name,
            arguments
        )
        
        # Update metrics
        if success:
            self.metrics["successful_calls"] += 1
        else:
            self.metrics["failed_calls"] += 1
        
        retry_attempts = result.get("retry_attempts", 0)
        if retry_attempts > 0:
            self.metrics["retried_calls"] += 1
        
        return success, result
    
    async def _execute_tool_fresh_connection(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute tool using fresh connection (same as RealMCPToolManager)"""
        # Check if tool exists
        tool_info = self.available_tools.get(tool_name)
        if not tool_info:
            return False, {"error": f"Tool '{tool_name}' not found"}
        
        server_name = tool_info["server"]
        
        # Check if server failed during discovery
        if server_name in self.failed_servers:
            return False, {"error": f"Server '{server_name}' is unavailable", "retryable": False}
        
        server_path = MCP_SERVERS.get(server_name)
        if not server_path:
            return False, {"error": f"Server '{server_name}' not configured", "retryable": False}
        
        # Use fresh connection (same as RealMCPToolManager for reliability)
        env_vars = dict(os.environ)
        env_vars.update(dotenv_values(str(ENV_PATH)))
        
        params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        
        try:
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments), 
                        timeout=10.0  # Reduced timeout to prevent hanging
                    )
                    
                    text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                    return True, {"content": text}
                    
        except asyncio.TimeoutError:
            return False, {"error": "timeout", "retryable": True}
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            return False, {"error": str(e), "retryable": True}
    
    def execute_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Synchronous wrapper for tool execution"""
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop - this is problematic for sync execution
                # Create a new thread to run the async function
                import concurrent.futures
                import threading
                
                result_container = {}
                exception_container = {}
                
                def run_in_thread():
                    try:
                        # Create a new event loop in this thread
                        new_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(new_loop)
                        try:
                            success, result = new_loop.run_until_complete(
                                self.execute_tool(tool_name, arguments)
                            )
                            result_container['result'] = (success, result)
                        finally:
                            new_loop.close()
                    except Exception as e:
                        exception_container['error'] = e
                
                thread = threading.Thread(target=run_in_thread)
                thread.start()
                thread.join(timeout=30)  # Reduced timeout
                
                if thread.is_alive():
                    raise TimeoutError(f"Tool execution timed out after 30 seconds")
                
                if 'error' in exception_container:
                    raise exception_container['error']
                
                if 'result' not in result_container:
                    raise RuntimeError("Tool execution completed but no result returned")
                
                success, result = result_container['result']
                
            except RuntimeError as e:
                if "no running event loop" in str(e):
                    # No event loop running - create one
                    success, result = asyncio.run(self.execute_tool(tool_name, arguments))
                else:
                    raise
            
            if success:
                return result.get("content", "")
            else:
                raise RuntimeError(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Sync execution failed for {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup resources (mainly for metrics logging)"""
        logger.info("Cleaning up simple shared tool manager")
        
        # Log final metrics
        if self.metrics['total_calls'] > 0:
            success_rate = self.metrics['successful_calls'] / self.metrics['total_calls']
            retry_rate = self.metrics['retried_calls'] / self.metrics['total_calls']
            
            logger.info(f"Final metrics:")
            logger.info(f"  - Total calls: {self.metrics['total_calls']}")
            logger.info(f"  - Success rate: {success_rate:.2%}")
            logger.info(f"  - Retry rate: {retry_rate:.2%}")
            logger.info(f"  - Initialization reuses: {self.metrics['initialization_reuses']}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "total_calls": self.metrics["total_calls"],
            "success_rate": 0.0,
            "retry_rate": 0.0,
            "initialization_reuses": self.metrics["initialization_reuses"]
        }
        
        if self.metrics["total_calls"] > 0:
            stats["success_rate"] = self.metrics["successful_calls"] / self.metrics["total_calls"]
            stats["retry_rate"] = self.metrics["retried_calls"] / self.metrics["total_calls"]
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            "initialized": self._initialized,
            "metrics": self.metrics.copy(),
            "available_tools": len(self.available_tools),
            "failed_servers": list(self.failed_servers)
        }
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)"""
        with cls._lock:
            if cls._instance:
                # Simple cleanup
                try:
                    asyncio.run(cls._instance.cleanup())
                except:
                    pass
            cls._instance = None


# Convenience function to get or create the shared instance
def get_simple_shared_tool_manager(retry_config: Optional[RetryConfig] = None) -> SimpleSharedMCPToolManager:
    """Get or create the simple shared tool manager instance."""
    return SimpleSharedMCPToolManager(retry_config)

# Alias for backward compatibility with training scripts
SimpleSharedManager = SimpleSharedMCPToolManager