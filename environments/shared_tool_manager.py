#!/usr/bin/env python3
"""
Shared MCP Tool Manager
=======================

Singleton tool manager with connection pooling, retry logic, and health monitoring
for efficient tool execution during training.

Author: SkyRL Tool Agent Team
Date: 2024-01-31
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, ClassVar
from pathlib import Path
import threading

from .connection_pool import ConnectionPool, PoolConfig
from .retry_utils import (
    RetryableToolExecutor, RetryConfig, ServerHealthMonitor,
    CircuitBreakerConfig, ServerHealthMetrics
)

# Configure logging
logger = logging.getLogger(__name__)

# Load environment path
ENV_PATH = "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env"

# MCP Server Configuration (same as real_tool_manager.py)
MCP_SERVERS = {
    "slack": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/slack_limited_server.py",
    "tavily": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/tavily_limited_server.py",
    "polygon": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/polygon_limited_server.py",
    "fmp": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/fmp_limited_server.py",
    "python": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/python_execution_server.py"
}


class SharedMCPToolManager:
    """
    Shared singleton tool manager with connection pooling and retry logic.
    
    This manager maintains persistent connections to MCP servers and reuses them
    across multiple tool calls, significantly reducing overhead during training.
    """
    
    # Singleton instance
    _instance: ClassVar[Optional['SharedMCPToolManager']] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        pool_config: Optional[PoolConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    ):
        """
        Initialize shared tool manager (only runs once due to singleton)
        
        Args:
            pool_config: Configuration for connection pooling
            retry_config: Configuration for retry logic
            circuit_breaker_config: Configuration for circuit breakers
        """
        # Skip if already initialized
        if hasattr(self, '_initialized'):
            return
            
        self.pool_config = pool_config or PoolConfig()
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Core components
        self.connection_pool = ConnectionPool(ENV_PATH, self.pool_config)
        self.retry_executor = RetryableToolExecutor(self.retry_config)
        self.health_monitor = ServerHealthMonitor()
        
        # Tool registry
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        
        # State tracking
        self._initialized = False
        self._init_lock = asyncio.Lock()
        
        # Performance metrics
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retried_calls": 0,
            "connection_reuses": 0
        }
    
    async def initialize(self):
        """Initialize the shared tool manager"""
        async with self._init_lock:
            if self._initialized:
                logger.debug("Shared tool manager already initialized")
                return
            
            logger.info("Initializing shared MCP tool manager with connection pooling")
            
            # Initialize connection pool
            await self.connection_pool.initialize(MCP_SERVERS)
            
            # Copy available tools from pool
            self.available_tools = self.connection_pool.available_tools.copy()
            
            # Register servers with health monitor
            for server_name in MCP_SERVERS.keys():
                self.health_monitor.register_server(
                    server_name,
                    self.circuit_breaker_config,
                    self._create_health_check_callback(server_name)
                )
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            self._initialized = True
            
            # Log initialization summary
            stats = self.connection_pool.get_pool_stats()
            logger.info(f"Shared tool manager initialized:")
            logger.info(f"  - Servers: {len(stats['servers'])}")
            logger.info(f"  - Connections: {stats['total_connections']}")
            logger.info(f"  - Tools: {stats['total_tools']}")
            logger.info(f"  - Failed servers: {stats['failed_servers']}")
    
    def _create_health_check_callback(self, server_name: str):
        """Create a health check callback for a server"""
        async def health_check(name: str) -> bool:
            try:
                # Try to get connection and list tools
                async with self.connection_pool.get_connection(name) as session:
                    tools = await asyncio.wait_for(
                        session.list_tools(),
                        timeout=5.0
                    )
                    return len(tools.tools) > 0
            except Exception as e:
                logger.debug(f"Health check failed for {name}: {e}")
                return False
        
        return health_check
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of all available tools"""
        return list(self.available_tools.values())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a tool with connection pooling, retry logic, and health monitoring.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Success flag and result/error dict
        """
        if not self._initialized:
            await self.initialize()
        
        self.metrics["total_calls"] += 1
        
        # Find server for tool
        tool_info = self.available_tools.get(tool_name)
        if not tool_info:
            logger.error(f"Tool '{tool_name}' not found")
            self.metrics["failed_calls"] += 1
            return False, {"error": f"Tool '{tool_name}' not found"}
        
        server_name = tool_info["server"]
        
        # Check circuit breaker
        circuit_breaker = self.health_monitor.circuit_breakers.get(server_name)
        if circuit_breaker:
            try:
                # Execute with circuit breaker protection
                success, result = await circuit_breaker.call(
                    self._execute_with_retry,
                    tool_name,
                    arguments
                )
                
                # Record health metrics
                await self.health_monitor.record_call_result(server_name, success)
                
                if success:
                    self.metrics["successful_calls"] += 1
                else:
                    self.metrics["failed_calls"] += 1
                
                return success, result
                
            except Exception as e:
                logger.error(f"Circuit breaker error for {server_name}: {e}")
                self.metrics["failed_calls"] += 1
                await self.health_monitor.record_call_result(server_name, False)
                return False, {"error": str(e), "circuit_breaker": True}
        else:
            # No circuit breaker, execute directly with retry
            success, result = await self._execute_with_retry(tool_name, arguments)
            
            if success:
                self.metrics["successful_calls"] += 1
            else:
                self.metrics["failed_calls"] += 1
            
            return success, result
    
    async def _execute_with_retry(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute tool with retry logic"""
        # Use retry executor
        success, result = await self.retry_executor.execute_with_retry(
            self.connection_pool.execute_tool,
            tool_name,
            arguments
        )
        
        # Track retries
        retry_attempts = result.get("retry_attempts", 0)
        if retry_attempts > 0:
            self.metrics["retried_calls"] += 1
            logger.info(f"Tool {tool_name} succeeded after {retry_attempts} retries")
        
        # Track connection reuse (approximation based on pool stats)
        pool_stats = self.connection_pool.get_pool_stats()
        total_uses = sum(
            server_stats.get("total_uses", 0)
            for server_stats in pool_stats.get("servers", {}).values()
        )
        if total_uses > self.metrics["total_calls"]:
            self.metrics["connection_reuses"] = total_uses - self.metrics["total_calls"]
        
        return success, result
    
    def execute_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Synchronous wrapper for tool execution (for backward compatibility)
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            str: Tool execution result
            
        Raises:
            RuntimeError: If tool execution fails
        """
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - use run_coroutine_threadsafe
                import concurrent.futures
                
                future = asyncio.run_coroutine_threadsafe(
                    self.execute_tool(tool_name, arguments),
                    loop
                )
                success, result = future.result(timeout=30)
                
            except RuntimeError:
                # No event loop running - create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success, result = loop.run_until_complete(
                        self.execute_tool(tool_name, arguments)
                    )
                finally:
                    loop.close()
            
            if success:
                return result.get("content", "")
            else:
                raise RuntimeError(f"Tool execution failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Sync execution failed for {tool_name}: {e}")
            raise
    
    async def cleanup(self):
        """
        Cleanup resources (only when completely done with the manager)
        
        Note: In normal usage, you should keep the shared manager alive
        throughout your training run for maximum efficiency.
        """
        logger.info("Cleaning up shared tool manager")
        
        # Stop health monitoring
        await self.health_monitor.stop_monitoring()
        
        # Close connection pool
        await self.connection_pool.close()
        
        # Log final metrics
        logger.info(f"Final metrics:")
        logger.info(f"  - Total calls: {self.metrics['total_calls']}")
        logger.info(f"  - Successful: {self.metrics['successful_calls']}")
        logger.info(f"  - Failed: {self.metrics['failed_calls']}")
        logger.info(f"  - Retried: {self.metrics['retried_calls']}")
        logger.info(f"  - Connection reuses: {self.metrics['connection_reuses']}")
        
        if self.metrics['total_calls'] > 0:
            success_rate = self.metrics['successful_calls'] / self.metrics['total_calls']
            reuse_rate = self.metrics['connection_reuses'] / self.metrics['total_calls']
            logger.info(f"  - Success rate: {success_rate:.2%}")
            logger.info(f"  - Connection reuse rate: {reuse_rate:.2%}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            "initialized": self._initialized,
            "metrics": self.metrics.copy(),
            "pool_stats": self.connection_pool.get_pool_stats() if self._initialized else {},
            "server_health": self.health_monitor.get_all_server_health() if self._initialized else {}
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            "total_calls": self.metrics["total_calls"],
            "success_rate": 0.0,
            "retry_rate": 0.0,
            "connection_reuse_rate": 0.0,
            "average_connections_per_server": 0.0
        }
        
        if self.metrics["total_calls"] > 0:
            stats["success_rate"] = self.metrics["successful_calls"] / self.metrics["total_calls"]
            stats["retry_rate"] = self.metrics["retried_calls"] / self.metrics["total_calls"]
            stats["connection_reuse_rate"] = self.metrics["connection_reuses"] / self.metrics["total_calls"]
        
        if self._initialized:
            pool_stats = self.connection_pool.get_pool_stats()
            if pool_stats["servers"]:
                total_conns = sum(s["total"] for s in pool_stats["servers"].values())
                stats["average_connections_per_server"] = total_conns / len(pool_stats["servers"])
        
        return stats
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance (mainly for testing)
        
        Warning: This will close all connections. Use with caution.
        """
        with cls._lock:
            if cls._instance and hasattr(cls._instance, '_initialized'):
                # Run cleanup in a new event loop if needed
                try:
                    loop = asyncio.get_running_loop()
                    asyncio.create_task(cls._instance.cleanup())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(cls._instance.cleanup())
                    loop.close()
            
            cls._instance = None


# Convenience function to get or create the shared instance
def get_shared_tool_manager(
    pool_config: Optional[PoolConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
) -> SharedMCPToolManager:
    """
    Get or create the shared tool manager instance.
    
    This is the recommended way to access the shared tool manager.
    """
    return SharedMCPToolManager(pool_config, retry_config, circuit_breaker_config)