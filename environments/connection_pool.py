#!/usr/bin/env python3
"""
Connection Pool
===============

Manages persistent connections to MCP servers with efficient pooling
and lifecycle management.

Author: SkyRL Tool Agent Team  
Date: 2024-01-31
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import time
from pathlib import Path

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import dotenv_values

logger = logging.getLogger(__name__)


@dataclass
class ConnectionInfo:
    """Information about a pooled connection"""
    server_name: str
    server_path: str
    session: ClientSession
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    is_healthy: bool = True
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    @property
    def age(self) -> timedelta:
        """Get connection age"""
        return datetime.now() - self.created_at
    
    @property
    def idle_time(self) -> timedelta:
        """Get time since last use"""
        return datetime.now() - self.last_used
    
    def mark_used(self):
        """Mark connection as used"""
        self.last_used = datetime.now()
        self.use_count += 1


@dataclass
class PoolConfig:
    """Configuration for connection pool"""
    min_connections: int = 1          # Minimum connections per server
    max_connections: int = 3          # Maximum connections per server
    max_connection_age: timedelta = timedelta(minutes=30)  # Max age before recycling
    max_idle_time: timedelta = timedelta(minutes=5)       # Max idle before closing
    connection_timeout: float = 30.0  # Timeout for establishing connection
    health_check_interval: float = 60.0  # Interval for health checks
    retry_failed_init: bool = True    # Retry failed server initialization
    retry_interval: float = 60.0      # Interval between retry attempts


class ConnectionPool:
    """
    Manages persistent connections to MCP servers with pooling
    """
    
    def __init__(self, env_path: str, config: Optional[PoolConfig] = None):
        self.env_path = env_path
        self.config = config or PoolConfig()
        self.connections: Dict[str, List[ConnectionInfo]] = {}
        self.available_tools: Dict[str, Dict[str, Any]] = {}
        self.failed_servers: Set[str] = set()
        self.server_configs: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._maintenance_task: Optional[asyncio.Task] = None
        self._is_initialized = False
        self._retry_tasks: Dict[str, asyncio.Task] = {}
        
    async def initialize(self, servers: Dict[str, str]):
        """
        Initialize connection pool with server configurations
        
        Args:
            servers: Dict mapping server names to script paths
        """
        if self._is_initialized:
            logger.warning("Connection pool already initialized")
            return
            
        self.server_configs = servers
        logger.info(f"Initializing connection pool for {len(servers)} servers")
        
        # Initialize connections for each server
        init_tasks = []
        for server_name, server_path in servers.items():
            init_tasks.append(self._initialize_server_pool(server_name, server_path))
        
        # Wait for all initializations
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Log results
        for i, (server_name, result) in enumerate(zip(servers.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to initialize {server_name}: {result}")
                self.failed_servers.add(server_name)
                if self.config.retry_failed_init:
                    self._schedule_retry(server_name)
            else:
                logger.info(f"Successfully initialized {server_name}")
        
        # Start maintenance task
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        self._is_initialized = True
        
        # Log summary
        successful = len(servers) - len(self.failed_servers)
        logger.info(f"Connection pool initialized: {successful}/{len(servers)} servers ready")
    
    async def _initialize_server_pool(self, server_name: str, server_path: str):
        """Initialize connection pool for a single server"""
        try:
            # Create minimum connections
            for _ in range(self.config.min_connections):
                await self._create_connection(server_name, server_path)
                
            # Get available tools from first connection
            if server_name in self.connections and self.connections[server_name]:
                conn = self.connections[server_name][0]
                async with conn.lock:
                    tools = await conn.session.list_tools()
                    for tool in tools.tools:
                        self.available_tools[tool.name] = {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                            "server": server_name
                        }
                logger.info(f"Loaded {len(tools.tools)} tools from {server_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize pool for {server_name}: {e}")
            raise
    
    async def _create_connection(self, server_name: str, server_path: str) -> ConnectionInfo:
        """Create a new connection to a server"""
        env_vars = dict(os.environ)
        env_vars.update(dotenv_values(self.env_path))
        
        params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        
        # Create connection with timeout - using the same approach as RealMCPToolManager
        start_time = time.time()
        try:
            # Use the same pattern as the working RealMCPToolManager
            client_context = stdio_client(params)
            reader, writer = await client_context.__aenter__()
            
            # Create session
            session = ClientSession(reader, writer)
            await asyncio.wait_for(
                session.initialize(),
                timeout=self.config.connection_timeout
            )
            
            # Create connection info - store the context manager for proper cleanup
            conn_info = ConnectionInfo(
                server_name=server_name,
                server_path=server_path,
                session=session,
                reader=reader,
                writer=writer,
                created_at=datetime.now(),
                last_used=datetime.now()
            )
            
            # Store the context manager for cleanup
            conn_info._client_context = client_context
            
            # Add to pool
            async with self._lock:
                if server_name not in self.connections:
                    self.connections[server_name] = []
                self.connections[server_name].append(conn_info)
            
            elapsed = time.time() - start_time
            logger.debug(f"Created connection to {server_name} in {elapsed:.2f}s")
            
            return conn_info
            
        except asyncio.TimeoutError:
            logger.error(f"Timeout creating connection to {server_name} after {self.config.connection_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Error creating connection to {server_name}: {e}")
            raise
    
    @asynccontextmanager
    async def get_connection(self, server_name: str):
        """
        Get a connection from the pool with context manager
        
        Usage:
            async with pool.get_connection("slack") as session:
                result = await session.call_tool("tool_name", args)
        """
        if server_name in self.failed_servers:
            raise ConnectionError(f"Server {server_name} is marked as failed")
        
        conn = await self._acquire_connection(server_name)
        try:
            yield conn.session
        finally:
            await self._release_connection(conn)
    
    async def _acquire_connection(self, server_name: str) -> ConnectionInfo:
        """Acquire a connection from the pool"""
        # Try to get an existing connection
        async with self._lock:
            if server_name in self.connections:
                # Find an available healthy connection
                for conn in self.connections[server_name]:
                    if conn.is_healthy and not conn.lock.locked():
                        await conn.lock.acquire()
                        conn.mark_used()
                        return conn
        
        # No available connection, create new one if under limit
        async with self._lock:
            current_count = len(self.connections.get(server_name, []))
            if current_count < self.config.max_connections:
                # Create new connection
                server_path = self.server_configs.get(server_name)
                if not server_path:
                    raise ValueError(f"No configuration for server {server_name}")
                
                try:
                    conn = await self._create_connection(server_name, server_path)
                    await conn.lock.acquire()
                    conn.mark_used()
                    return conn
                except Exception as e:
                    logger.error(f"Failed to create new connection for {server_name}: {e}")
                    raise
        
        # Wait for a connection to become available
        logger.info(f"Waiting for available connection to {server_name}")
        while True:
            async with self._lock:
                for conn in self.connections.get(server_name, []):
                    if conn.is_healthy and not conn.lock.locked():
                        await conn.lock.acquire()
                        conn.mark_used()
                        return conn
            
            await asyncio.sleep(0.1)
    
    async def _release_connection(self, conn: ConnectionInfo):
        """Release a connection back to the pool"""
        conn.lock.release()
        
        # Check if connection needs recycling
        if (conn.age > self.config.max_connection_age or 
            not await self._check_connection_health(conn)):
            await self._recycle_connection(conn)
    
    async def _check_connection_health(self, conn: ConnectionInfo) -> bool:
        """Check if a connection is healthy"""
        try:
            # Simple health check - list tools with timeout
            await asyncio.wait_for(
                conn.session.list_tools(),
                timeout=5.0
            )
            conn.is_healthy = True
            return True
        except Exception as e:
            logger.warning(f"Health check failed for {conn.server_name}: {e}")
            conn.is_healthy = False
            return False
    
    async def _recycle_connection(self, conn: ConnectionInfo):
        """Recycle an old or unhealthy connection"""
        logger.info(f"Recycling connection to {conn.server_name} (age: {conn.age}, healthy: {conn.is_healthy})")
        
        async with self._lock:
            # Remove from pool
            if conn.server_name in self.connections:
                self.connections[conn.server_name].remove(conn)
        
        # Close connection
        try:
            conn.writer.close()
            await conn.writer.wait_closed()
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
        
        # Create replacement if below minimum
        async with self._lock:
            current_count = len(self.connections.get(conn.server_name, []))
            if current_count < self.config.min_connections:
                try:
                    await self._create_connection(conn.server_name, conn.server_path)
                except Exception as e:
                    logger.error(f"Failed to create replacement connection: {e}")
    
    async def _maintenance_loop(self):
        """Background maintenance of connection pool"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._perform_maintenance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in maintenance loop: {e}")
    
    async def _perform_maintenance(self):
        """Perform maintenance tasks on the pool"""
        logger.debug("Performing connection pool maintenance")
        
        # Check all connections
        async with self._lock:
            all_connections = []
            for server_conns in self.connections.values():
                all_connections.extend(server_conns)
        
        # Health check and cleanup
        for conn in all_connections:
            # Skip locked connections
            if conn.lock.locked():
                continue
                
            # Check idle timeout
            if conn.idle_time > self.config.max_idle_time:
                logger.info(f"Closing idle connection to {conn.server_name}")
                await self._recycle_connection(conn)
                continue
            
            # Health check
            if not await self._check_connection_health(conn):
                await self._recycle_connection(conn)
    
    def _schedule_retry(self, server_name: str):
        """Schedule retry for failed server initialization"""
        if server_name in self._retry_tasks:
            return  # Already scheduled
            
        async def retry_init():
            await asyncio.sleep(self.config.retry_interval)
            logger.info(f"Retrying initialization for {server_name}")
            
            try:
                server_path = self.server_configs[server_name]
                await self._initialize_server_pool(server_name, server_path)
                self.failed_servers.discard(server_name)
                logger.info(f"Successfully recovered {server_name}")
            except Exception as e:
                logger.error(f"Retry failed for {server_name}: {e}")
                # Schedule another retry
                self._schedule_retry(server_name)
            finally:
                self._retry_tasks.pop(server_name, None)
        
        self._retry_tasks[server_name] = asyncio.create_task(retry_init())
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Execute a tool using pooled connection"""
        # Find server for tool
        tool_info = self.available_tools.get(tool_name)
        if not tool_info:
            return False, {"error": f"Tool '{tool_name}' not found"}
        
        server_name = tool_info["server"]
        
        try:
            async with self.get_connection(server_name) as session:
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments),
                    timeout=30.0
                )
                
                text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                return True, {"content": text}
                
        except asyncio.TimeoutError:
            return False, {"error": "timeout", "retryable": True}
        except ConnectionError as e:
            return False, {"error": str(e), "retryable": False}
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return False, {"error": str(e), "retryable": True}
    
    async def close(self):
        """Close all connections and cleanup"""
        logger.info("Closing connection pool")
        
        # Cancel maintenance task
        if self._maintenance_task and not self._maintenance_task.done():
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        # Cancel retry tasks
        for task in self._retry_tasks.values():
            if not task.done():
                task.cancel()
        
        # Close all connections
        async with self._lock:
            for server_name, conns in self.connections.items():
                for conn in conns:
                    try:
                        conn.writer.close()
                        await conn.writer.wait_closed()
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
            
            self.connections.clear()
        
        logger.info("Connection pool closed")
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the connection pool"""
        stats = {
            "servers": {},
            "total_connections": 0,
            "healthy_connections": 0,
            "failed_servers": list(self.failed_servers),
            "total_tools": len(self.available_tools)
        }
        
        for server_name, conns in self.connections.items():
            healthy = sum(1 for c in conns if c.is_healthy)
            stats["servers"][server_name] = {
                "total": len(conns),
                "healthy": healthy,
                "locked": sum(1 for c in conns if c.lock.locked()),
                "avg_age_minutes": sum(c.age.total_seconds() for c in conns) / 60 / len(conns) if conns else 0,
                "total_uses": sum(c.use_count for c in conns)
            }
            stats["total_connections"] += len(conns)
            stats["healthy_connections"] += healthy
        
        return stats