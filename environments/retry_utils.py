#!/usr/bin/env python3
"""
Retry Utilities
===============

Provides retry logic with exponential backoff and circuit breaker patterns
for robust tool execution.

Author: SkyRL Tool Agent Team
Date: 2024-01-31
"""

import asyncio
import time
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls  
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class RetryConfig:
    """Configuration for retry logic"""
    max_attempts: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 16.0     # seconds
    exponential_base: float = 2.0
    jitter: bool = True         # Add randomness to prevent thundering herd
    
    # Retryable error patterns
    retryable_errors: Tuple[type, ...] = (
        asyncio.TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
        BrokenPipeError,
    )
    
    # Error messages that indicate retryable conditions
    retryable_messages: Tuple[str, ...] = (
        "timeout",
        "connection reset",
        "connection refused",
        "broken pipe",
        "temporarily unavailable",
        "rate limit",
    )


@dataclass 
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5        # Failures before opening
    success_threshold: int = 2        # Successes to close from half-open
    timeout: timedelta = timedelta(seconds=60)  # Time before trying half-open
    window_size: int = 10            # Size of sliding window for tracking


@dataclass
class ServerHealthMetrics:
    """Track health metrics for a server"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_calls == 0:
            return 1.0
        return self.successful_calls / self.total_calls
    
    @property
    def recent_success_rate(self) -> float:
        """Calculate recent success rate from sliding window"""
        if not self.recent_results:
            return 1.0
        successes = sum(1 for result in self.recent_results if result)
        return successes / len(self.recent_results)
    
    def record_success(self):
        """Record a successful call"""
        self.total_calls += 1
        self.successful_calls += 1
        self.last_success = datetime.now()
        self.recent_results.append(True)
    
    def record_failure(self):
        """Record a failed call"""
        self.total_calls += 1
        self.failed_calls += 1
        self.last_failure = datetime.now()
        self.recent_results.append(False)


class CircuitBreaker:
    """
    Circuit breaker implementation to prevent cascading failures
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change = datetime.now()
        self._lock = asyncio.Lock()
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            if not await self._can_execute():
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            
        try:
            result = await func(*args, **kwargs) 
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _can_execute(self) -> bool:
        """Check if we can execute a call"""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.now() - self.last_failure_time > self.config.timeout:
                logger.info(f"Circuit breaker {self.name}: transitioning to HALF_OPEN")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()
                return True
            return False
            
        # HALF_OPEN state
        return True
    
    async def _on_success(self):
        """Handle successful call"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    logger.info(f"Circuit breaker {self.name}: closing circuit after recovery")
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    self.last_state_change = datetime.now()
            elif self.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            
            if self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    logger.warning(f"Circuit breaker {self.name}: opening circuit after {self.failure_count} failures")
                    self.state = CircuitState.OPEN
                    self.last_state_change = datetime.now()
                    
            elif self.state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit breaker {self.name}: reopening circuit after failure in HALF_OPEN state")
                self.state = CircuitState.OPEN
                self.success_count = 0
                self.last_state_change = datetime.now()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_state_change": self.last_state_change.isoformat(),
            "can_execute": self.state != CircuitState.OPEN or (
                self.state == CircuitState.OPEN and 
                datetime.now() - self.last_failure_time > self.config.timeout
            )
        }


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass


class RetryableToolExecutor:
    """
    Handles retry logic with exponential backoff for tool execution
    """
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        
    async def execute_with_retry(
        self, 
        func: Callable[..., Tuple[bool, Dict[str, Any]]],
        *args,
        **kwargs
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a tool function with retry logic.
        
        Returns:
            Tuple[bool, Dict[str, Any]]: Success flag and result/error dict
        """
        last_error = None
        attempts = 0
        
        for attempt in range(self.config.max_attempts):
            attempts = attempt + 1
            
            try:
                # Execute the function
                success, result = await func(*args, **kwargs)
                
                if success:
                    # Add retry metadata to successful result
                    if attempts > 1:
                        result["retry_attempts"] = attempts - 1
                    return success, result
                
                # Check if error is retryable
                error_msg = result.get("error", "").lower()
                is_retryable = result.get("retryable", False) or any(
                    msg in error_msg for msg in self.config.retryable_messages
                )
                
                if not is_retryable or attempt == self.config.max_attempts - 1:
                    # Non-retryable error or last attempt
                    result["retry_attempts"] = attempts - 1
                    return success, result
                
                last_error = result
                
            except Exception as e:
                # Check if exception type is retryable
                is_retryable = isinstance(e, self.config.retryable_errors)
                
                if not is_retryable or attempt == self.config.max_attempts - 1:
                    # Non-retryable or last attempt
                    return False, {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "retry_attempts": attempts - 1,
                        "retryable": False
                    }
                
                last_error = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            
            # Calculate delay with exponential backoff
            delay = min(
                self.config.initial_delay * (self.config.exponential_base ** attempt),
                self.config.max_delay
            )
            
            # Add jitter if enabled
            if self.config.jitter:
                import random
                delay *= (0.5 + random.random())
            
            logger.info(f"Retrying after {delay:.2f}s (attempt {attempts}/{self.config.max_attempts})")
            await asyncio.sleep(delay)
        
        # Should not reach here, but just in case
        return False, {
            "error": "Max retry attempts exceeded",
            "last_error": last_error,
            "retry_attempts": attempts
        }


class ServerHealthMonitor:
    """
    Monitors server health and implements recovery strategies
    """
    
    def __init__(self, check_interval: float = 60.0):
        self.servers: Dict[str, ServerHealthMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.check_interval = check_interval
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_callbacks: Dict[str, Callable] = {}
        
    def register_server(
        self, 
        server_name: str, 
        circuit_config: Optional[CircuitBreakerConfig] = None,
        health_check_callback: Optional[Callable] = None
    ):
        """Register a server for health monitoring"""
        self.servers[server_name] = ServerHealthMetrics()
        self.circuit_breakers[server_name] = CircuitBreaker(
            server_name, 
            circuit_config or CircuitBreakerConfig()
        )
        if health_check_callback:
            self._health_check_callbacks[server_name] = health_check_callback
    
    async def record_call_result(self, server_name: str, success: bool):
        """Record the result of a server call"""
        if server_name not in self.servers:
            self.register_server(server_name)
        
        metrics = self.servers[server_name]
        if success:
            metrics.record_success()
        else:
            metrics.record_failure()
    
    def get_server_health(self, server_name: str) -> Dict[str, Any]:
        """Get health status for a server"""
        if server_name not in self.servers:
            return {"status": "unknown", "server": server_name}
        
        metrics = self.servers[server_name]
        circuit_status = self.circuit_breakers[server_name].get_status()
        
        return {
            "server": server_name,
            "total_calls": metrics.total_calls,
            "success_rate": metrics.success_rate,
            "recent_success_rate": metrics.recent_success_rate,
            "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
            "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
            "circuit_breaker": circuit_status,
            "status": self._determine_health_status(metrics, circuit_status)
        }
    
    def _determine_health_status(self, metrics: ServerHealthMetrics, circuit_status: Dict[str, Any]) -> str:
        """Determine overall health status"""
        if circuit_status["state"] == "open":
            return "unhealthy"
        elif circuit_status["state"] == "half_open":
            return "recovering"
        elif metrics.recent_success_rate < 0.5:
            return "degraded"
        elif metrics.recent_success_rate < 0.9:
            return "warning"
        else:
            return "healthy"
    
    async def start_monitoring(self):
        """Start background health monitoring"""
        if self._monitoring_task is None or self._monitoring_task.done():
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop background health monitoring"""
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Background loop for health checks"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered servers"""
        for server_name, callback in self._health_check_callbacks.items():
            try:
                # Run health check callback
                is_healthy = await callback(server_name)
                await self.record_call_result(server_name, is_healthy)
                
                if is_healthy:
                    logger.debug(f"Health check passed for {server_name}")
                else:
                    logger.warning(f"Health check failed for {server_name}")
                    
            except Exception as e:
                logger.error(f"Health check error for {server_name}: {e}")
                await self.record_call_result(server_name, False)
    
    def get_all_server_health(self) -> Dict[str, Dict[str, Any]]:
        """Get health status for all servers"""
        return {
            server_name: self.get_server_health(server_name)
            for server_name in self.servers
        }