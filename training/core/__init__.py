"""
Core Training Components for GRPO
=================================

This module provides core infrastructure for GRPO training including:
- Environment adapter for MCPToolEnvironment integration
- Policy interfaces for external policies
- Task data loading and formatting
- Async-to-sync bridges for tool execution
"""

from .environment_adapter import (
    EnvironmentAdapter,
    PolicyAdapter,
    PolicyInterface,
    SharedToolManagerEnvironment,
    TaskDataLoader,
    AsyncToSyncBridge,
    create_environment_adapter
)

__all__ = [
    "EnvironmentAdapter",
    "PolicyAdapter",
    "PolicyInterface",
    "SharedToolManagerEnvironment",
    "TaskDataLoader", 
    "AsyncToSyncBridge",
    "create_environment_adapter"
]