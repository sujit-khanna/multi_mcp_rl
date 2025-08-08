"""
SkyRL Tool Agent Environments
"""

from .mcp_tool_environment import MCPToolEnvironment, make_mcp_tool_env

# Register the environment with SkyRL
try:
    from skyrl.env.registry import register
    
    register(
        id="MCPToolEnv-v0",
        entry_point="skyrl_tool_agent.environments:make_mcp_tool_env",
    )
except ImportError:
    # SkyRL not installed yet
    pass

__all__ = ["MCPToolEnvironment", "make_mcp_tool_env"]