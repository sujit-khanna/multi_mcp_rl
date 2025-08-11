#!/usr/bin/env python3
"""Test script to verify MCP servers can start and list tools."""

import asyncio
import sys
from pathlib import Path
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

MCP_SERVERS = {
    "slack": str(PROJECT_ROOT / "mcp_tools" / "limited" / "slack_limited_server.py"),
    "tavily": str(PROJECT_ROOT / "mcp_tools" / "limited" / "tavily_limited_server.py"),
    "polygon": str(PROJECT_ROOT / "mcp_tools" / "limited" / "polygon_limited_server.py"),
    "fmp": str(PROJECT_ROOT / "mcp_tools" / "limited" / "fmp_limited_server.py"),
    "python": str(PROJECT_ROOT / "mcp_tools" / "limited" / "python_execution_server.py")
}

async def test_server(server_name: str, server_path: str):
    """Test if a single MCP server can start and list tools."""
    try:
        import os
        from dotenv import dotenv_values
        
        # Load environment variables like the real manager does
        env_vars = dict(os.environ)
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            env_vars.update(dotenv_values(str(env_file)))
        
        # Add current Python path to ensure subprocess has all modules
        env_vars["PYTHONPATH"] = ":".join([
            str(PROJECT_ROOT),
            "/home/ubuntu/.local/lib/python3.12/site-packages",
            env_vars.get("PYTHONPATH", "")
        ])
        
        server_params = StdioServerParameters(
            command="python",
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        
        async with stdio_client(server_params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                tools = tools_result.tools if tools_result.tools else []
                print(f"✓ {server_name}: Started successfully, {len(tools)} tools available")
                for tool in tools:
                    print(f"  - {tool.name}: {tool.description}")
                return True
                
    except Exception as e:
        print(f"✗ {server_name}: Failed to start - {str(e)}")
        return False

async def main():
    """Test all MCP servers."""
    print("Testing MCP servers...")
    print("=" * 50)
    
    results = {}
    for server_name, server_path in MCP_SERVERS.items():
        print(f"\nTesting {server_name} server...")
        results[server_name] = await test_server(server_name, server_path)
    
    print("\n" + "=" * 50)
    print("Summary:")
    working_count = sum(results.values())
    total_count = len(results)
    
    for server_name, working in results.items():
        status = "✓ Working" if working else "✗ Failed"
        print(f"  {server_name}: {status}")
    
    print(f"\nOverall: {working_count}/{total_count} servers working")
    return working_count == total_count

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)