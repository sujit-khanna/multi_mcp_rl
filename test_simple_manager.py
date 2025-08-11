#!/usr/bin/env python3
"""Test script to verify the SimpleSharedMCPToolManager works."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from environments.simple_shared_manager import SimpleSharedMCPToolManager

async def test_tool_manager():
    """Test the existing tool manager."""
    print("Testing SimpleSharedMCPToolManager...")
    print("=" * 50)
    
    try:
        # Initialize the tool manager
        manager = SimpleSharedMCPToolManager()
        await manager.initialize()
        
        # Get available tools
        tools = manager.get_available_tools()
        print(f"✓ Manager initialized successfully")
        print(f"✓ Found {len(tools)} tools across all servers")
        
        # List tools by server
        servers = {}
        for tool in tools:
            server_name = tool['server']
            if server_name not in servers:
                servers[server_name] = []
            servers[server_name].append(tool['name'])
        
        print("\nTools by server:")
        for server_name, tool_names in servers.items():
            print(f"  {server_name}: {len(tool_names)} tools")
            for tool_name in tool_names[:3]:  # Show first 3 tools
                print(f"    - {tool_name}")
            if len(tool_names) > 3:
                print(f"    ... and {len(tool_names) - 3} more")
        
        # Check health status
        health = manager.get_health_status()
        print(f"\nHealth Status:")
        print(f"  Initialized: {health['initialized']}")
        print(f"  Available tools: {health['available_tools']}")
        print(f"  Failed servers: {health['failed_servers']}")
        
        # Test a simple tool if available
        if len(tools) > 0:
            test_tool = tools[0]
            print(f"\nTesting tool: {test_tool['name']}")
            try:
                # Most tools need specific arguments, so this might fail
                # but at least we can see if the connection works
                success, result = await manager.execute_tool(test_tool['name'], {})
                if success:
                    print(f"✓ Tool executed successfully: {result.get('content', '')[:100]}...")
                else:
                    print(f"⚠ Tool execution failed (expected for most tools without proper args): {result.get('error', '')[:100]}")
            except Exception as e:
                print(f"⚠ Tool execution error (might be expected): {str(e)[:100]}")
        
        await manager.cleanup()
        return True
        
    except Exception as e:
        print(f"✗ Manager test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_tool_manager())
    sys.exit(0 if success else 1)