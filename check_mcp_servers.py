#!/usr/bin/env python3
"""Check if MCP servers are live and responding"""

import asyncio
import sys
import json
from pathlib import Path
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "environments"))

async def check_mcp_servers():
    """Test each MCP server to see if it's live"""
    
    print("Testing MCP Server Connectivity...\n")
    print("="*60)
    
    from simple_shared_manager import SimpleSharedManager
    
    # Initialize manager
    manager = SimpleSharedManager()
    start_time = time.time()
    await manager.initialize()
    init_time = time.time() - start_time
    
    print(f"✓ Manager initialized in {init_time:.2f} seconds")
    print(f"✓ Total tools available: {len(manager.available_tools)}\n")
    
    # Group tools by server
    tools_by_server = {}
    for tool_name, tool_info in manager.available_tools.items():
        server = tool_info.get('server', 'unknown')
        if server not in tools_by_server:
            tools_by_server[server] = []
        tools_by_server[server].append(tool_name)
    
    print("Tools by Server:")
    for server, tools in sorted(tools_by_server.items()):
        print(f"  {server}: {len(tools)} tools")
        for tool in sorted(tools)[:3]:  # Show first 3 tools
            print(f"    - {tool}")
        if len(tools) > 3:
            print(f"    ... and {len(tools)-3} more")
    
    print("\n" + "="*60)
    print("Testing Server Responsiveness:\n")
    
    # Test one tool from each server
    test_cases = [
        ("slack", "list_slack_channels", {}),
        ("tavily", "tavily_search", {"query": "test"}),
        ("polygon", "polygon_get_market_status", {}),
        ("fmp", "fmp_get_quote", {"symbol": "AAPL"}),
        ("python", "execute_python", {"code": "print('test')"})
    ]
    
    results = []
    for server, tool_name, args in test_cases:
        if tool_name not in manager.available_tools:
            print(f"❌ {server:10} - Tool '{tool_name}' not available")
            results.append(False)
            continue
            
        print(f"Testing {server:10} - {tool_name}...", end=" ")
        start = time.time()
        try:
            result = manager.execute_tool_sync(tool_name, args)
            elapsed = time.time() - start
            
            if result:
                print(f"✓ Response in {elapsed:.2f}s")
                results.append(True)
            else:
                print(f"✗ Empty response after {elapsed:.2f}s")
                results.append(False)
        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e)
            if "not found" in error_msg.lower():
                print(f"✗ Tool not found after {elapsed:.2f}s")
            else:
                print(f"✗ Error after {elapsed:.2f}s: {error_msg[:50]}")
            results.append(False)
    
    print("\n" + "="*60)
    print("Summary:")
    live_count = sum(results)
    total_count = len(test_cases)
    
    if live_count == total_count:
        print(f"✅ ALL {total_count} MCP SERVERS ARE LIVE AND RESPONDING")
    elif live_count > 0:
        print(f"⚠️  {live_count}/{total_count} MCP SERVERS ARE LIVE")
        print("   Some servers may be down or misconfigured")
    else:
        print(f"❌ NO MCP SERVERS ARE RESPONDING")
        print("   Check server configurations and API keys")
    
    return live_count, total_count

if __name__ == "__main__":
    live, total = asyncio.run(check_mcp_servers())
    sys.exit(0 if live == total else 1)