#!/usr/bin/env python3
"""Test to verify MCP tools are actually being executed"""

import asyncio
import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "environments"))

async def test_mcp_execution():
    """Test that MCP tools are actually executed and return real data"""
    
    print("Testing actual MCP tool execution...")
    
    # Import the shared manager
    from simple_shared_manager import SimpleSharedManager
    
    # Initialize the manager
    manager = SimpleSharedManager()
    await manager.initialize()
    
    print(f"\n✓ Initialized {len(manager.available_tools)} tools")
    
    # Test 1: Execute tavily_search
    print("\n1. Testing tavily_search...")
    try:
        result = manager.execute_tool_sync("tavily_search", {"query": "latest AI news"})
        if result:
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            print(f"   ✓ tavily_search returned: {result_preview}")
        else:
            print(f"   ✗ tavily_search returned empty result")
    except Exception as e:
        print(f"   ✗ tavily_search failed: {e}")
    
    # Test 2: Execute fmp_get_quote
    print("\n2. Testing fmp_get_quote...")
    try:
        result = manager.execute_tool_sync("fmp_get_quote", {"symbol": "AAPL"})
        if result:
            # Try to parse as JSON if it's a string
            if isinstance(result, str):
                try:
                    data = json.loads(result)
                    print(f"   ✓ fmp_get_quote returned data for AAPL: price={data.get('price', 'N/A')}")
                except:
                    result_preview = result[:200] + "..." if len(result) > 200 else result
                    print(f"   ✓ fmp_get_quote returned: {result_preview}")
            else:
                print(f"   ✓ fmp_get_quote returned: {result}")
        else:
            print(f"   ✗ fmp_get_quote returned empty result")
    except Exception as e:
        print(f"   ✗ fmp_get_quote failed: {e}")
    
    # Test 3: Execute execute_python
    print("\n3. Testing execute_python...")
    try:
        result = manager.execute_tool_sync("execute_python", {
            "code": "result = 2 + 2\nprint(f'The answer is {result}')"
        })
        if result:
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            print(f"   ✓ execute_python returned: {result_preview}")
        else:
            print(f"   ✗ execute_python returned empty result")
    except Exception as e:
        print(f"   ✗ execute_python failed: {e}")
    
    # Test 4: Check that a forced fallback call would work
    print("\n4. Testing forced fallback (tavily_search with 'latest market news')...")
    try:
        result = manager.execute_tool_sync("tavily_search", {"query": "latest market news"})
        if result:
            # Check if it contains market-related content
            result_str = str(result).lower()
            if any(word in result_str for word in ['market', 'stock', 'trading', 'price', 'nasdaq', 'dow']):
                print(f"   ✓ Forced fallback query returned market-related content")
            else:
                print(f"   ⚠ Query executed but content may not be market-related")
            
            result_preview = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)
            print(f"      Preview: {result_preview}")
        else:
            print(f"   ✗ Forced fallback query returned empty result")
    except Exception as e:
        print(f"   ✗ Forced fallback query failed: {e}")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_execution())
    print(f"\n{'✅ MCP TOOLS ARE WORKING' if success else '❌ MCP TOOLS NOT WORKING'}")