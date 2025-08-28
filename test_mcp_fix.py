#!/usr/bin/env python3
"""Test script to verify MCP tool execution is fixed"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "environments"))
sys.path.insert(0, str(Path(__file__).parent / "training"))

async def test_mcp_tools():
    """Test that MCP tools can be executed after the fixes"""
    
    # Import as training script does (without package prefix)
    from simple_shared_manager import SimpleSharedManager
    from mcp_tool_environment import MCPToolEnvironment
    
    print("1. Testing SimpleSharedManager initialization...")
    tool_manager = SimpleSharedManager()
    await tool_manager.initialize()
    
    available = list(tool_manager.available_tools.keys())
    print(f"   ✓ Discovered {len(available)} tools: {available[:5]}...")
    
    print("\n2. Testing MCPToolEnvironment with injected manager...")
    env = MCPToolEnvironment(task_data={
        "task_id": "test_task",
        "query": "Test the MCP tool execution",
        "complexity": "easy"
    })
    
    # Inject the shared manager (as trajectory collector does)
    env.tool_manager = tool_manager
    
    # Initialize tools (this should now work with the fix)
    await env.initialize_tools()
    
    print(f"   ✓ Environment has {len(env.available_tools)} available tools")
    
    print("\n3. Testing tool execution...")
    # Test with a simple tool call
    test_action = """<tool_call>
{"name": "execute_python", "arguments": {"code": "print('Hello from Python!'); result = 2 + 2; print(f'Result: {result}')"}}
</tool_call>"""
    
    # Execute step
    result = env.step(test_action)
    print(f"   ✓ Step executed: done={result.done}, reward={result.reward}")
    print(f"   ✓ Observation: {result.text[:100]}...")
    
    # Check if tool was actually executed (not "Tool not found")
    if "Tool 'execute_python' not found" in result.text:
        print("   ✗ ERROR: Tool still not found!")
        return False
    elif "error" in result.text.lower() and "not found" in result.text.lower():
        print("   ✗ ERROR: Tool execution failed!")
        return False
    else:
        print("   ✓ Tool execution successful!")
        return True

if __name__ == "__main__":
    success = asyncio.run(test_mcp_tools())
    sys.exit(0 if success else 1)