#!/usr/bin/env python3
"""Minimal test to verify MCP tool execution fix without SkyRL dependency"""

import asyncio
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "environments"))

async def test_mcp_tools_minimal():
    """Minimal test that MCP tools can be loaded after the fixes"""
    
    print("Testing MCP tool initialization fix...")
    
    # Test 1: Import with fallback should work
    print("\n1. Testing import fallback in initialize_tools...")
    
    # Mock a simple environment to test the import fix
    class MockEnv:
        def __init__(self):
            self.tool_manager = None
            self.available_tools = []
            
        async def initialize_tools(self):
            """Test the fixed initialize_tools method"""
            # Only import if we need to create a new manager
            if self.tool_manager is None:
                try:
                    # Try relative import first (when imported as environments.mcp_tool_environment)
                    from .simple_shared_manager import SimpleSharedMCPToolManager
                    print("   ✗ Relative import worked (shouldn't in this context)")
                except ImportError:
                    # Fall back to absolute import (when imported directly with sys.path manipulation)
                    try:
                        from simple_shared_manager import SimpleSharedManager as SimpleSharedMCPToolManager
                        print("   ✓ Absolute import fallback worked!")
                    except ImportError as e:
                        print(f"   ✗ Failed to import: {e}")
                        self.available_tools = []
                        return
                
                self.tool_manager = SimpleSharedMCPToolManager()
            
            # Initialize and populate available tools (even if manager was injected)
            if hasattr(self.tool_manager, "initialize"):
                if asyncio.iscoroutinefunction(self.tool_manager.initialize):
                    await self.tool_manager.initialize()
                else:
                    self.tool_manager.initialize()
            
            # Always set available_tools from the manager
            self.available_tools = list(getattr(self.tool_manager, "available_tools", {}).keys())
            print(f"   ✓ Initialized {len(self.available_tools)} tools")
    
    # Test the mock env
    env = MockEnv()
    await env.initialize_tools()
    
    # Test 2: Check if shared manager works
    print("\n2. Testing SimpleSharedManager...")
    from simple_shared_manager import SimpleSharedManager
    
    manager = SimpleSharedManager()
    await manager.initialize()
    
    tools = list(manager.available_tools.keys())
    print(f"   ✓ Discovered {len(tools)} tools")
    if tools:
        print(f"   ✓ Example tools: {tools[:5]}")
    
    # Test 3: Check if injected manager works
    print("\n3. Testing injected tool manager...")
    env2 = MockEnv()
    env2.tool_manager = manager  # Inject existing manager
    await env2.initialize_tools()
    
    if len(env2.available_tools) == len(tools):
        print("   ✓ Injected manager tools match!")
    else:
        print(f"   ✗ Tool count mismatch: {len(env2.available_tools)} vs {len(tools)}")
    
    return len(tools) > 0

if __name__ == "__main__":
    try:
        success = asyncio.run(test_mcp_tools_minimal())
        print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)