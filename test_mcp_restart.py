#!/usr/bin/env python3
"""Test the MCP server health check and restart logic"""

import asyncio
import sys
from pathlib import Path

# Add environment path
ROOT_DIR = Path('/home/ubuntu/multi_mcp_rl')
sys.path.insert(0, str(ROOT_DIR / 'environments'))

async def check_and_restart_mcp_servers():
    try:
        from simple_shared_manager import SimpleSharedManager
        
        print('🔍 Initializing MCP server manager...')
        manager = SimpleSharedManager()
        await manager.initialize()
        
        # Check tool availability
        tools = list(manager.available_tools.keys())
        print(f'✓ Found {len(tools)} tools from MCP servers')
        
        # Group by server
        servers = {}
        for tool_name, tool_info in manager.available_tools.items():
            server = tool_info.get('server', 'unknown')
            servers[server] = servers.get(server, 0) + 1
        
        print('MCP Servers Status:')
        expected_servers = ['slack', 'tavily', 'polygon', 'fmp', 'python']
        active_servers = list(servers.keys())
        
        for server in expected_servers:
            if server in servers:
                print(f'  ✅ {server}: {servers[server]} tools')
            else:
                print(f'  ❌ {server}: NOT RESPONDING')
        
        # Check if we have all expected servers
        missing_servers = set(expected_servers) - set(active_servers)
        if missing_servers:
            print(f'\n⚠️  Missing servers: {", ".join(missing_servers)}')
            print('   Attempting to restart MCP servers...')
            
            # Force re-initialization which will restart servers
            manager._initialized = False
            await manager.initialize()
            
            # Re-check
            tools = list(manager.available_tools.keys())
            servers = {}
            for tool_name, tool_info in manager.available_tools.items():
                server = tool_info.get('server', 'unknown')
                servers[server] = servers.get(server, 0) + 1
            
            print('\nAfter restart attempt:')
            for server in expected_servers:
                if server in servers:
                    print(f'  ✅ {server}: {servers[server]} tools')
                else:
                    print(f'  ⚠️  {server}: Still not responding (check API keys)')
        
        total_tools = len(tools)
        if total_tools >= 20:
            print(f'\n✅ MCP servers ready with {total_tools} tools available')
            return True
        elif total_tools >= 10:
            print(f'\n⚠️  Only {total_tools} tools available (some servers may be down)')
            print('   Training will proceed with reduced tool availability')
            return True
        else:
            print(f'\n❌ Only {total_tools} tools available - insufficient for training')
            return False
            
    except Exception as e:
        print(f'❌ Error checking MCP servers: {e}')
        print('   Training will proceed but tool execution may fail')
        return True  # Allow training to proceed

# Run the check
if __name__ == '__main__':
    result = asyncio.run(check_and_restart_mcp_servers())
    sys.exit(0 if result else 1)