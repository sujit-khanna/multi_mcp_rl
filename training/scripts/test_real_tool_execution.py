#!/usr/bin/env python3
"""
Test script to verify real tool execution is happening
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "environments"))

from mcp_tool_environment_with_logging import MCPToolEnvironmentWithLogging
from simple_shared_manager import SimpleSharedManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_trajectory():
    """Test a single trajectory with real tool execution"""
    
    # Create test task that requires tool use
    test_task = {
        "task_metadata": {
            "task_id": "test_real_tools_001",
            "complexity": "easy",
            "category": "finance"
        },
        "prompt": [
            {
                "role": "user",
                "content": "What is the current stock price of Apple (AAPL)?"
            }
        ],
        "reward_spec": {
            "ground_truth": {
                "success_criteria": {
                    "expected_tools": ["fmp_get_quote"],
                    "min_tool_calls": 1
                }
            }
        }
    }
    
    logger.info("=" * 80)
    logger.info("TESTING REAL TOOL EXECUTION")
    logger.info("=" * 80)
    
    # Initialize shared tool manager
    logger.info("\n1. Initializing shared tool manager...")
    tool_manager = SimpleSharedManager()
    await tool_manager.initialize()
    
    available_tools = tool_manager.get_available_tools()
    logger.info(f"   Available tools: {len(available_tools)}")
    for tool in available_tools[:5]:  # Show first 5 tools
        logger.info(f"   - {tool['name']}: {tool['description'][:50]}...")
    
    # Create environment
    logger.info("\n2. Creating environment...")
    env = MCPToolEnvironmentWithLogging(test_task)
    env.tool_manager = tool_manager
    env.available_tools = [tool['name'] for tool in available_tools]
    
    # Test action with tool call
    test_action = """<think>
The user wants to know the current stock price of Apple (AAPL). I'll use the fmp_get_quote tool to get this information.
</think>

I'll check the current stock price of Apple for you.

<tool_call>
{
    "name": "fmp_get_quote",
    "arguments": {
        "symbol": "AAPL"
    }
}
</tool_call>"""
    
    logger.info("\n3. Executing test action...")
    logger.info(f"   Action: {test_action[:100]}...")
    
    # Execute step
    step_result = env.step(test_action)
    
    # Log results
    logger.info("\n4. Step results:")
    if isinstance(step_result, dict):
        reward = step_result.get('reward', 0)
        done = step_result.get('done', False)
        observation = step_result.get('observation', '')
    else:
        reward = getattr(step_result, 'reward', 0)
        done = getattr(step_result, 'done', False)
        observation = getattr(step_result, 'observations', '')
    
    logger.info(f"   Reward: {reward}")
    logger.info(f"   Done: {done}")
    logger.info(f"   Observation preview: {str(observation)[:200]}...")
    
    # Get execution summary
    summary = env.get_execution_summary()
    logger.info("\n5. Execution summary:")
    logger.info(f"   Total tool calls: {summary['total_tool_calls']}")
    logger.info(f"   Successful calls: {summary['successful_calls']}")
    logger.info(f"   Failed calls: {summary['failed_calls']}")
    
    if summary['tool_execution_log']:
        logger.info("\n6. Tool execution log:")
        for i, log_entry in enumerate(summary['tool_execution_log']):
            logger.info(f"\n   Call #{i+1}:")
            logger.info(f"   - Tool: {log_entry['tool']}")
            logger.info(f"   - Arguments: {log_entry['arguments']}")
            logger.info(f"   - Success: {log_entry['success']}")
            if log_entry['success']:
                logger.info(f"   - Result preview: {log_entry.get('result_preview', 'N/A')}")
            else:
                logger.info(f"   - Error: {log_entry.get('error', 'N/A')}")
    
    # Test a failing tool call
    logger.info("\n" + "=" * 80)
    logger.info("TESTING ERROR HANDLING")
    logger.info("=" * 80)
    
    error_action = """<tool_call>
{
    "name": "nonexistent_tool",
    "arguments": {}
}
</tool_call>"""
    
    logger.info("\n7. Testing error handling with non-existent tool...")
    error_result = env.step(error_action)
    
    # Cleanup
    await tool_manager.cleanup()
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_single_trajectory())