#!/usr/bin/env python3
"""
Test Environment Adapter Integration
====================================

This script tests the integration between MCPToolEnvironment and training infrastructure.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from core.environment_adapter import (
    create_environment_adapter,
    PolicyInterface,
    PolicyAdapter
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestPolicy:
    """Simple test policy that demonstrates multi-turn reasoning"""
    
    def __init__(self):
        self.turn_count = 0
        self.tools_used = []
        
    def generate_action(self, conversation: List[Dict[str, str]]) -> str:
        """Generate action based on conversation history"""
        self.turn_count += 1
        
        # Get user query
        user_query = ""
        for msg in conversation:
            if msg["role"] == "user":
                user_query = msg["content"]
                break
        
        logger.info(f"Policy turn {self.turn_count}: Processing '{user_query[:50]}...'")
        
        # Simple policy logic
        if self.turn_count == 1:
            # First turn: reasoning and tool call
            if "AMZN" in user_query:
                return """<think>
The user is asking about AMZN (Amazon). They want:
1. Previous close price
2. Number of institutional holders

I'll need to use polygon_get_previous_close for the price data.
</think>

<tool_call>{"name": "polygon_get_previous_close", "arguments": {"ticker": "AMZN"}}</tool_call>"""
            elif "AAPL" in user_query:
                return """<think>
The user wants the 1-day percent change for AAPL and a bar chart.
First, let me get the quote data.
</think>

<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>"""
            else:
                return "<think>I need to understand what the user wants.</think>\nCould you please clarify your request?"
        
        elif self.turn_count == 2:
            # Second turn: follow-up based on first result
            if "AMZN" in user_query:
                return """<think>
Now I need to find information about institutional holders.
I'll search for this information.
</think>

<tool_call>{"name": "tavily_search", "arguments": {"query": "Amazon AMZN institutional holders count"}}</tool_call>"""
            else:
                return "Based on the data, the task is complete."
        
        else:
            # Final turn
            return "Task completed successfully. I've provided the requested information."
    
    def reset(self):
        """Reset policy state"""
        self.turn_count = 0
        self.tools_used = []


def test_basic_integration():
    """Test basic environment adapter functionality"""
    logger.info("=== Testing Basic Integration ===")
    
    try:
        # Create adapter
        adapter = create_environment_adapter(
            data_path="data/processed/train.json",
            use_shared_tools=True
        )
        logger.info("✓ Environment adapter created successfully")
        
        # Test data loading
        tasks = adapter.data_loader.get_batch(5)
        logger.info(f"✓ Loaded {len(tasks)} tasks")
        
        # Display sample task
        if tasks:
            task = tasks[0]
            logger.info(f"Sample task ID: {task['task_metadata']['task_id']}")
            logger.info(f"Complexity: {task['task_metadata']['complexity']}")
            
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic integration test failed: {e}")
        return False


def test_policy_integration():
    """Test policy integration with environment"""
    logger.info("\n=== Testing Policy Integration ===")
    
    try:
        # Create adapter without shared tools for faster testing
        adapter = create_environment_adapter(use_shared_tools=False)
        
        # Create test policy with PolicyInterface wrapper
        test_policy = TestPolicy()
        policy = PolicyInterface(
            generate_action=test_policy.generate_action,
            reset=test_policy.reset
        )
        
        # Get a task
        tasks = adapter.data_loader.get_batch(1)
        if not tasks:
            logger.error("No tasks available")
            return False
        
        task = tasks[0]
        logger.info(f"Testing with task: {task['task_metadata']['task_id']}")
        
        # Run episode
        logger.info("Running episode...")
        episode_data = adapter._run_episode(task, policy)
        
        # Check results
        logger.info(f"✓ Episode completed")
        logger.info(f"  Total reward: {episode_data['total_reward']:.3f}")
        logger.info(f"  Turns: {len(episode_data['actions'])}")
        logger.info(f"  Task completed: {episode_data['final_metrics']['task_completed']}")
        logger.info(f"  Tool calls: {episode_data['final_metrics']['total_tool_calls']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Policy integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_iterator():
    """Test training iterator functionality"""
    logger.info("\n=== Testing Training Iterator ===")
    
    try:
        # Create adapter without shared tools for faster testing
        adapter = create_environment_adapter(use_shared_tools=False)
        
        # Create test policy with PolicyInterface wrapper
        test_policy = TestPolicy()
        policy = PolicyInterface(
            generate_action=test_policy.generate_action,
            reset=test_policy.reset
        )
        
        # Create training iterator
        iterator = adapter.create_training_iterator(
            policy_fn=policy,
            batch_size=2,
            max_episodes_per_task=1,
            shuffle=True
        )
        
        # Get one batch
        batch = next(iterator)
        
        logger.info(f"✓ Training iterator created")
        logger.info(f"  Batch size: {batch['batch_size']}")
        logger.info(f"  Episodes in batch: {len(batch['episodes'])}")
        
        # Check episode structure
        if batch['episodes']:
            episode = batch['episodes'][0]
            logger.info(f"  Sample episode task: {episode['task_id']}")
            logger.info(f"  Sample episode reward: {episode['total_reward']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Training iterator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_async_bridge():
    """Test async-to-sync bridge functionality"""
    logger.info("\n=== Testing Async Bridge ===")
    
    try:
        from core.environment_adapter import AsyncToSyncBridge
        import asyncio
        
        # Create bridge
        bridge = AsyncToSyncBridge()
        
        # Test async operation
        async def test_coro():
            await asyncio.sleep(0.1)
            return "Async operation completed"
        
        result = bridge.run_async(test_coro())
        logger.info(f"✓ Async bridge test: {result}")
        
        # Cleanup
        bridge.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Async bridge test failed: {e}")
        return False


def main():
    """Run all integration tests"""
    logger.info("🔧 Testing Environment Adapter Integration")
    logger.info("=" * 60)
    
    tests = [
        ("Basic Integration", test_basic_integration),
        ("Policy Integration", test_policy_integration),
        ("Training Iterator", test_training_iterator),
        ("Async Bridge", test_async_bridge)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    total = len(tests)
    logger.info("\n" + "=" * 60)
    logger.info("INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        logger.info("✅ Environment adapter is ready for GRPO training")
        return True
    else:
        logger.error(f"⚠️  {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)