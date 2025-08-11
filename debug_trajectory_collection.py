#!/usr/bin/env python3
"""
Debug script to test trajectory collection without full training
"""
import sys
import asyncio
import logging
import json
import torch
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "SkyRL" / "skyrl-train"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_single_trajectory():
    """Test collecting a single trajectory"""
    try:
        # Import components
        from environments.simple_shared_manager import SimpleSharedManager
        from training.data.trajectory_collector import TrajectoryCollector
        from training.core.qwen_policy_with_value_prompting import QwenPolicyWithValuePrompting
        
        logger.info("🔧 Creating tool manager...")
        tool_manager = SimpleSharedManager()
        
        logger.info("🤖 Loading policy...")
        # Use a simple policy configuration
        policy = QwenPolicyWithValuePrompting(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        logger.info("📊 Creating trajectory collector...")
        collector = TrajectoryCollector(
            policy=policy,
            shared_tool_manager=tool_manager,
            num_parallel_envs=1,  # Just 1 for testing
            timeout_seconds=60,   # Short timeout
            max_episode_length=5  # Short episodes
        )
        
        # Load a simple task
        logger.info("📝 Loading test task...")
        with open('data/inputs/train.json', 'r') as f:
            train_data = json.load(f)
        
        # Use the first task
        test_tasks = [train_data[0]]
        
        logger.info(f"🚀 Collecting trajectory for task: {test_tasks[0].get('task_metadata', {}).get('original_question', 'Unknown')}")
        
        # Collect trajectories
        trajectories = await collector.collect_batch(test_tasks)
        
        logger.info(f"✅ Collection completed! Got {len(trajectories)} trajectories")
        
        for i, traj in enumerate(trajectories):
            logger.info(f"  Trajectory {i}: {len(traj.get('turns', []))} turns, reward: {traj.get('total_reward', 0)}")
            
        return trajectories
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

async def main():
    """Main test function"""
    logger.info("🔥 Starting trajectory collection debug test...")
    
    # Test PyTorch
    import torch
    logger.info(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    
    # Run test
    trajectories = await test_single_trajectory()
    
    if trajectories:
        logger.info("✅ Debug test successful!")
    else:
        logger.error("❌ Debug test failed!")

if __name__ == "__main__":
    asyncio.run(main())