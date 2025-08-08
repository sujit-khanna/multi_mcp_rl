#!/usr/bin/env python3
"""
Isolated GRPO test - bypasses MCP environments to test GRPO logic directly
"""

import sys
import logging
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from core.qwen_policy import QwenPolicy
from core.grpo_trainer import GRPOTrainer, Trajectory

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_trajectory(task_id: str, length: int = 3) -> Trajectory:
    """Create a mock trajectory for testing."""
    
    # Mock conversation states
    states = []
    actions = []
    rewards = []
    dones = []
    
    for i in range(length):
        # Mock state (conversation history)
        state = [
            {"role": "user", "content": f"Task {task_id} step {i}: Find stock price"},
            {"role": "assistant", "content": f"I will help with step {i}"}
        ]
        states.append(state)
        
        # Mock action
        action = f"<think>This is step {i} of task {task_id}</think> I'll help you find that information."
        actions.append(action)
        
        # Mock reward and done
        rewards.append(0.1 * (i + 1))  # Increasing rewards
        dones.append(i == length - 1)  # Only last step is done
    
    return Trajectory(
        task_id=task_id,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones
    )

def test_isolated_grpo():
    """Test GRPO training with mock data, bypassing MCP environments."""
    
    try:
        logger.info("🧪 Starting isolated GRPO test...")
        
        # Initialize policies
        logger.info("Initializing policy...")
        policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        
        # Enable training mode
        policy.enable_training_mode()
        logger.info(f"Policy initialized: {policy.get_trainable_parameters():,} trainable parameters")
        
        # Create reference policy
        logger.info("Creating reference policy...")
        ref_policy = QwenPolicy(
            model_config_path="configs/model_config_mps.yaml",
            training_config_path="configs/training_config_mps.yaml",
            use_lora=True,
            device="cpu",
            load_in_4bit=False,
        )
        ref_policy.enable_eval_mode()
        
        # Load configurations
        import yaml
        with open("configs/grpo_config_mps.yaml", 'r') as f:
            grpo_config = yaml.safe_load(f)
        with open("configs/training_config_mps.yaml", 'r') as f:
            training_config = yaml.safe_load(f)
        
        # Initialize trainer
        logger.info("Initializing GRPO trainer...")
        trainer = GRPOTrainer(
            policy=policy,
            reference_policy=ref_policy,
            grpo_config=grpo_config,
            training_config=training_config,
            device=torch.device("cpu"),
        )
        logger.info("✅ Trainer initialized")
        
        # Create mock trajectories
        logger.info("Creating mock trajectories...")
        trajectories = [
            create_mock_trajectory("test_001", length=3),
            create_mock_trajectory("test_002", length=4),
        ]
        logger.info(f"Created {len(trajectories)} mock trajectories")
        
        # Test training step
        logger.info("🎯 Testing GRPO training step...")
        metrics = trainer.train_step(trajectories)
        logger.info(f"✅ Training step completed successfully!")
        logger.info(f"Metrics: {metrics}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Isolated GRPO test failed: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_isolated_grpo()
    if success:
        print("🎉 Isolated GRPO test PASSED!")
    else:
        print("❌ Isolated GRPO test FAILED!")
    
    sys.exit(0 if success else 1)