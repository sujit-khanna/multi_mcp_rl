#!/usr/bin/env python3
"""
Test policy compute_log_probs with real conversation data
"""

import sys
import json
import logging
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
from core.qwen_policy import QwenPolicy

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_real_conversation_format():
    """Test compute_log_probs with real training data format."""
    
    try:
        logger.info("🧪 Testing real conversation format...")
        
        # Load real training data
        with open("/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/data/inputs/train.json", 'r') as f:
            tasks = json.load(f)
        
        logger.info(f"Loaded {len(tasks)} tasks")
        
        # Initialize policy
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
        
        # Test with first task
        task = tasks[0]
        conversation = task["prompt"]
        
        logger.info(f"Testing with conversation of {len(conversation)} messages")
        logger.info(f"First message: {conversation[0]}")
        logger.info(f"Second message type: {type(conversation[1]['content'])}")
        logger.info(f"Second message length: {len(conversation[1]['content'])}")
        
        # Extract user and assistant pairs
        states = []
        actions = []
        
        for i in range(0, len(conversation) - 1, 2):
            if i + 1 < len(conversation):
                # State: conversation up to user message
                state = conversation[:i+1]
                states.append(state)
                
                # Action: assistant response
                action = conversation[i+1]["content"]
                actions.append(action)
        
        logger.info(f"Created {len(states)} state-action pairs")
        logger.info(f"First state: {len(states[0])} messages")
        logger.info(f"First action length: {len(actions[0])} chars")
        
        # Test compute_log_probs
        logger.info("🎯 Testing compute_log_probs...")
        try:
            log_probs = policy.compute_log_probs(states, actions)
            logger.info(f"✅ Success! Log probs shape: {log_probs.shape}")
            logger.info(f"Log prob values: {log_probs}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in compute_log_probs: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Try with a single simple case
            logger.info("Trying with simplified conversation...")
            simple_state = [{"role": "user", "content": "Hello"}]
            simple_action = "Hi there!"
            
            try:
                simple_log_probs = policy.compute_log_probs([simple_state], [simple_action])
                logger.info(f"✅ Simple case works: {simple_log_probs.shape}")
                logger.error("❌ Issue is with real conversation format")
            except Exception as e2:
                logger.error(f"❌ Even simple case fails: {e2}")
            
            return False
        
    except Exception as e:
        logger.error(f"❌ Test failed: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_real_conversation_format()
    if success:
        print("🎉 Real conversation format test PASSED!")
    else:
        print("❌ Real conversation format test FAILED!")
    
    sys.exit(0 if success else 1)