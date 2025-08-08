#!/usr/bin/env python3
"""
Minimal Test for GRPO Training Components

This script tests the core components without requiring model downloads.
"""

import sys
import logging
import tempfile
import json
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

import yaml
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from core.qwen_policy import QwenPolicy
        logger.info("✅ QwenPolicy imported successfully")
    except Exception as e:
        logger.error(f"❌ QwenPolicy import failed: {e}")
        return False
    
    try:
        from core.grpo_trainer import GRPOTrainer, Trajectory
        logger.info("✅ GRPOTrainer and Trajectory imported successfully")
    except Exception as e:
        logger.error(f"❌ GRPOTrainer import failed: {e}")
        return False
    
    try:
        from data.data_loader import StreamingDataset, CurriculumSampler, TaskBatcher
        logger.info("✅ Data loader components imported successfully")
    except Exception as e:
        logger.error(f"❌ Data loader import failed: {e}")
        return False
    
    try:
        from data.trajectory_collector import TrajectoryCollector, EpisodeResult
        logger.info("✅ Trajectory collector imported successfully")
    except Exception as e:
        logger.error(f"❌ Trajectory collector import failed: {e}")
        return False
    
    return True

def test_data_components():
    """Test data loading components."""
    logger.info("Testing data components...")
    
    try:
        from data.data_loader import CurriculumSampler, TaskBatcher, LRUCache
        
        # Test LRU Cache
        cache = LRUCache(capacity=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)  # Should evict "a"
        
        assert cache.get("a") is None, "LRU eviction failed"
        assert cache.get("d") == 4, "LRU insertion failed"
        logger.info("✅ LRU Cache working correctly")
        
        # Test Curriculum Sampler
        sampler = CurriculumSampler()
        dist_early = sampler.get_current_distribution(0, 10)
        dist_late = sampler.get_current_distribution(9, 10)
        
        assert isinstance(dist_early, dict), "Distribution not a dict"
        assert abs(sum(dist_early.values()) - 1.0) < 1e-6, "Distribution doesn't sum to 1"
        logger.info("✅ Curriculum Sampler working correctly")
        
        # Test Task Batcher
        batcher = TaskBatcher(target_total_turns=16, max_batch_size=4)
        
        # Create sample tasks
        sample_tasks = [
            {"task_metadata": {"complexity": "easy"}},
            {"task_metadata": {"complexity": "medium"}},
            {"task_metadata": {"complexity": "hard"}},
        ]
        
        batch = batcher.create_batch(sample_tasks)
        assert len(batch) > 0, "No batch created"
        logger.info("✅ Task Batcher working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data components test failed: {e}")
        return False

def test_trajectory_components():
    """Test trajectory components."""
    logger.info("Testing trajectory components...")
    
    try:
        from core.grpo_trainer import Trajectory
        from data.trajectory_collector import EpisodeResult
        
        # Test Trajectory creation
        trajectory = Trajectory(
            task_id="test_001",
            states=[
                [{"role": "user", "content": "Hello"}],
                [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
            ],
            actions=["Hi", "How are you?"],
            rewards=[0.5, 1.0],
            dones=[False, True],
        )
        
        assert trajectory.task_id == "test_001"
        assert trajectory.length == 2
        assert trajectory.total_reward == 1.5
        logger.info("✅ Trajectory creation working correctly")
        
        # Test EpisodeResult
        episode = EpisodeResult(
            task_id="test_002",
            trajectory=[
                {
                    "turn": 1,
                    "action": "Hello",
                    "observation": "Hi there",
                    "reward": 1.0,
                    "done": True,
                    "metadata": {}
                }
            ],
            total_reward=1.0,
            turns=1,
            success=True,
        )
        
        assert episode.is_valid(), "Episode should be valid"
        logger.info("✅ EpisodeResult working correctly")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Trajectory components test failed: {e}")
        return False

def test_config_handling():
    """Test configuration file handling."""
    logger.info("Testing configuration handling...")
    
    try:
        # Create sample configs
        model_config = {
            "model_name": "test_model",
            "max_length": 1024,
            "lora_mode": {"enabled": True, "r": 16},
            "generation": {"max_new_tokens": 256}
        }
        
        training_config = {
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-4
        }
        
        grpo_config = {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2
        }
        
        # Test YAML serialization/deserialization
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            for config_name, config_data in [
                ("model", model_config),
                ("training", training_config), 
                ("grpo", grpo_config)
            ]:
                config_file = temp_path / f"{config_name}_config.yaml"
                
                # Write config
                with open(config_file, 'w') as f:
                    yaml.dump(config_data, f)
                
                # Read config back
                with open(config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                assert loaded_config == config_data, f"Config {config_name} not preserved"
        
        logger.info("✅ Configuration handling working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration handling test failed: {e}")
        return False

def test_device_detection():
    """Test device detection."""
    logger.info("Testing device detection...")
    
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"✅ CUDA available: {device}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"✅ MPS available: {device}")
        else:
            device = torch.device("cpu")
            logger.info(f"✅ Using CPU: {device}")
        
        # Test basic tensor operations
        x = torch.randn(2, 3).to(device)
        y = torch.randn(3, 2).to(device)
        z = torch.mm(x, y)
        
        assert z.shape == (2, 2), "Matrix multiplication failed"
        assert z.device.type == device.type, "Device placement failed"
        
        logger.info("✅ Device detection and tensor operations working correctly")
        return True
        
    except Exception as e:
        logger.error(f"❌ Device detection test failed: {e}")
        return False

def test_memory_monitoring():
    """Test memory monitoring functions."""
    logger.info("Testing memory monitoring...")
    
    try:
        # Test memory functions
        device = torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
        
        def get_memory_usage():
            try:
                if device.type == "cuda":
                    return torch.cuda.memory_allocated() / (1024**3)
                elif device.type == "mps":
                    return torch.mps.current_allocated_memory() / (1024**3)
                else:
                    import psutil
                    return psutil.Process().memory_info().rss / (1024**3)
            except Exception:
                return 0.0
        
        initial_memory = get_memory_usage()
        
        # Allocate some memory
        big_tensor = torch.randn(1000, 1000).to(device)
        new_memory = get_memory_usage()
        
        # Clean up
        del big_tensor
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Memory monitoring working: {initial_memory:.2f}GB -> {new_memory:.2f}GB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Memory monitoring test failed: {e}")
        return False

def run_all_tests():
    """Run all minimal tests."""
    logger.info("🔥 Running minimal GRPO component tests...")
    
    tests = [
        ("Imports", test_imports),
        ("Data Components", test_data_components),
        ("Trajectory Components", test_trajectory_components),
        ("Configuration Handling", test_config_handling),
        ("Device Detection", test_device_detection),
        ("Memory Monitoring", test_memory_monitoring),
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            success = test_func()
            results[test_name] = success
            if success:
                passed += 1
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "="*60)
    print("MINIMAL TEST RESULTS")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nSummary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        return True
    else:
        print("⚠️  Some tests failed.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)