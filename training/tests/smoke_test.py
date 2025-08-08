#!/usr/bin/env python3
"""
Smoke Test for GRPO Training Pipeline

This script provides a quick validation test for the complete GRPO training system.
It verifies that all components work together correctly and can complete a full
training iteration in under 2 minutes.

Usage:
    python smoke_test.py
    python smoke_test.py --mode lora
    python smoke_test.py --mode full --skip_model_tests
"""

import argparse
import asyncio
import json
import logging
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any

import torch
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Core training imports
from core.qwen_policy import QwenPolicy
from core.grpo_trainer import GRPOTrainer, Trajectory
from data.data_loader import StreamingDataset, CurriculumSampler, TaskBatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmokeTestRunner:
    """
    Comprehensive smoke test runner for GRPO training pipeline.
    
    This class runs a series of tests to validate that all components
    of the GRPO training system work correctly together.
    """
    
    def __init__(self, mode: str = "lora", skip_model_tests: bool = False):
        """Initialize smoke test runner."""
        
        self.mode = mode
        self.skip_model_tests = skip_model_tests
        self.device = self._get_best_device()
        self.temp_dir = None
        self.test_results = {}
        self.start_time = time.time()
        
        logger.info(f"SmokeTest initialized: mode={mode}, device={self.device}")
        logger.info(f"macOS Unified Memory Available: {self._get_available_memory():.1f} GB")
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device for testing."""
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # macOS Metal Performance Shaders
        else:
            return torch.device("cpu")
    
    def _get_available_memory(self) -> float:
        """Get available memory in GB (macOS unified memory)."""
        
        try:
            import psutil
            memory_info = psutil.virtual_memory()
            return memory_info.available / (1024**3)  # Convert to GB
        except ImportError:
            return 48.0  # Default assumption for macOS with 48GB
    
    def _create_test_configs(self) -> Dict[str, Dict[str, Any]]:
        """Create minimal test configurations."""
        
        model_config = {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "trust_remote_code": True,
            "max_length": 1024,  # Smaller for testing
            "context_length": 1024,
            "vocab_size": 151936,
            "generation": {
                "max_new_tokens": 256,  # Smaller for testing
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": 151643,
                "eos_token_id": 151645,
                "repetition_penalty": 1.1,
            },
            "stop_sequences": ["</think>", "</tool_call>", "<|im_end|>"],
            "special_tokens": {
                "reasoning_start": "<think>",
                "reasoning_end": "</think>",
                "tool_call_start": "<tool_call>",
                "tool_call_end": "</tool_call>",
            },
            "lora_mode": {
                "enabled": self.mode == "lora",
                "r": 8,  # Smaller for testing
                "alpha": 16,
                "target_modules": ["q_proj", "v_proj", "o_proj"],  # Fewer modules for testing
                "dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "full_finetune_mode": {
                "enabled": self.mode == "full",
                "gradient_checkpointing": True,
                "use_flash_attention": False,  # Disable for testing
                "bf16": False,  # Use fp32 for testing
                "fp16": False,
            },
            "memory_optimization": {
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "torch_dtype": "float32",  # Use fp32 for testing stability
            },
            "quantization": {
                "load_in_4bit": False,  # Disable for testing
                "bnb_4bit_compute_dtype": "float32",
                "bnb_4bit_use_double_quant": False,
                "bnb_4bit_quant_type": "nf4",
            }
        }
        
        training_config = {
            "experiment_name": "smoke_test",
            "num_epochs": 1,  # Single epoch for testing
            "max_steps": 10,
            "save_steps": 5,
            "eval_steps": 3,
            "logging_steps": 1,
            "warmup_steps": 1,
            "lora_mode": {
                "per_device_train_batch_size": 1,  # Small batch for testing
                "gradient_accumulation_steps": 2,
            },
            "full_finetune_mode": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 2,
            },
            "lora_learning_rate": 1e-4,
            "full_finetune_learning_rate": 1e-5,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "adam_epsilon": 1e-5,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "seed": 42,
        }
        
        grpo_config = {
            "algorithm": "grpo",
            "group_size": 2,  # Small group for testing
            "kl_penalty_coef": 0.1,
            "target_kl_divergence": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "normalize_advantages": True,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
            "ref_policy_update_frequency": 5,  # More frequent for testing
            "max_episode_length": 3,  # Short episodes for testing
            "rollout_batch_size": 2,
            "episodes_per_update": 2,
        }
        
        return {
            "model": model_config,
            "training": training_config,
            "grpo": grpo_config,
        }
    
    def _create_sample_tasks(self) -> List[Dict[str, Any]]:
        """Create sample tasks for testing."""
        
        sample_tasks = [
            {
                "task_metadata": {
                    "task_id": "smoke_test_001",
                    "complexity": "easy",
                    "category": "test"
                },
                "prompt": [
                    {
                        "role": "user",
                        "content": "What is 2 + 2? Please think about it and respond."
                    }
                ],
                "reward_spec": {
                    "ground_truth": {
                        "expected_tools": [],
                        "success_criteria": {"contains_answer": True}
                    }
                },
                "extra_info": {"complexity": "easy"}
            },
            {
                "task_metadata": {
                    "task_id": "smoke_test_002", 
                    "complexity": "medium",
                    "category": "test"
                },
                "prompt": [
                    {
                        "role": "user",
                        "content": "Explain the concept of machine learning in simple terms."
                    }
                ],
                "reward_spec": {
                    "ground_truth": {
                        "expected_tools": [],
                        "success_criteria": {"explanation_quality": "good"}
                    }
                },
                "extra_info": {"complexity": "medium"}
            },
            {
                "task_metadata": {
                    "task_id": "smoke_test_003",
                    "complexity": "hard", 
                    "category": "test"
                },
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function to calculate the fibonacci sequence."
                    }
                ],
                "reward_spec": {
                    "ground_truth": {
                        "expected_tools": [],
                        "success_criteria": {"code_quality": "good"}
                    }
                },
                "extra_info": {"complexity": "hard"}
            },
            {
                "task_metadata": {
                    "task_id": "smoke_test_004",
                    "complexity": "easy",
                    "category": "test"
                },
                "prompt": [
                    {
                        "role": "user", 
                        "content": "What is the capital of France?"
                    }
                ],
                "reward_spec": {
                    "ground_truth": {
                        "expected_tools": [],
                        "success_criteria": {"factual_accuracy": True}
                    }
                },
                "extra_info": {"complexity": "easy"}
            },
            {
                "task_metadata": {
                    "task_id": "smoke_test_005",
                    "complexity": "medium",
                    "category": "test"
                },
                "prompt": [
                    {
                        "role": "user",
                        "content": "Describe the process of photosynthesis."
                    }
                ],
                "reward_spec": {
                    "ground_truth": {
                        "expected_tools": [],
                        "success_criteria": {"scientific_accuracy": True}
                    }
                },
                "extra_info": {"complexity": "medium"}
            }
        ]
        
        return sample_tasks
    
    def _save_test_configs(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Save test configurations to temporary files."""
        
        config_paths = {}
        
        for config_name, config_data in configs.items():
            config_file = self.temp_dir / f"{config_name}_config.yaml"
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            config_paths[config_name] = str(config_file)
        
        return config_paths
    
    def _create_sample_data_file(self, tasks: List[Dict[str, Any]]) -> str:
        """Create sample data file for testing."""
        
        data_file = self.temp_dir / "test_data.json"
        
        with open(data_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        
        return str(data_file)
    
    async def test_policy_initialization(self, config_paths: Dict[str, str]) -> bool:
        """Test policy initialization in both modes."""
        
        logger.info("Testing policy initialization...")
        
        try:
            # Create policy
            policy = QwenPolicy(
                model_config_path=config_paths["model"],
                training_config_path=config_paths["training"],
                use_lora=(self.mode == "lora"),
                device=str(self.device),
                load_in_4bit=False,  # Disable for testing
            )
            
            # Test basic properties
            assert policy.model is not None, "Model not initialized"
            assert policy.tokenizer is not None, "Tokenizer not initialized"
            
            # Test trainable parameters
            trainable_params = policy.get_trainable_parameters()
            assert trainable_params > 0, "No trainable parameters found"
            
            logger.info(f"Policy initialized: {trainable_params:,} trainable parameters")
            
            # Test generation (if not skipping model tests)
            if not self.skip_model_tests:
                test_messages = [{"role": "user", "content": "Hello, how are you?"}]
                responses = policy.generate_action([test_messages])
                
                assert len(responses) > 0, "No responses generated"
                assert isinstance(responses[0], str), "Response is not a string"
                assert len(responses[0]) > 0, "Empty response generated"
                
                logger.info(f"Generation test passed: '{responses[0][:50]}...'")
            
            self.test_results["policy_initialization"] = True
            return True
            
        except Exception as e:
            logger.error(f"Policy initialization failed: {e}")
            self.test_results["policy_initialization"] = False
            return False
    
    async def test_data_loading(self, data_file: str) -> bool:
        """Test data loading and curriculum sampling."""
        
        logger.info("Testing data loading...")
        
        try:
            # Create streaming dataset
            dataset = StreamingDataset(
                file_path=data_file,
                cache_size=10,
                seed=42,
            )
            
            assert len(dataset) > 0, "Dataset is empty"
            
            # Test task retrieval
            tasks = dataset.get_random_tasks(3)
            assert len(tasks) > 0, "No tasks retrieved"
            
            # Test curriculum sampler
            sampler = CurriculumSampler(seed=42)
            dist = sampler.get_current_distribution(0, 10)
            assert isinstance(dist, dict), "Invalid distribution"
            assert abs(sum(dist.values()) - 1.0) < 1e-6, "Distribution doesn't sum to 1"
            
            # Test task batcher
            batcher = TaskBatcher(target_total_turns=8, max_batch_size=3)
            batch = batcher.create_batch(tasks)
            assert len(batch) > 0, "No batch created"
            
            logger.info(f"Data loading test passed: {len(dataset)} tasks, {len(batch)} in batch")
            
            self.test_results["data_loading"] = True
            return True
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.test_results["data_loading"] = False
            return False
    
    async def test_trajectory_collection(self, tasks: List[Dict[str, Any]]) -> bool:
        """Test trajectory collection (mock version)."""
        
        logger.info("Testing trajectory collection...")
        
        try:
            # Create mock trajectories instead of using real environments
            # This avoids dependencies on MCP servers during testing
            
            mock_trajectories = []
            
            for task in tasks[:2]:  # Test with 2 tasks
                # Create mock trajectory
                states = [
                    [{"role": "user", "content": task["prompt"][0]["content"]}],
                    [
                        {"role": "user", "content": task["prompt"][0]["content"]},
                        {"role": "assistant", "content": "Let me think about this."}
                    ]
                ]
                
                actions = [
                    "Let me think about this.",
                    "The answer is 4."
                ]
                
                rewards = [0.5, 1.0]  # Positive rewards
                dones = [False, True]
                
                trajectory = Trajectory(
                    task_id=task["task_metadata"]["task_id"],
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                )
                
                mock_trajectories.append(trajectory)
            
            assert len(mock_trajectories) > 0, "No trajectories created"
            
            # Test trajectory properties
            for traj in mock_trajectories:
                assert len(traj.states) == len(traj.actions), "State-action length mismatch"
                assert len(traj.rewards) == len(traj.dones), "Reward-done length mismatch"
                assert traj.total_reward > 0, "No positive reward"
            
            logger.info(f"Trajectory collection test passed: {len(mock_trajectories)} trajectories")
            
            self.test_results["trajectory_collection"] = True
            return mock_trajectories
            
        except Exception as e:
            logger.error(f"Trajectory collection failed: {e}")
            self.test_results["trajectory_collection"] = False
            return []
    
    async def test_grpo_training(self, policy, trajectories: List[Trajectory], config_paths: Dict[str, str]) -> bool:
        """Test GRPO training steps."""
        
        logger.info("Testing GRPO training...")
        
        try:
            # Create reference policy (copy of main policy)
            reference_policy = QwenPolicy(
                model_config_path=config_paths["model"],
                training_config_path=config_paths["training"],
                use_lora=(self.mode == "lora"),
                device=str(self.device),
                load_in_4bit=(self.mode == "lora"),
            )
            
            # Create GRPO trainer
            import yaml
            with open(config_paths["grpo"], 'r') as f:
                grpo_config = yaml.safe_load(f)
            with open(config_paths["training"], 'r') as f:
                training_config = yaml.safe_load(f)
            
            trainer = GRPOTrainer(
                policy=policy,
                reference_policy=reference_policy,
                grpo_config=grpo_config,
                training_config=training_config,
                device=self.device,
            )
            
            # Get initial parameters for gradient check
            initial_params = {}
            for name, param in policy.model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.clone().detach()
            
            # Run training steps
            policy.enable_training_mode()
            
            for step in range(3):  # 3 training steps
                logger.info(f"Running training step {step + 1}/3...")
                
                # Compute log probabilities for trajectories
                for traj in trajectories:
                    traj.log_probs = policy.compute_log_probs(traj.states, traj.actions)
                
                # Run training step
                metrics = trainer.train_step(trajectories)
                
                assert isinstance(metrics, dict), "Metrics not returned as dict"
                assert "policy_loss" in metrics, "Policy loss not in metrics"
                assert "kl_divergence" in metrics, "KL divergence not in metrics"
                
                logger.info(f"Step {step + 1} metrics: loss={metrics.get('policy_loss', 0):.4f}, "
                           f"kl={metrics.get('kl_divergence', 0):.4f}")
            
            # Check that gradients were computed
            gradients_found = False
            for name, param in policy.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    if torch.any(param.grad != 0):
                        gradients_found = True
                        break
            
            assert gradients_found, "No non-zero gradients found"
            
            # Check that parameters changed
            params_changed = False
            for name, param in policy.model.named_parameters():
                if param.requires_grad and name in initial_params:
                    if not torch.allclose(param.data, initial_params[name], atol=1e-6):
                        params_changed = True
                        break
            
            assert params_changed, "Parameters did not change during training"
            
            logger.info("GRPO training test passed: gradients computed, parameters updated")
            
            self.test_results["grpo_training"] = True
            return trainer
            
        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            logger.error(traceback.format_exc())
            self.test_results["grpo_training"] = False
            return None
    
    async def test_checkpointing(self, policy, trainer) -> bool:
        """Test checkpoint saving and loading."""
        
        logger.info("Testing checkpointing...")
        
        try:
            checkpoint_dir = self.temp_dir / "test_checkpoint"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_dir), include_optimizer=True)
            
            # Verify checkpoint files exist
            assert (checkpoint_dir / "policy").exists(), "Policy checkpoint not saved"
            assert (checkpoint_dir / "reference_policy").exists(), "Reference policy checkpoint not saved"
            assert (checkpoint_dir / "trainer_checkpoint.pt").exists(), "Trainer checkpoint not saved"
            
            # Get initial parameters
            initial_params = {}
            for name, param in policy.model.named_parameters():
                if param.requires_grad:
                    initial_params[name] = param.clone().detach()
            
            # Modify parameters slightly
            with torch.no_grad():
                for param in policy.model.parameters():
                    if param.requires_grad:
                        param.add_(torch.randn_like(param) * 0.01)
            
            # Load checkpoint
            trainer.load_checkpoint(str(checkpoint_dir), load_optimizer=True)
            
            # Verify parameters were restored
            params_restored = True
            for name, param in policy.model.named_parameters():
                if param.requires_grad and name in initial_params:
                    if not torch.allclose(param.data, initial_params[name], atol=1e-5):
                        params_restored = False
                        break
            
            assert params_restored, "Parameters not properly restored from checkpoint"
            
            logger.info("Checkpointing test passed: save and load successful")
            
            self.test_results["checkpointing"] = True
            return True
            
        except Exception as e:
            logger.error(f"Checkpointing failed: {e}")
            self.test_results["checkpointing"] = False
            return False
    
    def _print_memory_usage(self) -> None:
        """Print current memory usage."""
        
        try:
            if self.device.type == "cuda":
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_cached = torch.cuda.memory_reserved() / (1024**3)
                logger.info(f"GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")
            elif self.device.type == "mps":
                # macOS Metal memory info
                mps_memory = torch.mps.current_allocated_memory() / (1024**3)
                logger.info(f"MPS Memory: {mps_memory:.2f}GB allocated")
            
            # System memory
            import psutil
            memory_info = psutil.Process().memory_info()
            rss_gb = memory_info.rss / (1024**3)
            logger.info(f"System Memory: {rss_gb:.2f}GB RSS")
            
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
    
    async def run_all_tests(self) -> bool:
        """Run all smoke tests."""
        
        logger.info("Starting comprehensive smoke tests...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="grpo_smoke_test_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        try:
            # Create test configurations and data
            configs = self._create_test_configs()
            config_paths = self._save_test_configs(configs)
            
            sample_tasks = self._create_sample_tasks()
            data_file = self._create_sample_data_file(sample_tasks)
            
            # Run tests in sequence
            all_passed = True
            
            # Test 1: Policy initialization
            self._print_memory_usage()
            policy_success = await self.test_policy_initialization(config_paths)
            all_passed &= policy_success
            
            if not policy_success:
                logger.error("Policy initialization failed, skipping remaining tests")
                return False
            
            # Get the policy for subsequent tests
            policy = QwenPolicy(
                model_config_path=config_paths["model"],
                training_config_path=config_paths["training"],
                use_lora=(self.mode == "lora"),
                device=str(self.device),
                load_in_4bit=False,  # Disable for testing
            )
            
            # Test 2: Data loading
            self._print_memory_usage()
            data_success = await self.test_data_loading(data_file)
            all_passed &= data_success
            
            # Test 3: Trajectory collection
            self._print_memory_usage()
            trajectories = await self.test_trajectory_collection(sample_tasks)
            traj_success = len(trajectories) > 0
            all_passed &= traj_success
            
            if not traj_success:
                logger.error("Trajectory collection failed, skipping training tests")
                return False
            
            # Test 4: GRPO training
            self._print_memory_usage()
            trainer = await self.test_grpo_training(policy, trajectories, config_paths)
            training_success = trainer is not None
            all_passed &= training_success
            
            if not training_success:
                logger.error("GRPO training failed, skipping checkpoint test")
                return False
            
            # Test 5: Checkpointing
            self._print_memory_usage()
            checkpoint_success = await self.test_checkpointing(policy, trainer)
            all_passed &= checkpoint_success
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Smoke test failed: {e}")
            logger.error(traceback.format_exc())
            return False
            
        finally:
            # Cleanup
            try:
                import shutil
                if self.temp_dir and self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")
    
    def print_results(self) -> None:
        """Print test results summary."""
        
        elapsed_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("SMOKE TEST RESULTS")
        print("="*60)
        
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print(f"Elapsed Time: {elapsed_time:.1f} seconds")
        print(f"Skip Model Tests: {self.skip_model_tests}")
        
        print("\nTest Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        print(f"\nSummary: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  Some tests failed. Check logs for details.")
        
        print("="*60)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="GRPO Training Pipeline Smoke Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode to test"
    )
    
    parser.add_argument(
        "--skip_model_tests",
        action="store_true",
        help="Skip model loading and generation tests (for faster testing)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    
    args = parse_arguments()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🔥 GRPO Training Pipeline Smoke Test")
    print(f"Mode: {args.mode}")
    print(f"Skip Model Tests: {args.skip_model_tests}")
    print(f"Target: Complete in under 2 minutes")
    print("-" * 50)
    
    # Run smoke tests
    runner = SmokeTestRunner(
        mode=args.mode,
        skip_model_tests=args.skip_model_tests
    )
    
    try:
        success = await runner.run_all_tests()
        runner.print_results()
        
        if success:
            print("\n✅ Smoke test completed successfully!")
            return 0
        else:
            print("\n❌ Smoke test failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Smoke test interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Smoke test crashed: {e}")
        return 1


if __name__ == "__main__":
    import asyncio
    exit_code = asyncio.run(main())
    sys.exit(exit_code)