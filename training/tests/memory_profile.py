#!/usr/bin/env python3
"""
Memory Profiler for GRPO Training

This script profiles memory usage for different GRPO training configurations
and provides recommendations for optimal settings on different GPU sizes.
Specifically optimized for macOS with 48GB unified memory.

Usage:
    python memory_profile.py
    python memory_profile.py --mode lora --max_batch_size 8
    python memory_profile.py --mode full --sequence_lengths 512,1024
"""

import argparse
import gc
import json
import logging
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import numpy as np

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

# Core training imports
from core.qwen_policy import QwenPolicy, estimate_memory_usage
from core.grpo_trainer import GRPOTrainer, Trajectory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Memory profiler for GRPO training configurations.
    
    This class systematically tests different training configurations
    to measure memory usage and find optimal settings for different
    hardware configurations.
    """
    
    def __init__(self, mode: str = "lora"):
        """Initialize memory profiler."""
        
        self.mode = mode
        self.device = self._get_best_device()
        self.temp_dir = None
        self.results = []
        self.baseline_memory = self._get_current_memory()
        
        logger.info(f"MemoryProfiler initialized: mode={mode}, device={self.device}")
        logger.info(f"Baseline memory usage: {self.baseline_memory:.2f} GB")
        logger.info(f"Total system memory: {self._get_total_memory():.1f} GB")
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device for profiling."""
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")  # macOS Metal Performance Shaders
        else:
            return torch.device("cpu")
    
    def _get_total_memory(self) -> float:
        """Get total system memory in GB."""
        
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 48.0  # Default for macOS with 48GB
    
    def _get_current_memory(self) -> float:
        """Get current memory usage in GB."""
        
        try:
            if self.device.type == "cuda":
                return torch.cuda.memory_allocated() / (1024**3)
            elif self.device.type == "mps":
                return torch.mps.current_allocated_memory() / (1024**3)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**3)
        except Exception:
            return 0.0
    
    def _get_peak_memory(self) -> float:
        """Get peak memory usage in GB."""
        
        try:
            if self.device.type == "cuda":
                return torch.cuda.max_memory_allocated() / (1024**3)
            elif self.device.type == "mps":
                return torch.mps.driver_allocated_memory() / (1024**3)
            else:
                import psutil
                return psutil.Process().memory_info().rss / (1024**3)
        except Exception:
            return 0.0
    
    def _reset_memory_stats(self) -> None:
        """Reset memory statistics."""
        
        try:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Could not reset memory stats: {e}")
    
    def _create_test_config(
        self,
        sequence_length: int = 1024,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """Create test configuration for memory profiling."""
        
        model_config = {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "tokenizer_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "trust_remote_code": True,
            "max_length": sequence_length,
            "context_length": sequence_length,
            "vocab_size": 151936,
            "generation": {
                "max_new_tokens": min(256, sequence_length // 4),
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "pad_token_id": 151643,
                "eos_token_id": 151645,
            },
            "stop_sequences": ["</think>", "</tool_call>", "<|im_end|>"],
            "lora_mode": {
                "enabled": self.mode == "lora",
                "r": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            },
            "full_finetune_mode": {
                "enabled": self.mode == "full",
                "gradient_checkpointing": True,
                "use_flash_attention": False,  # May not be available on macOS
                "bf16": False,  # Use fp32 for consistent profiling
                "fp16": False,
            },
            "memory_optimization": {
                "low_cpu_mem_usage": True,
                "device_map": "auto" if self.device.type == "cuda" else None,
                "torch_dtype": "float32",
            },
            "quantization": {
                "load_in_4bit": False,  # Disable for testing
                "bnb_4bit_compute_dtype": "float32",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }
        }
        
        training_config = {
            "experiment_name": "memory_profile",
            "lora_mode": {
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "full_finetune_mode": {
                "per_device_train_batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            "lora_learning_rate": 2e-4,
            "full_finetune_learning_rate": 5e-6,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "max_grad_norm": 1.0,
        }
        
        grpo_config = {
            "algorithm": "grpo",
            "group_size": batch_size,
            "kl_penalty_coef": 0.1,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "entropy_coef": 0.01,
            "value_loss_coef": 0.5,
        }
        
        return {
            "model": model_config,
            "training": training_config,
            "grpo": grpo_config,
        }
    
    def _save_config_files(self, configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Save configurations to temporary files."""
        
        config_paths = {}
        
        import yaml
        for config_name, config_data in configs.items():
            config_file = self.temp_dir / f"{config_name}_config.yaml"
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f)
            
            config_paths[config_name] = str(config_file)
        
        return config_paths
    
    def _create_sample_trajectories(
        self,
        num_trajectories: int,
        sequence_length: int,
        turns_per_trajectory: int = 3,
    ) -> List[Trajectory]:
        """Create sample trajectories for memory testing."""
        
        trajectories = []
        
        for i in range(num_trajectories):
            states = []
            actions = []
            rewards = []
            dones = []
            
            # Create conversation states
            conversation = []
            
            for turn in range(turns_per_trajectory):
                # Add current conversation state
                states.append(conversation.copy())
                
                # Generate action (padded to test sequence length)
                action = f"This is turn {turn + 1} of trajectory {i + 1}. " * (sequence_length // 50)
                action = action[:sequence_length]  # Truncate to exact length
                actions.append(action)
                
                # Add reward and done flag
                rewards.append(1.0 if turn == turns_per_trajectory - 1 else 0.5)
                dones.append(turn == turns_per_trajectory - 1)
                
                # Update conversation
                conversation.append({"role": "assistant", "content": action})
                if turn < turns_per_trajectory - 1:
                    conversation.append({"role": "user", "content": f"Continue turn {turn + 2}"})
            
            trajectory = Trajectory(
                task_id=f"memory_test_{i:03d}",
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def profile_configuration(
        self,
        sequence_length: int,
        batch_size: int,
        gradient_accumulation_steps: int = 1,
        num_training_steps: int = 3,
    ) -> Dict[str, Any]:
        """Profile memory usage for a specific configuration."""
        
        logger.info(f"Profiling: seq_len={sequence_length}, batch={batch_size}, "
                   f"grad_accum={gradient_accumulation_steps}")
        
        self._reset_memory_stats()
        start_memory = self._get_current_memory()
        
        result = {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mode": self.mode,
            "device": str(self.device),
            "start_memory_gb": start_memory,
            "success": False,
            "error": None,
            "peak_memory_gb": 0.0,
            "model_memory_gb": 0.0,
            "training_memory_gb": 0.0,
            "memory_per_sample": 0.0,
            "recommended_batch_size": 0,
        }
        
        try:
            # Create test configuration
            configs = self._create_test_config(sequence_length, batch_size, gradient_accumulation_steps)
            config_paths = self._save_config_files(configs)
            
            # Initialize policy
            logger.info("  Loading model...")
            policy = QwenPolicy(
                model_config_path=config_paths["model"],
                training_config_path=config_paths["training"],
                use_lora=(self.mode == "lora"),
                device=str(self.device),
                load_in_4bit=False,  # Disable for testing
            )
            
            model_memory = self._get_current_memory()
            result["model_memory_gb"] = model_memory - start_memory
            
            # Create reference policy
            logger.info("  Creating reference policy...")
            reference_policy = QwenPolicy(
                model_config_path=config_paths["model"],
                training_config_path=config_paths["training"],
                use_lora=(self.mode == "lora"),
                device=str(self.device),
                load_in_4bit=False,  # Disable for testing
            )
            
            # Create trainer
            logger.info("  Creating trainer...")
            trainer = GRPOTrainer(
                policy=policy,
                reference_policy=reference_policy,
                grpo_config=configs["grpo"],
                training_config=configs["training"],
                device=self.device,
            )
            
            trainer_memory = self._get_current_memory()
            
            # Create sample trajectories
            trajectories = self._create_sample_trajectories(
                num_trajectories=batch_size,
                sequence_length=sequence_length // 2,  # Half for input, half for generation
                turns_per_trajectory=2,
            )
            
            # Run training steps
            logger.info(f"  Running {num_training_steps} training steps...")
            policy.enable_training_mode()
            
            for step in range(num_training_steps):
                # Compute log probabilities
                for traj in trajectories:
                    traj.log_probs = policy.compute_log_probs(traj.states, traj.actions)
                
                # Training step
                metrics = trainer.train_step(trajectories)
                
                # Check memory after each step
                current_memory = self._get_current_memory()
                logger.debug(f"    Step {step + 1}: {current_memory:.2f} GB")
            
            # Final memory measurements
            final_memory = self._get_current_memory()
            peak_memory = self._get_peak_memory()
            
            result.update({
                "success": True,
                "peak_memory_gb": peak_memory,
                "training_memory_gb": final_memory - trainer_memory,
                "memory_per_sample": (final_memory - trainer_memory) / batch_size if batch_size > 0 else 0,
            })
            
            # Estimate recommended batch size for different GPU sizes
            if peak_memory > 0:
                # Conservative estimates (leave 20% margin)
                gpu_sizes = [8, 16, 24, 32, 40, 48, 80]  # Common GPU memory sizes in GB
                for gpu_size in gpu_sizes:
                    available_memory = gpu_size * 0.8  # 80% utilization
                    if peak_memory <= available_memory:
                        scaling_factor = available_memory / peak_memory
                        recommended_batch = int(batch_size * scaling_factor)
                        result[f"recommended_batch_{gpu_size}gb"] = recommended_batch
            
            logger.info(f"  ✅ Success: {peak_memory:.2f} GB peak memory")
            
        except torch.cuda.OutOfMemoryError as e:
            result["error"] = f"CUDA OOM: {str(e)}"
            logger.warning(f"  ❌ CUDA OOM at configuration")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                result["error"] = f"OOM: {str(e)}"
                logger.warning(f"  ❌ OOM at configuration") 
            else:
                result["error"] = f"Runtime error: {str(e)}"
                logger.error(f"  ❌ Runtime error: {e}")
        
        except Exception as e:
            result["error"] = f"Unexpected error: {str(e)}"
            logger.error(f"  ❌ Unexpected error: {e}")
            logger.error(traceback.format_exc())
        
        finally:
            # Cleanup
            try:
                del policy, reference_policy, trainer, trajectories
                self._reset_memory_stats()
            except:
                pass
        
        return result
    
    def run_batch_size_sweep(
        self,
        sequence_length: int = 1024,
        max_batch_size: int = 8,
        gradient_accumulation_steps: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run memory profiling across different batch sizes."""
        
        logger.info(f"Running batch size sweep: seq_len={sequence_length}, "
                   f"max_batch={max_batch_size}, grad_accum={gradient_accumulation_steps}")
        
        results = []
        
        for batch_size in range(1, max_batch_size + 1):
            try:
                result = self.profile_configuration(
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                results.append(result)
                
                # Stop if we hit OOM
                if not result["success"] and "oom" in result.get("error", "").lower():
                    logger.info(f"Stopping batch size sweep at {batch_size} due to OOM")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Batch size sweep interrupted by user")
                break
        
        return results
    
    def run_sequence_length_sweep(
        self,
        sequence_lengths: List[int],
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> List[Dict[str, Any]]:
        """Run memory profiling across different sequence lengths."""
        
        logger.info(f"Running sequence length sweep: lengths={sequence_lengths}, "
                   f"batch={batch_size}, grad_accum={gradient_accumulation_steps}")
        
        results = []
        
        for seq_len in sequence_lengths:
            try:
                result = self.profile_configuration(
                    sequence_length=seq_len,
                    batch_size=batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                results.append(result)
                
                # Stop if we hit OOM
                if not result["success"] and "oom" in result.get("error", "").lower():
                    logger.info(f"Stopping sequence length sweep at {seq_len} due to OOM")
                    break
                    
            except KeyboardInterrupt:
                logger.info("Sequence length sweep interrupted by user")
                break
        
        return results
    
    def run_gradient_accumulation_sweep(
        self,
        sequence_length: int = 1024,
        batch_size: int = 1,
        max_gradient_accumulation: int = 16,
    ) -> List[Dict[str, Any]]:
        """Run memory profiling across different gradient accumulation steps."""
        
        logger.info(f"Running gradient accumulation sweep: seq_len={sequence_length}, "
                   f"batch={batch_size}, max_grad_accum={max_gradient_accumulation}")
        
        results = []
        
        grad_accum_steps = [1, 2, 4, 8, 16]
        grad_accum_steps = [x for x in grad_accum_steps if x <= max_gradient_accumulation]
        
        for grad_accum in grad_accum_steps:
            try:
                result = self.profile_configuration(
                    sequence_length=sequence_length,
                    batch_size=batch_size,
                    gradient_accumulation_steps=grad_accum,
                )
                results.append(result)
                
            except KeyboardInterrupt:
                logger.info("Gradient accumulation sweep interrupted by user")
                break
        
        return results
    
    def generate_recommendations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate optimization recommendations based on profiling results."""
        
        # Filter successful results
        successful_results = [r for r in results if r["success"]]
        
        if not successful_results:
            return {"error": "No successful configurations found"}
        
        # Find optimal configurations
        recommendations = {
            "mode": self.mode,
            "device": str(self.device),
            "total_configurations_tested": len(results),
            "successful_configurations": len(successful_results),
            "baseline_memory_gb": self.baseline_memory,
        }
        
        # Best configuration by memory efficiency (lowest memory per sample)
        if successful_results:
            best_efficiency = min(successful_results, key=lambda x: x.get("memory_per_sample", float('inf')))
            recommendations["most_memory_efficient"] = {
                "sequence_length": best_efficiency["sequence_length"],
                "batch_size": best_efficiency["batch_size"],
                "gradient_accumulation_steps": best_efficiency["gradient_accumulation_steps"],
                "peak_memory_gb": best_efficiency["peak_memory_gb"],
                "memory_per_sample": best_efficiency["memory_per_sample"],
            }
        
        # GPU-specific recommendations
        gpu_recommendations = {}
        gpu_sizes = [8, 16, 24, 32, 40, 48, 80]
        
        for gpu_size in gpu_sizes:
            # Find configurations that fit in this GPU size (with 80% utilization)
            available_memory = gpu_size * 0.8
            fitting_configs = [r for r in successful_results if r["peak_memory_gb"] <= available_memory]
            
            if fitting_configs:
                # Find best batch size for this GPU
                best_config = max(fitting_configs, key=lambda x: x["batch_size"])
                
                gpu_recommendations[f"{gpu_size}gb"] = {
                    "max_batch_size": best_config["batch_size"],
                    "recommended_sequence_length": best_config["sequence_length"],
                    "gradient_accumulation_steps": best_config["gradient_accumulation_steps"],
                    "expected_memory_usage": best_config["peak_memory_gb"],
                    "memory_utilization": (best_config["peak_memory_gb"] / gpu_size) * 100,
                }
        
        recommendations["gpu_specific"] = gpu_recommendations
        
        # Memory scaling analysis
        if len(successful_results) > 1:
            # Analyze memory scaling with batch size
            batch_size_results = {}
            for r in successful_results:
                batch_size = r["batch_size"]
                if batch_size not in batch_size_results:
                    batch_size_results[batch_size] = []
                batch_size_results[batch_size].append(r["peak_memory_gb"])
            
            scaling_analysis = {}
            for batch_size, memories in batch_size_results.items():
                scaling_analysis[batch_size] = {
                    "avg_memory_gb": np.mean(memories),
                    "min_memory_gb": np.min(memories),
                    "max_memory_gb": np.max(memories),
                }
            
            recommendations["memory_scaling"] = scaling_analysis
        
        # Optimization suggestions
        optimization_tips = []
        
        if self.mode == "lora":
            optimization_tips.extend([
                "LoRA mode uses 4-bit quantization to reduce memory usage",
                "Consider increasing batch size for better GPU utilization",
                "Gradient accumulation can simulate larger batches without memory increase",
            ])
        else:
            optimization_tips.extend([
                "Full fine-tuning requires more memory but may give better results",
                "Use gradient checkpointing to trade compute for memory",
                "Consider bf16 precision for better memory efficiency on supported hardware",
            ])
        
        # macOS specific tips
        if self.device.type == "mps":
            optimization_tips.extend([
                "macOS Metal Performance Shaders (MPS) provides unified memory access",
                "Memory is shared between CPU and GPU on Apple Silicon",
                "Consider using smaller sequence lengths for better performance",
            ])
        
        recommendations["optimization_tips"] = optimization_tips
        
        return recommendations
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None) -> str:
        """Save profiling results to JSON file."""
        
        if filename is None:
            timestamp = int(time.time())
            filename = f"memory_profile_{self.mode}_{timestamp}.json"
        
        filepath = Path(filename)
        
        # Include recommendations in the saved results
        recommendations = self.generate_recommendations(results)
        
        output_data = {
            "metadata": {
                "mode": self.mode,
                "device": str(self.device),
                "timestamp": time.time(),
                "total_system_memory_gb": self._get_total_memory(),
                "baseline_memory_gb": self.baseline_memory,
            },
            "results": results,
            "recommendations": recommendations,
        }
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def print_summary(self, results: List[Dict[str, Any]]) -> None:
        """Print a summary of profiling results."""
        
        print("\n" + "="*80)
        print("MEMORY PROFILING RESULTS")
        print("="*80)
        
        print(f"Mode: {self.mode}")
        print(f"Device: {self.device}")
        print(f"Total System Memory: {self._get_total_memory():.1f} GB")
        print(f"Baseline Memory: {self.baseline_memory:.2f} GB")
        
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        print(f"\nConfigurations Tested: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print("\n📊 SUCCESSFUL CONFIGURATIONS:")
            print("-" * 80)
            print(f"{'SeqLen':>8} {'Batch':>8} {'GradAcc':>8} {'Peak(GB)':>10} {'Mem/Sample':>12}")
            print("-" * 80)
            
            for result in successful_results:
                print(f"{result['sequence_length']:>8} "
                      f"{result['batch_size']:>8} "
                      f"{result['gradient_accumulation_steps']:>8} "
                      f"{result['peak_memory_gb']:>10.2f} "
                      f"{result['memory_per_sample']:>12.3f}")
        
        if failed_results:
            print(f"\n❌ FAILED CONFIGURATIONS: {len(failed_results)}")
            for result in failed_results:
                error_short = result.get('error', 'Unknown')[:50]
                print(f"  SeqLen={result['sequence_length']}, Batch={result['batch_size']}: {error_short}")
        
        # Generate and display recommendations
        recommendations = self.generate_recommendations(results)
        
        if "gpu_specific" in recommendations:
            print("\n🎯 GPU-SPECIFIC RECOMMENDATIONS:")
            print("-" * 80)
            
            for gpu_size, rec in recommendations["gpu_specific"].items():
                print(f"{gpu_size.upper()}: "
                      f"Max Batch = {rec['max_batch_size']}, "
                      f"Seq Len = {rec['recommended_sequence_length']}, "
                      f"Memory = {rec['expected_memory_usage']:.1f}GB "
                      f"({rec['memory_utilization']:.1f}% util)")
        
        if "optimization_tips" in recommendations:
            print(f"\n💡 OPTIMIZATION TIPS:")
            for tip in recommendations["optimization_tips"]:
                print(f"  • {tip}")
        
        print("="*80)
    
    def run_comprehensive_profile(
        self,
        sequence_lengths: List[int],
        max_batch_size: int,
        max_gradient_accumulation: int,
    ) -> List[Dict[str, Any]]:
        """Run comprehensive memory profiling."""
        
        logger.info("Starting comprehensive memory profiling...")
        
        # Create temporary directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="grpo_memory_profile_"))
        logger.info(f"Using temporary directory: {self.temp_dir}")
        
        all_results = []
        
        try:
            # Test different sequence lengths with batch size 1
            logger.info("Phase 1: Sequence length sweep")
            seq_results = self.run_sequence_length_sweep(
                sequence_lengths=sequence_lengths,
                batch_size=1,
                gradient_accumulation_steps=1,
            )
            all_results.extend(seq_results)
            
            # Find maximum viable sequence length
            viable_seq_lengths = [r["sequence_length"] for r in seq_results if r["success"]]
            if not viable_seq_lengths:
                logger.error("No viable sequence lengths found!")
                return all_results
            
            max_seq_len = max(viable_seq_lengths)
            logger.info(f"Maximum viable sequence length: {max_seq_len}")
            
            # Test batch sizes with maximum sequence length
            logger.info("Phase 2: Batch size sweep")
            batch_results = self.run_batch_size_sweep(
                sequence_length=max_seq_len,
                max_batch_size=max_batch_size,
                gradient_accumulation_steps=1,
            )
            all_results.extend(batch_results)
            
            # Test gradient accumulation with optimal settings
            viable_batch_sizes = [r["batch_size"] for r in batch_results if r["success"]]
            if viable_batch_sizes:
                optimal_batch_size = max(viable_batch_sizes)
                
                logger.info("Phase 3: Gradient accumulation sweep")
                grad_results = self.run_gradient_accumulation_sweep(
                    sequence_length=max_seq_len,
                    batch_size=optimal_batch_size,
                    max_gradient_accumulation=max_gradient_accumulation,
                )
                all_results.extend(grad_results)
            
            return all_results
            
        finally:
            # Cleanup temporary directory
            try:
                import shutil
                if self.temp_dir and self.temp_dir.exists():
                    shutil.rmtree(self.temp_dir)
                    logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Memory Profiler for GRPO Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lora", "full"],
        default="lora",
        help="Training mode to profile"
    )
    
    parser.add_argument(
        "--sequence_lengths",
        type=str,
        default="512,1024,2048",
        help="Comma-separated list of sequence lengths to test"
    )
    
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size to test"
    )
    
    parser.add_argument(
        "--max_gradient_accumulation",
        type=int,
        default=16,
        help="Maximum gradient accumulation steps to test"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick profile (fewer configurations)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_arguments()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Parse sequence lengths
    sequence_lengths = [int(x.strip()) for x in args.sequence_lengths.split(",")]
    
    print("🔍 GRPO Memory Profiler")
    print(f"Mode: {args.mode}")
    print(f"Sequence lengths: {sequence_lengths}")
    print(f"Max batch size: {args.max_batch_size}")
    print(f"Max gradient accumulation: {args.max_gradient_accumulation}")
    print("-" * 50)
    
    # Adjust parameters for quick mode
    if args.quick:
        sequence_lengths = sequence_lengths[:2]  # Test fewer sequence lengths
        max_batch_size = min(args.max_batch_size, 4)  # Smaller batch size range
        max_gradient_accumulation = min(args.max_gradient_accumulation, 8)
        print("⚡ Quick mode enabled - testing fewer configurations")
    else:
        max_batch_size = args.max_batch_size
        max_gradient_accumulation = args.max_gradient_accumulation
    
    # Create profiler and run comprehensive profile
    profiler = MemoryProfiler(mode=args.mode)
    
    try:
        results = profiler.run_comprehensive_profile(
            sequence_lengths=sequence_lengths,
            max_batch_size=max_batch_size,
            max_gradient_accumulation=max_gradient_accumulation,
        )
        
        # Print summary
        profiler.print_summary(results)
        
        # Save results
        output_file = args.output_file or f"memory_profile_{args.mode}.json"
        saved_file = profiler.save_results(results, output_file)
        
        print(f"\n✅ Memory profiling completed!")
        print(f"📄 Results saved to: {saved_file}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Memory profiling interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Memory profiling failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)