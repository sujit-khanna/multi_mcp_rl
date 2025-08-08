"""
Logging Utility Functions for GRPO Training

This module provides comprehensive logging capabilities for distributed training including:
- Distributed logging with per-rank files
- Metrics aggregation across GPUs
- Pretty printing for training progress
- WandB integration with proper configuration
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TextIO
import warnings

import torch
import numpy as np

# WandB integration (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available. WandB logging disabled.")

# Weave integration (optional)
try:
    import weave
    WEAVE_AVAILABLE = True
except ImportError:
    WEAVE_AVAILABLE = False
    warnings.warn("weave not available. Weave logging disabled.")

# Set up module logger
logger = logging.getLogger(__name__)


class DistributedFormatter(logging.Formatter):
    """Custom formatter that includes rank information."""
    
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        super().__init__()
    
    def format(self, record):
        # Add rank info to the record
        if self.world_size > 1:
            record.rank_info = f"[Rank {self.rank}/{self.world_size}]"
        else:
            record.rank_info = ""
        
        # Format: timestamp - rank - level - logger - message
        if self.world_size > 1:
            fmt = "%(asctime)s - %(rank_info)s - %(levelname)s - %(name)s - %(message)s"
        else:
            fmt = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_distributed_logging(
    log_dir: Union[str, Path],
    rank: int = 0,
    world_size: int = 1,
    log_level: str = "INFO",
    console_log_level: str = "INFO",
    file_log_level: str = "DEBUG",
) -> logging.Logger:
    """
    Setup distributed logging with separate files per rank.
    
    Args:
        log_dir: Directory to store log files
        rank: Process rank
        world_size: Total number of processes
        log_level: Overall log level
        console_log_level: Console log level
        file_log_level: File log level
        
    Returns:
        Configured logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create rank-specific logger name
    logger_name = f"grpo_training_rank_{rank}"
    training_logger = logging.getLogger(logger_name)
    
    # Clear any existing handlers
    training_logger.handlers.clear()
    training_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create distributed formatter
    formatter = DistributedFormatter(rank=rank, world_size=world_size)
    
    # File handler (rank-specific file)
    log_file = log_dir / f"training_rank_{rank:02d}.log"
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(getattr(logging, file_log_level.upper()))
    file_handler.setFormatter(formatter)
    training_logger.addHandler(file_handler)
    
    # Console handler (only rank 0 logs to console to avoid spam)
    if rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, console_log_level.upper()))
        console_handler.setFormatter(formatter)
        training_logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    training_logger.propagate = False
    
    # Log initialization
    training_logger.info(f"Distributed logging initialized for rank {rank}/{world_size}")
    training_logger.info(f"Log file: {log_file}")
    
    return training_logger


class MetricsAggregator:
    """
    Aggregates metrics across distributed processes.
    """
    
    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.metrics_history = []
        
    def aggregate_metrics(
        self,
        metrics: Dict[str, float],
        reduction: str = "mean"
    ) -> Dict[str, float]:
        """
        Aggregate metrics across all processes.
        
        Args:
            metrics: Dictionary of metrics to aggregate
            reduction: Type of reduction ("mean", "sum", "max", "min")
            
        Returns:
            Aggregated metrics
        """
        if self.world_size == 1:
            return metrics.copy()
        
        try:
            aggregated = {}
            
            for key, value in metrics.items():
                if not isinstance(value, (int, float)):
                    # Skip non-numeric values
                    if self.rank == 0:
                        aggregated[key] = value
                    continue
                
                # Convert to tensor for distributed operations
                tensor_value = torch.tensor(float(value), dtype=torch.float32)
                
                if torch.distributed.is_initialized():
                    # Perform distributed reduction
                    if reduction == "mean":
                        torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.SUM)
                        tensor_value = tensor_value / self.world_size
                    elif reduction == "sum":
                        torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.SUM)
                    elif reduction == "max":
                        torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.MAX)
                    elif reduction == "min":
                        torch.distributed.all_reduce(tensor_value, op=torch.distributed.ReduceOp.MIN)
                    else:
                        raise ValueError(f"Unknown reduction type: {reduction}")
                
                aggregated[key] = tensor_value.item()
            
            return aggregated
            
        except Exception as e:
            logger.warning(f"Failed to aggregate metrics: {e}")
            return metrics.copy()
    
    def add_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Add metrics to history."""
        metrics_entry = {
            "step": step,
            "timestamp": time.time(),
            "metrics": metrics.copy()
        }
        self.metrics_history.append(metrics_entry)
    
    def get_metrics_summary(self, last_n_steps: int = 100) -> Dict[str, Any]:
        """Get summary statistics for recent metrics."""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.metrics_history[-last_n_steps:]
        
        # Extract all metric keys
        all_keys = set()
        for entry in recent_metrics:
            all_keys.update(entry["metrics"].keys())
        
        summary = {}
        for key in all_keys:
            values = []
            for entry in recent_metrics:
                if key in entry["metrics"] and isinstance(entry["metrics"][key], (int, float)):
                    values.append(entry["metrics"][key])
            
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "latest": values[-1] if values else None,
                }
        
        return summary


class TrainingLogger:
    """
    Comprehensive training logger with pretty printing and progress tracking.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        metrics_aggregator: Optional[MetricsAggregator] = None,
        log_every_n_steps: int = 10,
        save_metrics_every_n_steps: int = 100,
        metrics_file: Optional[Union[str, Path]] = None,
    ):
        self.logger = logger
        self.metrics_aggregator = metrics_aggregator or MetricsAggregator()
        self.log_every_n_steps = log_every_n_steps
        self.save_metrics_every_n_steps = save_metrics_every_n_steps
        self.metrics_file = Path(metrics_file) if metrics_file else None
        
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.step_times = []
        
    def log_training_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Log training step with metrics and progress."""
        
        # Add metrics to aggregator
        self.metrics_aggregator.add_metrics(metrics, step)
        
        # Aggregate across processes
        aggregated_metrics = self.metrics_aggregator.aggregate_metrics(metrics)
        
        # Calculate timing information
        current_time = time.time()
        step_time = current_time - self.last_log_time
        self.step_times.append(step_time)
        
        # Keep only recent step times for moving average
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]
        
        avg_step_time = np.mean(self.step_times)
        
        # Log periodically
        if step % self.log_every_n_steps == 0:
            self._log_progress(step, epoch, aggregated_metrics, avg_step_time, model)
        
        # Save metrics periodically
        if step % self.save_metrics_every_n_steps == 0 and self.metrics_file:
            self._save_metrics_to_file()
        
        self.last_log_time = current_time
    
    def _log_progress(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        avg_step_time: float,
        model: Optional[torch.nn.Module] = None,
    ) -> None:
        """Log training progress with pretty formatting."""
        
        # Calculate total training time
        total_time = time.time() - self.start_time
        
        # Format metrics for display
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if abs(value) >= 1000:
                    metric_strs.append(f"{key}={value:.2e}")
                elif abs(value) >= 1:
                    metric_strs.append(f"{key}={value:.4f}")
                else:
                    metric_strs.append(f"{key}={value:.6f}")
            else:
                metric_strs.append(f"{key}={value}")
        
        metrics_str = " | ".join(metric_strs)
        
        # Calculate steps per second
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0
        
        # Log main progress line
        self.logger.info(
            f"Epoch {epoch:3d} | Step {step:6d} | "
            f"Time: {total_time:8.1f}s | "
            f"Step/s: {steps_per_sec:5.2f} | "
            f"{metrics_str}"
        )
        
        # Log additional model information if available
        if model is not None and step % (self.log_every_n_steps * 10) == 0:
            self._log_model_info(model)
    
    def _log_model_info(self, model: torch.nn.Module) -> None:
        """Log additional model information."""
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Calculate gradient norms
            total_grad_norm = 0.0
            param_count = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                total_grad_norm = total_grad_norm ** (1. / 2)
            
            self.logger.info(
                f"Model Info | Total params: {total_params:,} | "
                f"Trainable: {trainable_params:,} | "
                f"Grad norm: {total_grad_norm:.6f}"
            )
            
        except Exception as e:
            self.logger.debug(f"Could not log model info: {e}")
    
    def _save_metrics_to_file(self) -> None:
        """Save metrics history to file."""
        if not self.metrics_file:
            return
        
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(self.metrics_file, 'w') as f:
                json.dump({
                    "metrics_history": self.metrics_aggregator.metrics_history,
                    "summary": self.metrics_aggregator.get_metrics_summary(),
                    "metadata": {
                        "start_time": self.start_time,
                        "last_update": time.time(),
                        "total_steps": len(self.metrics_aggregator.metrics_history),
                    }
                }, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save metrics to file: {e}")
    
    def log_epoch_summary(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch summary."""
        metrics_str = " | ".join([
            f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ])
        
        self.logger.info(f"EPOCH {epoch} SUMMARY | {metrics_str}")
    
    def log_training_start(self, config: Dict[str, Any]) -> None:
        """Log training start with configuration."""
        self.logger.info("="*80)
        self.logger.info("GRPO TRAINING STARTED")
        self.logger.info("="*80)
        
        # Log key configuration parameters
        if "model" in config:
            model_config = config["model"]
            self.logger.info(f"Model: {model_config.get('model_name', 'Unknown')}")
            self.logger.info(f"Max Length: {model_config.get('max_length', 'Unknown')}")
            self.logger.info(f"LoRA Mode: {model_config.get('lora_mode', {}).get('enabled', False)}")
        
        if "training" in config:
            training_config = config["training"]
            self.logger.info(f"Learning Rate: {training_config.get('learning_rate', 'Unknown')}")
            self.logger.info(f"Batch Size: {training_config.get('batch_size', 'Unknown')}")
            self.logger.info(f"Max Steps: {training_config.get('max_steps', 'Unknown')}")
        
        self.logger.info("="*80)
    
    def log_training_end(self, total_time: float, final_metrics: Dict[str, float]) -> None:
        """Log training completion."""
        self.logger.info("="*80)
        self.logger.info("GRPO TRAINING COMPLETED")
        self.logger.info("="*80)
        self.logger.info(f"Total Training Time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)")
        
        # Log final metrics
        for key, value in final_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"Final {key}: {value:.6f}")
            else:
                self.logger.info(f"Final {key}: {value}")
        
        self.logger.info("="*80)


def setup_wandb_logging(
    project_name: str,
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    resume: Optional[str] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Optional[object]:
    """
    Setup WandB logging for training.
    
    Args:
        project_name: WandB project name
        run_name: Run name (if None, WandB will generate one)
        config: Configuration to log
        resume: Resume mode ("allow", "must", "never", or run_id)
        rank: Process rank
        world_size: Total number of processes
        
    Returns:
        WandB run object or None if not available
    """
    if not WANDB_AVAILABLE:
        logger.warning("WandB not available, skipping WandB logging setup")
        return None
    
    # Only initialize WandB on rank 0 to avoid multiple runs
    if rank != 0:
        return None
    
    try:
        # Initialize WandB
        run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            resume=resume,
            settings=wandb.Settings(start_method="fork")  # Better for multiprocessing
        )
        
        logger.info(f"WandB initialized: project={project_name}, run_id={run.id}")
        
        # Log system information
        if config:
            wandb.config.update({
                "distributed_training": world_size > 1,
                "world_size": world_size,
                "rank": rank,
            })
        
        return run
        
    except Exception as e:
        logger.error(f"Failed to initialize WandB: {e}")
        return None


def log_metrics_to_wandb(
    metrics: Dict[str, float],
    step: int,
    prefix: str = "",
    wandb_run: Optional[object] = None,
) -> None:
    """
    Log metrics to WandB.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Training step
        prefix: Prefix to add to metric names
        wandb_run: WandB run object
    """
    if not WANDB_AVAILABLE or wandb_run is None:
        return
    
    try:
        # Add prefix to metric names
        if prefix:
            prefixed_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            prefixed_metrics = metrics.copy()
        
        # Log to WandB
        wandb.log(prefixed_metrics, step=step)
        
    except Exception as e:
        logger.warning(f"Failed to log metrics to WandB: {e}")


class WandBLogger:
    """Wrapper class for WandB logging with error handling."""
    
    def __init__(self, wandb_run: Optional[object] = None):
        self.wandb_run = wandb_run
        self.enabled = wandb_run is not None
    
    def log(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics with error handling."""
        if self.enabled:
            log_metrics_to_wandb(metrics, step, prefix, self.wandb_run)
    
    def log_model_gradients(self, model: torch.nn.Module, step: int) -> None:
        """Log model gradient histograms."""
        if not self.enabled:
            return
        
        try:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    wandb.log({
                        f"gradients/{name}": wandb.Histogram(param.grad.cpu().numpy()),
                    }, step=step)
                    
        except Exception as e:
            logger.warning(f"Failed to log gradients to WandB: {e}")
    
    def log_model_parameters(self, model: torch.nn.Module, step: int) -> None:
        """Log model parameter histograms."""
        if not self.enabled:
            return
        
        try:
            for name, param in model.named_parameters():
                wandb.log({
                    f"parameters/{name}": wandb.Histogram(param.detach().cpu().numpy()),
                }, step=step)
                
        except Exception as e:
            logger.warning(f"Failed to log parameters to WandB: {e}")
    
    def finish(self) -> None:
        """Finish WandB run."""
        if self.enabled:
            try:
                wandb.finish()
            except Exception as e:
                logger.warning(f"Failed to finish WandB run: {e}")


class WeaveLogger:
    """Wrapper class for Weave logging with error handling."""
    
    def __init__(self, project_name: str = "synergia_Agents/skyrl-qwen-0.6b", enabled: bool = True):
        self.project_name = project_name
        self.enabled = enabled and WEAVE_AVAILABLE
        self.weave_initialized = False
        
        if self.enabled:
            try:
                weave.init(project_name)
                self.weave_initialized = True
                logger.info(f"✅ Weave initialized for project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Weave: {e}")
                self.enabled = False
    
    def log_episode(
        self,
        episode_data: Dict[str, Any],
        step: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log episode data to Weave."""
        if not self.enabled:
            return
        
        try:
            # Create episode record
            episode_record = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                **episode_data
            }
            
            if metadata:
                episode_record["metadata"] = metadata
            
            # Log to Weave using publish
            weave.publish(episode_record, name=f"episode_{step}")
            
        except Exception as e:
            logger.warning(f"Failed to log episode to Weave: {e}")
    
    def log_training_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        stage: str = "training"
    ) -> None:
        """Log training metrics to Weave."""
        if not self.enabled:
            return
        
        try:
            metrics_record = {
                "step": step,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics
            }
            
            weave.publish(metrics_record, name=f"training_metrics_{step}")
            
        except Exception as e:
            logger.warning(f"Failed to log training metrics to Weave: {e}")
    
    def log_model_evaluation(
        self,
        evaluation_results: Dict[str, Any],
        step: int,
        model_checkpoint: Optional[str] = None
    ) -> None:
        """Log model evaluation results to Weave."""
        if not self.enabled:
            return
        
        try:
            eval_record = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "evaluation": evaluation_results
            }
            
            if model_checkpoint:
                eval_record["model_checkpoint"] = model_checkpoint
            
            weave.publish(eval_record, name=f"evaluation_{step}")
            
        except Exception as e:
            logger.warning(f"Failed to log evaluation to Weave: {e}")
    
    def finish(self) -> None:
        """Finish Weave session."""
        if self.enabled and self.weave_initialized:
            try:
                # Weave doesn't have an explicit finish method like WandB
                logger.info("✅ Weave logging session completed")
            except Exception as e:
                logger.warning(f"Failed to finish Weave session: {e}")


class EnhancedTrainingLogger:
    """Enhanced training logger that combines WandB and Weave logging."""
    
    def __init__(
        self,
        project_name: str = "skyrl-qwen-grpo",
        weave_project: str = "synergia_Agents/skyrl-qwen-0.6b",
        wandb_config: Optional[Dict[str, Any]] = None,
        enable_wandb: bool = True,
        enable_weave: bool = True,
        rank: int = 0,
        world_size: int = 1
    ):
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Initialize WandB (only on main process)
        self.wandb_logger = None
        if enable_wandb and WANDB_AVAILABLE and self.is_main_process:
            try:
                wandb_run = wandb.init(
                    project=project_name,
                    config=wandb_config or {},
                    save_code=True,
                    group=f"grpo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    job_type="training"
                )
                self.wandb_logger = WandBLogger(wandb_run)
                logger.info(f"✅ WandB initialized for project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize WandB: {e}")
        
        # Initialize Weave (only on main process) 
        # Use same project name as WandB to avoid project mismatch
        self.weave_logger = None
        if enable_weave and self.is_main_process:
            # Use WandB project name for Weave if WandB is enabled
            actual_weave_project = f"synergia_Agents/{project_name}" if enable_wandb else weave_project
            self.weave_logger = WeaveLogger(actual_weave_project, enabled=True)
        
        # Training state tracking
        self.training_start_time = time.time()
        self.episode_count = 0
        self.step_count = 0
    
    def log_training_step(
        self,
        metrics: Dict[str, float],
        step: int,
        episode_data: Optional[Dict[str, Any]] = None,
        stage: str = "training"
    ) -> None:
        """Log a training step with both WandB and Weave."""
        if not self.is_main_process:
            return
        
        self.step_count = step
        
        # Add training time to metrics
        training_time = time.time() - self.training_start_time
        metrics_with_time = {
            **metrics,
            "training_time_hours": training_time / 3600,
            "steps_per_second": step / training_time if training_time > 0 else 0
        }
        
        # Log to WandB
        if self.wandb_logger:
            self.wandb_logger.log(metrics_with_time, step, prefix=stage)
        
        # Log to Weave
        if self.weave_logger:
            self.weave_logger.log_training_metrics(metrics_with_time, step, stage)
            
            # Log episode data if provided
            if episode_data:
                self.weave_logger.log_episode(episode_data, step)
                self.episode_count += 1
    
    def log_episode_results(
        self,
        episode_results: Dict[str, Any],
        step: int,
        task_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log detailed episode results."""
        if not self.is_main_process:
            return
        
        # Prepare episode data
        episode_data = {
            "episode_id": self.episode_count,
            "results": episode_results,
            "task_metadata": task_metadata or {}
        }
        
        # Log to Weave (more detailed episode tracking)
        if self.weave_logger:
            self.weave_logger.log_episode(episode_data, step, metadata=task_metadata)
        
        # Log summary metrics to WandB
        if self.wandb_logger and "reward_breakdown" in episode_results:
            reward_metrics = {f"episode/{k}": v for k, v in episode_results["reward_breakdown"].items()}
            self.wandb_logger.log(reward_metrics, step, prefix="episode")
        
        self.episode_count += 1
    
    def log_model_evaluation(
        self,
        evaluation_results: Dict[str, Any],
        step: int,
        model_checkpoint: Optional[str] = None
    ) -> None:
        """Log model evaluation results."""
        if not self.is_main_process:
            return
        
        # Log to WandB
        if self.wandb_logger:
            eval_metrics = {f"eval/{k}": v for k, v in evaluation_results.items() if isinstance(v, (int, float))}
            self.wandb_logger.log(eval_metrics, step, prefix="evaluation")
        
        # Log to Weave (full evaluation details)
        if self.weave_logger:
            self.weave_logger.log_model_evaluation(evaluation_results, step, model_checkpoint)
    
    def log_model_parameters(self, model: torch.nn.Module, step: int) -> None:
        """Log model parameter and gradient information."""
        if not self.is_main_process:
            return
        
        if self.wandb_logger:
            self.wandb_logger.log_model_gradients(model, step)
            self.wandb_logger.log_model_parameters(model, step)
    
    def log_hyperparameters(self, config: Dict[str, Any]) -> None:
        """Log hyperparameters to both WandB and Weave."""
        if not self.is_main_process:
            return
        
        # Log to WandB config
        if self.wandb_logger and hasattr(self.wandb_logger.wandb_run, 'config'):
            try:
                self.wandb_logger.wandb_run.config.update(config)
            except Exception as e:
                logger.warning(f"Failed to update WandB config: {e}")
        
        # Log to Weave
        if self.weave_logger:
            self.weave_logger.log_training_metrics(
                {"hyperparameters": config}, 
                step=0, 
                stage="config"
            )
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Get summary metrics for the training session."""
        training_time = time.time() - self.training_start_time
        
        return {
            "total_episodes": self.episode_count,
            "total_steps": self.step_count,
            "training_time_hours": training_time / 3600,
            "episodes_per_hour": self.episode_count / (training_time / 3600) if training_time > 0 else 0,
            "steps_per_hour": self.step_count / (training_time / 3600) if training_time > 0 else 0
        }
    
    def finish(self) -> None:
        """Finish both WandB and Weave sessions."""
        if not self.is_main_process:
            return
        
        # Log final summary
        summary_metrics = self.get_summary_metrics()
        
        if self.wandb_logger:
            self.wandb_logger.log(summary_metrics, self.step_count, prefix="summary")
            self.wandb_logger.finish()
        
        if self.weave_logger:
            self.weave_logger.log_training_metrics(summary_metrics, self.step_count, "summary")
            self.weave_logger.finish()
        
        logger.info(f"✅ Training logging completed. Summary: {summary_metrics}")


# Convenience functions for creating loggers

def create_enhanced_training_logger(
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
    enable_wandb: bool = True,
    enable_weave: bool = True
) -> EnhancedTrainingLogger:
    """Create an enhanced training logger with configuration."""
    
    # Extract logging configuration
    logging_config = config.get('logging', {})
    
    # Project names
    wandb_project = logging_config.get('wandb_project', 'skyrl-qwen-grpo')
    weave_project = logging_config.get('weave_project', 'synergia_Agents/skyrl-qwen-0.6b')
    
    # WandB configuration
    wandb_config = {
        'model': config.get('model', {}),
        'training': config.get('training', {}),
        'grpo': config.get('grpo', {}),
        'environment': config.get('environment', {}),
        'rank': rank,
        'world_size': world_size
    }
    
    return EnhancedTrainingLogger(
        project_name=wandb_project,
        weave_project=weave_project,
        wandb_config=wandb_config,
        enable_wandb=enable_wandb,
        enable_weave=enable_weave,
        rank=rank,
        world_size=world_size
    )


def setup_experiment_logging(
    experiment_name: str,
    config: Dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
    tags: Optional[List[str]] = None
) -> EnhancedTrainingLogger:
    """Setup experiment logging with custom experiment name and tags."""
    
    # Create WandB config with experiment metadata
    wandb_config = {
        'experiment_name': experiment_name,
        'tags': tags or [],
        **config
    }
    
    return EnhancedTrainingLogger(
        project_name=f"skyrl-qwen-grpo-{experiment_name}",
        weave_project="synergia_Agents/skyrl-qwen-0.6b",
        wandb_config=wandb_config,
        enable_wandb=True,
        enable_weave=True,
        rank=rank,
        world_size=world_size
    )