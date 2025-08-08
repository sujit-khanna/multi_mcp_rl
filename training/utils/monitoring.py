"""
Training Monitoring Utilities for GRPO Training

This module provides comprehensive monitoring capabilities including:
- GPU memory monitoring with device-specific support
- Training speed tracking (tokens/second, examples/second)
- Gradient norm tracking and analysis
- Early stopping logic based on metrics
- Comprehensive training performance monitoring
"""

import logging
import psutil
import time
from collections import deque
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
import warnings

import torch
import numpy as np

# Set up module logger
logger = logging.getLogger(__name__)


class GPUMonitor:
    """
    GPU memory and utilization monitoring.
    
    Supports CUDA, MPS (Apple Silicon), and CPU fallback.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize GPU monitor.
        
        Args:
            device: Device to monitor (if None, auto-detect best device)
        """
        if device is None:
            device = self._get_best_device()
        
        self.device = device
        self.device_type = device.type
        
        # Check capabilities
        self.has_cuda = torch.cuda.is_available()
        self.has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Memory tracking
        self.memory_history = []
        self.peak_memory = 0.0
        
        logger.info(f"GPUMonitor initialized for device: {self.device}")
        logger.info(f"CUDA available: {self.has_cuda}, MPS available: {self.has_mps}")
    
    def _get_best_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage in GB.
        
        Returns:
            Dictionary with memory statistics
        """
        try:
            if self.device_type == "cuda" and self.has_cuda:
                return self._get_cuda_memory_usage()
            elif self.device_type == "mps" and self.has_mps:
                return self._get_mps_memory_usage()
            else:
                return self._get_cpu_memory_usage()
                
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            return {
                "allocated_gb": 0.0,
                "cached_gb": 0.0,
                "total_gb": 0.0,
                "utilization_percent": 0.0,
            }
    
    def _get_cuda_memory_usage(self) -> Dict[str, float]:
        """Get CUDA memory usage."""
        allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
        cached = torch.cuda.memory_reserved(self.device) / (1024**3)
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
        
        return {
            "allocated_gb": allocated,
            "cached_gb": cached,
            "total_gb": total_memory,
            "utilization_percent": utilization,
        }
    
    def _get_mps_memory_usage(self) -> Dict[str, float]:
        """Get MPS (Apple Silicon) memory usage."""
        try:
            # Current allocated memory
            allocated = torch.mps.current_allocated_memory() / (1024**3)
            
            # Driver allocated memory (includes cached)
            driver_allocated = torch.mps.driver_allocated_memory() / (1024**3)
            
            # Get system memory info for total
            system_memory = psutil.virtual_memory()
            total_memory = system_memory.total / (1024**3)
            
            # MPS shares system memory, so utilization is approximate
            utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
            
            return {
                "allocated_gb": allocated,
                "cached_gb": driver_allocated - allocated,
                "total_gb": total_memory,
                "utilization_percent": utilization,
            }
            
        except Exception as e:
            logger.warning(f"MPS memory query failed: {e}")
            return self._get_cpu_memory_usage()
    
    def _get_cpu_memory_usage(self) -> Dict[str, float]:
        """Get CPU memory usage."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            allocated = memory_info.rss / (1024**3)  # Resident set size
            total_memory = system_memory.total / (1024**3)
            utilization = (allocated / total_memory) * 100 if total_memory > 0 else 0
            
            return {
                "allocated_gb": allocated,
                "cached_gb": 0.0,  # Not applicable for CPU
                "total_gb": total_memory,
                "utilization_percent": utilization,
            }
            
        except Exception as e:
            logger.warning(f"CPU memory query failed: {e}")
            return {
                "allocated_gb": 0.0,
                "cached_gb": 0.0,
                "total_gb": 0.0,
                "utilization_percent": 0.0,
            }
    
    def update(self) -> Dict[str, float]:
        """Update memory tracking and return current usage."""
        memory_usage = self.get_memory_usage()
        
        # Track peak memory
        current_allocated = memory_usage["allocated_gb"]
        if current_allocated > self.peak_memory:
            self.peak_memory = current_allocated
        
        # Add to history with timestamp
        memory_entry = {
            "timestamp": time.time(),
            "allocated_gb": current_allocated,
            "cached_gb": memory_usage["cached_gb"],
            "utilization_percent": memory_usage["utilization_percent"],
        }
        self.memory_history.append(memory_entry)
        
        # Keep only recent history (last 1000 entries)
        if len(self.memory_history) > 1000:
            self.memory_history = self.memory_history[-1000:]
        
        return memory_usage
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB."""
        return self.peak_memory
    
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        try:
            if self.device_type == "cuda" and self.has_cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            elif self.device_type == "mps" and self.has_mps:
                # MPS doesn't have reset functionality, just reset our tracking
                pass
        except Exception as e:
            logger.warning(f"Failed to reset peak memory stats: {e}")
        
        self.peak_memory = 0.0
    
    def clear_cache(self) -> None:
        """Clear memory cache."""
        try:
            if self.device_type == "cuda" and self.has_cuda:
                torch.cuda.empty_cache()
            elif self.device_type == "mps" and self.has_mps:
                torch.mps.empty_cache()
            # CPU doesn't have explicit cache clearing
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.memory_history:
            return {}
        
        recent_entries = self.memory_history[-100:]  # Last 100 entries
        allocated_values = [entry["allocated_gb"] for entry in recent_entries]
        utilization_values = [entry["utilization_percent"] for entry in recent_entries]
        
        return {
            "current_allocated_gb": allocated_values[-1] if allocated_values else 0.0,
            "peak_allocated_gb": self.peak_memory,
            "avg_allocated_gb": np.mean(allocated_values),
            "max_utilization_percent": np.max(utilization_values),
            "avg_utilization_percent": np.mean(utilization_values),
            "memory_trend": "increasing" if len(allocated_values) > 1 and allocated_values[-1] > allocated_values[0] else "stable",
        }


class TrainingSpeedTracker:
    """
    Training speed and throughput monitoring.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize speed tracker.
        
        Args:
            window_size: Number of recent steps to consider for moving averages
        """
        self.window_size = window_size
        
        # Timing tracking
        self.step_times = deque(maxlen=window_size)
        self.batch_sizes = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        
        # Timestamps
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # Counters
        self.total_steps = 0
        self.total_tokens = 0
        self.total_examples = 0
    
    def update(
        self,
        batch_size: int,
        num_tokens: Optional[int] = None,
        step_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update speed tracking.
        
        Args:
            batch_size: Number of examples in the batch
            num_tokens: Number of tokens processed (if available)
            step_time: Time taken for this step (if None, calculated from last update)
            
        Returns:
            Dictionary with speed metrics
        """
        current_time = time.time()
        
        # Calculate step time
        if step_time is None:
            step_time = current_time - self.last_step_time
        
        # Update tracking
        self.step_times.append(step_time)
        self.batch_sizes.append(batch_size)
        if num_tokens is not None:
            self.token_counts.append(num_tokens)
        
        # Update totals
        self.total_steps += 1
        self.total_examples += batch_size
        if num_tokens is not None:
            self.total_tokens += num_tokens
        
        self.last_step_time = current_time
        
        return self.get_speed_metrics()
    
    def get_speed_metrics(self) -> Dict[str, float]:
        """Get current speed metrics."""
        if not self.step_times:
            return {}
        
        # Calculate averages
        avg_step_time = np.mean(self.step_times)
        avg_batch_size = np.mean(self.batch_sizes)
        
        # Basic speed metrics
        steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        examples_per_second = (avg_batch_size / avg_step_time) if avg_step_time > 0 else 0.0
        
        metrics = {
            "steps_per_second": steps_per_second,
            "examples_per_second": examples_per_second,
            "avg_step_time_seconds": avg_step_time,
            "avg_batch_size": avg_batch_size,
        }
        
        # Token-based metrics if available
        if self.token_counts:
            avg_tokens = np.mean(self.token_counts)
            tokens_per_second = (avg_tokens / avg_step_time) if avg_step_time > 0 else 0.0
            
            metrics.update({
                "tokens_per_second": tokens_per_second,
                "avg_tokens_per_batch": avg_tokens,
                "tokens_per_example": avg_tokens / avg_batch_size if avg_batch_size > 0 else 0.0,
            })
        
        # Overall training metrics
        total_time = time.time() - self.start_time
        if total_time > 0:
            metrics.update({
                "total_steps": self.total_steps,
                "total_examples": self.total_examples,
                "total_training_time_hours": total_time / 3600,
                "overall_examples_per_second": self.total_examples / total_time,
            })
            
            if self.total_tokens > 0:
                metrics["overall_tokens_per_second"] = self.total_tokens / total_time
        
        return metrics


class GradientMonitor:
    """
    Gradient norm tracking and analysis.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize gradient monitor.
        
        Args:
            window_size: Number of recent steps to track
        """
        self.window_size = window_size
        self.grad_norms = deque(maxlen=window_size)
        self.param_norms = deque(maxlen=window_size)
        self.grad_norm_ratios = deque(maxlen=window_size)
        
        # Gradient clipping tracking
        self.clip_events = 0
        self.total_updates = 0
        
    def update(
        self,
        model: torch.nn.Module,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update gradient monitoring.
        
        Args:
            model: Model to analyze gradients for
            max_grad_norm: Maximum gradient norm (for clipping detection)
            
        Returns:
            Dictionary with gradient metrics
        """
        # Calculate gradient norm
        grad_norm = self._calculate_gradient_norm(model)
        param_norm = self._calculate_parameter_norm(model)
        
        # Track gradient norm
        self.grad_norms.append(grad_norm)
        self.param_norms.append(param_norm)
        
        # Calculate gradient-to-parameter ratio
        if param_norm > 0:
            grad_norm_ratio = grad_norm / param_norm
            self.grad_norm_ratios.append(grad_norm_ratio)
        
        # Track gradient clipping
        self.total_updates += 1
        if max_grad_norm is not None and grad_norm > max_grad_norm:
            self.clip_events += 1
        
        return self.get_gradient_metrics()
    
    def _calculate_gradient_norm(self, model: torch.nn.Module) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def _calculate_parameter_norm(self, model: torch.nn.Module) -> float:
        """Calculate total parameter norm."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.requires_grad:
                param_norm = param.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def get_gradient_metrics(self) -> Dict[str, float]:
        """Get gradient analysis metrics."""
        if not self.grad_norms:
            return {}
        
        metrics = {
            "grad_norm": self.grad_norms[-1],
            "param_norm": self.param_norms[-1] if self.param_norms else 0.0,
            "avg_grad_norm": np.mean(self.grad_norms),
            "max_grad_norm": np.max(self.grad_norms),
            "min_grad_norm": np.min(self.grad_norms),
            "grad_norm_std": np.std(self.grad_norms),
        }
        
        if self.grad_norm_ratios:
            metrics.update({
                "grad_norm_ratio": self.grad_norm_ratios[-1],
                "avg_grad_norm_ratio": np.mean(self.grad_norm_ratios),
            })
        
        # Gradient clipping statistics
        if self.total_updates > 0:
            metrics["grad_clip_rate"] = self.clip_events / self.total_updates
        
        return metrics
    
    def detect_gradient_issues(self) -> List[str]:
        """Detect potential gradient issues."""
        issues = []
        
        if not self.grad_norms:
            return issues
        
        recent_grad_norms = list(self.grad_norms)[-10:]  # Last 10 steps
        
        # Check for exploding gradients
        if np.max(recent_grad_norms) > 100:
            issues.append("exploding_gradients")
        
        # Check for vanishing gradients
        if np.mean(recent_grad_norms) < 1e-6:
            issues.append("vanishing_gradients")
        
        # Check for unstable gradients (high variance)
        if len(recent_grad_norms) > 1 and np.std(recent_grad_norms) / np.mean(recent_grad_norms) > 10:
            issues.append("unstable_gradients")
        
        # Check for NaN gradients
        if np.any(np.isnan(recent_grad_norms)):
            issues.append("nan_gradients")
        
        return issues


class EarlyStoppingMonitor:
    """
    Early stopping logic based on training metrics.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        metric_name: str = "loss",
        mode: str = "min",
        restore_best_weights: bool = True,
    ):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Number of steps with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            metric_name: Name of metric to monitor
            mode: "min" for metrics that should decrease, "max" for metrics that should increase
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        # Tracking variables
        self.best_metric = None
        self.best_step = 0
        self.wait_count = 0
        self.stopped = False
        self.best_weights = None
        
        # Comparison function
        if mode == "min":
            self.is_better = lambda current, best: current < best - min_delta
        elif mode == "max":
            self.is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def update(
        self,
        metrics: Dict[str, float],
        step: int,
        model_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Update early stopping monitor.
        
        Args:
            metrics: Dictionary of current metrics
            step: Current training step
            model_state: Current model state (for saving best weights)
            
        Returns:
            Dictionary with early stopping information
        """
        if self.metric_name not in metrics:
            logger.warning(f"Metric '{self.metric_name}' not found in metrics")
            return {"should_stop": False, "improved": False}
        
        current_metric = metrics[self.metric_name]
        improved = False
        
        # Initialize or check for improvement
        if self.best_metric is None or self.is_better(current_metric, self.best_metric):
            self.best_metric = current_metric
            self.best_step = step
            self.wait_count = 0
            improved = True
            
            # Save best weights if requested
            if self.restore_best_weights and model_state is not None:
                self.best_weights = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                                   for k, v in model_state.items()}
        else:
            self.wait_count += 1
        
        # Check if should stop
        should_stop = self.wait_count >= self.patience
        if should_stop:
            self.stopped = True
        
        return {
            "should_stop": should_stop,
            "improved": improved,
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "wait_count": self.wait_count,
            "patience": self.patience,
        }
    
    def get_best_weights(self) -> Optional[Dict[str, Any]]:
        """Get the best weights saved during training."""
        return self.best_weights
    
    def reset(self) -> None:
        """Reset early stopping monitor."""
        self.best_metric = None
        self.best_step = 0
        self.wait_count = 0
        self.stopped = False
        self.best_weights = None


class TrainingMonitor:
    """
    Comprehensive training monitor that combines all monitoring capabilities.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        speed_window_size: int = 100,
        gradient_window_size: int = 100,
        early_stopping_config: Optional[Dict[str, Any]] = None,
        log_every_n_steps: int = 10,
    ):
        """
        Initialize comprehensive training monitor.
        
        Args:
            device: Device to monitor
            speed_window_size: Window size for speed tracking
            gradient_window_size: Window size for gradient tracking
            early_stopping_config: Configuration for early stopping
            log_every_n_steps: Log monitoring info every N steps
        """
        self.device = device
        self.log_every_n_steps = log_every_n_steps
        
        # Initialize individual monitors
        self.gpu_monitor = GPUMonitor(device)
        self.speed_tracker = TrainingSpeedTracker(speed_window_size)
        self.gradient_monitor = GradientMonitor(gradient_window_size)
        
        # Initialize early stopping if configured
        if early_stopping_config:
            self.early_stopping = EarlyStoppingMonitor(**early_stopping_config)
        else:
            self.early_stopping = None
        
        # Overall monitoring state
        self.monitoring_history = []
        self.start_time = time.time()
        
        logger.info("TrainingMonitor initialized with all monitoring capabilities")
    
    def update(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        model: torch.nn.Module,
        batch_size: int,
        num_tokens: Optional[int] = None,
        max_grad_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update all monitoring components.
        
        Args:
            step: Current training step
            epoch: Current epoch
            metrics: Training metrics
            model: Model being trained
            batch_size: Current batch size
            num_tokens: Number of tokens processed
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Comprehensive monitoring results
        """
        monitoring_results = {
            "step": step,
            "epoch": epoch,
            "timestamp": time.time(),
        }
        
        # Update GPU monitoring
        try:
            gpu_metrics = self.gpu_monitor.update()
            monitoring_results["gpu"] = gpu_metrics
        except Exception as e:
            logger.warning(f"GPU monitoring failed: {e}")
            monitoring_results["gpu"] = {}
        
        # Update speed tracking
        try:
            speed_metrics = self.speed_tracker.update(batch_size, num_tokens)
            monitoring_results["speed"] = speed_metrics
        except Exception as e:
            logger.warning(f"Speed tracking failed: {e}")
            monitoring_results["speed"] = {}
        
        # Update gradient monitoring
        try:
            gradient_metrics = self.gradient_monitor.update(model, max_grad_norm)
            monitoring_results["gradients"] = gradient_metrics
            
            # Check for gradient issues
            gradient_issues = self.gradient_monitor.detect_gradient_issues()
            if gradient_issues:
                monitoring_results["gradient_issues"] = gradient_issues
                logger.warning(f"Gradient issues detected: {gradient_issues}")
                
        except Exception as e:
            logger.warning(f"Gradient monitoring failed: {e}")
            monitoring_results["gradients"] = {}
        
        # Update early stopping
        if self.early_stopping:
            try:
                early_stopping_results = self.early_stopping.update(
                    metrics, step, model.state_dict()
                )
                monitoring_results["early_stopping"] = early_stopping_results
                
                if early_stopping_results["should_stop"]:
                    logger.info(f"Early stopping triggered at step {step}")
                    
            except Exception as e:
                logger.warning(f"Early stopping monitoring failed: {e}")
        
        # Add training metrics
        monitoring_results["metrics"] = metrics.copy()
        
        # Store in history
        self.monitoring_history.append(monitoring_results)
        
        # Keep only recent history
        if len(self.monitoring_history) > 1000:
            self.monitoring_history = self.monitoring_history[-1000:]
        
        # Log periodically
        if step % self.log_every_n_steps == 0:
            self._log_monitoring_summary(monitoring_results)
        
        return monitoring_results
    
    def _log_monitoring_summary(self, results: Dict[str, Any]) -> None:
        """Log monitoring summary."""
        step = results["step"]
        
        # GPU info
        gpu_info = results.get("gpu", {})
        gpu_mem = gpu_info.get("allocated_gb", 0)
        gpu_util = gpu_info.get("utilization_percent", 0)
        
        # Speed info
        speed_info = results.get("speed", {})
        steps_per_sec = speed_info.get("steps_per_second", 0)
        examples_per_sec = speed_info.get("examples_per_second", 0)
        
        # Gradient info
        grad_info = results.get("gradients", {})
        grad_norm = grad_info.get("grad_norm", 0)
        
        logger.info(
            f"Monitor [Step {step:6d}] | "
            f"GPU: {gpu_mem:.2f}GB ({gpu_util:.1f}%) | "
            f"Speed: {steps_per_sec:.2f} step/s, {examples_per_sec:.1f} ex/s | "
            f"Grad: {grad_norm:.6f}"
        )
        
        # Log warnings for issues
        if "gradient_issues" in results:
            logger.warning(f"Gradient issues: {results['gradient_issues']}")
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped."""
        if self.early_stopping:
            return self.early_stopping.stopped
        return False
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary."""
        if not self.monitoring_history:
            return {}
        
        total_time = time.time() - self.start_time
        
        summary = {
            "total_training_time_hours": total_time / 3600,
            "total_steps": len(self.monitoring_history),
            "gpu_summary": self.gpu_monitor.get_memory_summary(),
            "speed_summary": self.speed_tracker.get_speed_metrics(),
        }
        
        # Add early stopping summary
        if self.early_stopping:
            summary["early_stopping"] = {
                "stopped": self.early_stopping.stopped,
                "best_metric": self.early_stopping.best_metric,
                "best_step": self.early_stopping.best_step,
            }
        
        return summary
    
    def cleanup(self) -> None:
        """Cleanup monitoring resources."""
        try:
            self.gpu_monitor.clear_cache()
        except Exception as e:
            logger.warning(f"Failed to cleanup GPU monitor: {e}")
        
        logger.info("TrainingMonitor cleanup completed")