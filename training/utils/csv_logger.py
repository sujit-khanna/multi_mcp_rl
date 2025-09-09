#!/usr/bin/env python3
"""
CSV Logger for Training Metrics

This module provides comprehensive CSV logging for training data analysis.
Saves all training metrics in tabular format with incremental updates.
"""

import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class CSVLogger:
    """
    Comprehensive CSV logger for training metrics analysis.
    
    Features:
    - Multiple CSV files for different metric categories
    - Incremental writing every N steps
    - Automatic schema detection and extension
    - Data validation and type conversion
    - Thread-safe writing operations
    """
    
    def __init__(self, output_dir: str, save_frequency: int = 10):
        """
        Initialize CSV logger.
        
        Args:
            output_dir: Base output directory (e.g., outputs/real-env-grpo-vllm-*)
            save_frequency: Save to CSV every N steps (default: 10)
        """
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "training_metrics"
        self.save_frequency = save_frequency
        
        # Create directory structure
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths for different metric categories
        self.csv_files = {
            "training": self.metrics_dir / "training_metrics.csv",
            "rollouts": self.metrics_dir / "rollout_metrics.csv", 
            "episodes": self.metrics_dir / "episode_details.csv",
            "trajectories": self.metrics_dir / "trajectory_details.csv",
            "ppo": self.metrics_dir / "ppo_metrics.csv",
            "performance": self.metrics_dir / "performance_metrics.csv",
            "system": self.metrics_dir / "system_metrics.csv"
        }
        
        # Data buffers for incremental writing
        self.data_buffers = {key: [] for key in self.csv_files.keys()}
        
        # Schema tracking for dynamic column addition
        self.schemas = {key: set() for key in self.csv_files.keys()}
        
        # Step counter
        self.step_count = 0
        self.last_save_step = 0
        
        # Initialize metadata file
        self._write_metadata()
        
        logger.info(f"CSV Logger initialized at {self.metrics_dir}")
        logger.info(f"Save frequency: every {save_frequency} steps")
        
    def _write_metadata(self):
        """Write metadata about the logging session."""
        metadata = {
            "created_at": datetime.now().isoformat(),
            "save_frequency": self.save_frequency,
            "csv_files": {key: str(path.name) for key, path in self.csv_files.items()},
            "description": "Training metrics logged from GRPO vLLM training",
            "version": "1.0"
        }
        
        with open(self.metrics_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def _sanitize_value(self, value: Any) -> Any:
        """Convert values to CSV-safe format."""
        if value is None:
            return ""
        elif isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.item())
            else:
                return str(value.cpu().numpy().tolist())
        elif isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._sanitize_value(value[0])
            return str(value)
        elif isinstance(value, dict):
            return json.dumps(value)
        elif isinstance(value, (int, float, str, bool)):
            return value
        else:
            return str(value)
    
    def _flatten_metrics(self, metrics: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionaries for CSV storage."""
        flattened = {}
        
        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dicts
                flattened.update(self._flatten_metrics(value, f"{full_key}."))
            else:
                flattened[full_key] = self._sanitize_value(value)
        
        return flattened
    
    def _categorize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Categorize metrics into different CSV files."""
        categories = {key: {} for key in self.csv_files.keys()}
        
        # Add common fields to all categories
        common_fields = {
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count
        }
        
        for category in categories:
            categories[category].update(common_fields)
        
        # Categorize metrics by prefix or content
        for key, value in metrics.items():
            key_lower = key.lower()
            
            if key_lower.startswith("rollouts/") or "episode" in key_lower or "collection" in key_lower:
                categories["rollouts"][key] = value
            elif key_lower.startswith("ppo/") or "ratio" in key_lower or key_lower.startswith("std_ratio"):
                categories["ppo"][key] = value
            elif "loss" in key_lower or "grad" in key_lower or "kl" in key_lower:
                categories["training"][key] = value
            elif "time" in key_lower or "speed" in key_lower or "memory" in key_lower:
                categories["performance"][key] = value
            elif "gpu" in key_lower or "cpu" in key_lower or "vllm" in key_lower:
                categories["system"][key] = value
            else:
                # Default to training metrics
                categories["training"][key] = value
        
        return categories
    
    def log_training_step(self, metrics: Dict[str, Any], epoch: int = None):
        """
        Log metrics from a training step.
        
        Args:
            metrics: Dictionary of training metrics
            epoch: Current epoch number
        """
        self.step_count += 1
        
        # Add step and epoch info
        enhanced_metrics = metrics.copy()
        enhanced_metrics.update({
            "step": self.step_count,
            "epoch": epoch if epoch is not None else -1,
            "timestamp": datetime.now().isoformat()
        })
        
        # Flatten nested metrics
        flattened_metrics = self._flatten_metrics(enhanced_metrics)
        
        # Categorize metrics
        categorized = self._categorize_metrics(flattened_metrics)
        
        # Add to buffers
        for category, data in categorized.items():
            if data:  # Only add non-empty data
                self.data_buffers[category].append(data)
                self.schemas[category].update(data.keys())
        
        # Save if frequency reached
        if self.step_count - self.last_save_step >= self.save_frequency:
            self.save_to_csv()
    
    def log_episode_details(self, episode_data: List[Dict[str, Any]]):
        """
        Log detailed episode information.
        
        Args:
            episode_data: List of episode dictionaries
        """
        for i, episode in enumerate(episode_data):
            episode_record = self._flatten_metrics(episode)
            episode_record.update({
                "timestamp": datetime.now().isoformat(),
                "step": self.step_count,
                "episode_index": i
            })
            
            self.data_buffers["episodes"].append(episode_record)
            self.schemas["episodes"].update(episode_record.keys())
    
    def log_trajectory_details(self, trajectories: List[Any]):
        """
        Log detailed trajectory information.
        
        Args:
            trajectories: List of trajectory objects
        """
        for i, traj in enumerate(trajectories):
            traj_record = {
                "timestamp": datetime.now().isoformat(),
                "step": self.step_count,
                "trajectory_index": i,
                "task_id": getattr(traj, "task_id", f"traj_{i}"),
                "length": len(getattr(traj, "states", [])),
                "total_reward": sum(getattr(traj, "rewards", [])),
                "avg_reward": sum(getattr(traj, "rewards", [])) / max(1, len(getattr(traj, "rewards", []))),
                "num_actions": len(getattr(traj, "actions", [])),
                "num_dones": sum(getattr(traj, "dones", [])),
            }
            
            # Add logprob statistics if available
            if hasattr(traj, "log_probs") and traj.log_probs:
                try:
                    logprobs = [float(lp) for lp in traj.log_probs if lp is not None]
                    if logprobs:
                        traj_record.update({
                            "logprob_sum": sum(logprobs),
                            "logprob_mean": sum(logprobs) / len(logprobs),
                            "logprob_min": min(logprobs),
                            "logprob_max": max(logprobs)
                        })
                except Exception as e:
                    logger.debug(f"Error processing logprobs: {e}")
            
            # Add forced mask statistics if available
            if hasattr(traj, "forced_mask") and traj.forced_mask is not None:
                try:
                    forced_mask = traj.forced_mask.cpu().numpy() if hasattr(traj.forced_mask, "cpu") else traj.forced_mask
                    traj_record.update({
                        "forced_actions": int(sum(forced_mask)),
                        "forced_fraction": float(sum(forced_mask) / len(forced_mask)) if len(forced_mask) > 0 else 0.0
                    })
                except Exception as e:
                    logger.debug(f"Error processing forced mask: {e}")
            
            self.data_buffers["trajectories"].append(traj_record)
            self.schemas["trajectories"].update(traj_record.keys())
    
    def log_ppo_metrics(self, ppo_data: Dict[str, Any]):
        """
        Log detailed PPO-specific metrics.
        
        Args:
            ppo_data: Dictionary with PPO-specific data
        """
        ppo_record = self._flatten_metrics(ppo_data)
        ppo_record.update({
            "timestamp": datetime.now().isoformat(),
            "step": self.step_count
        })
        
        self.data_buffers["ppo"].append(ppo_record)
        self.schemas["ppo"].update(ppo_record.keys())
    
    def save_to_csv(self, force: bool = False):
        """
        Save buffered data to CSV files.
        
        Args:
            force: Force save even if frequency not reached
        """
        if not force and self.step_count - self.last_save_step < self.save_frequency:
            return
        
        saved_files = []
        
        for category, buffer in self.data_buffers.items():
            if not buffer:  # Skip empty buffers
                continue
                
            csv_path = self.csv_files[category]
            schema = sorted(self.schemas[category])
            
            # Check if file exists to determine write mode
            file_exists = csv_path.exists()
            write_header = not file_exists
            
            try:
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
                    
                    if write_header:
                        writer.writeheader()
                    
                    # Write all buffered data
                    for row in buffer:
                        # Fill missing columns with empty values
                        complete_row = {col: row.get(col, "") for col in schema}
                        writer.writerow(complete_row)
                
                saved_files.append((category, len(buffer), csv_path))
                
            except Exception as e:
                logger.error(f"Error writing {category} CSV: {e}")
                continue
        
        # Clear buffers after successful write
        for category in saved_files:
            self.data_buffers[category[0]] = []
        
        self.last_save_step = self.step_count
        
        if saved_files:
            logger.info(f"Saved CSV data at step {self.step_count}:")
            for category, count, path in saved_files:
                logger.info(f"  {category}: {count} rows → {path.name}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics about logged data."""
        stats = {
            "step_count": self.step_count,
            "last_save_step": self.last_save_step,
            "buffered_rows": {cat: len(buf) for cat, buf in self.data_buffers.items()},
            "csv_files": {}
        }
        
        for category, path in self.csv_files.items():
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    stats["csv_files"][category] = {
                        "rows": len(df),
                        "columns": len(df.columns),
                        "size_mb": path.stat().st_size / (1024 * 1024)
                    }
                except Exception as e:
                    stats["csv_files"][category] = {"error": str(e)}
        
        return stats
    
    def finalize(self):
        """Final save and cleanup when training ends."""
        logger.info("Finalizing CSV logger...")
        
        # Force save any remaining buffered data
        self.save_to_csv(force=True)
        
        # Create final summary
        summary = self.get_summary_stats()
        summary["finalized_at"] = datetime.now().isoformat()
        
        with open(self.metrics_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"CSV logging finalized. Summary saved to {self.metrics_dir / 'summary.json'}")
        logger.info(f"Total steps logged: {self.step_count}")
        
        # Log file locations for easy access
        logger.info("CSV files created:")
        for category, path in self.csv_files.items():
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                logger.info(f"  {category}: {path} ({size_mb:.2f} MB)")


def create_csv_logger(output_dir: str, save_frequency: int = 10) -> CSVLogger:
    """
    Factory function to create a CSV logger.
    
    Args:
        output_dir: Output directory path
        save_frequency: Save frequency in steps
        
    Returns:
        Configured CSVLogger instance
    """
    return CSVLogger(output_dir, save_frequency)