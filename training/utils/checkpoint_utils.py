"""
Checkpoint Utility Functions for GRPO Training

This module provides comprehensive checkpoint management for both LoRA and full model training,
with support for distributed training, sharded checkpoints, and metadata tracking.
"""

import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
import warnings

import torch
import torch.nn as nn
from torch.optim import Optimizer

# Set up logging
logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Comprehensive checkpoint manager for GRPO training.
    
    Handles both LoRA adapters and full model weights with support for:
    - Automatic checkpoint rotation
    - Metadata tracking
    - Distributed training
    - Sharded checkpoints
    - Resume functionality
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
        save_every_n_steps: int = 1000,
        save_every_n_epochs: int = 1,
        use_sharding: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_every_n_steps: Save checkpoint every N steps
            save_every_n_epochs: Save checkpoint every N epochs
            use_sharding: Whether to use sharded checkpoints for large models
            rank: Process rank for distributed training
            world_size: Total number of processes
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_every_n_steps = save_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.use_sharding = use_sharding
        self.rank = rank
        self.world_size = world_size
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        logger.info(f"  Max checkpoints: {self.max_checkpoints}")
        logger.info(f"  Save every {self.save_every_n_steps} steps, {self.save_every_n_epochs} epochs")
        logger.info(f"  Sharding: {self.use_sharding}, Rank: {self.rank}/{self.world_size}")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
        
        return {
            "checkpoints": [],
            "latest_checkpoint": None,
            "best_checkpoint": None,
            "best_metric": None,
        }
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def should_save_checkpoint(self, step: int, epoch: int) -> bool:
        """Determine if a checkpoint should be saved."""
        return (
            step % self.save_every_n_steps == 0 or
            epoch % self.save_every_n_epochs == 0
        )
    
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        is_best: bool = False,
        save_optimizer: bool = True,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a training checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            epoch: Current epoch
            step: Current step
            metrics: Training metrics
            is_best: Whether this is the best checkpoint so far
            save_optimizer: Whether to save optimizer state
            additional_data: Additional data to save with checkpoint
            
        Returns:
            str: Path to saved checkpoint
        """
        # Only rank 0 saves in distributed training
        if self.world_size > 1 and self.rank != 0:
            return ""
        
        checkpoint_name = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving checkpoint: {checkpoint_path}")
        
        try:
            # Determine if this is a LoRA model
            is_lora_model = self._is_lora_model(model)
            
            # Save model
            if is_lora_model:
                self._save_lora_model(model, checkpoint_path)
            else:
                self._save_full_model(model, checkpoint_path)
            
            # Save optimizer
            if save_optimizer and optimizer is not None:
                self._save_optimizer(optimizer, checkpoint_path)
            
            # Save training state
            training_state = {
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "model_type": "lora" if is_lora_model else "full",
                "timestamp": time.time(),
                "rank": self.rank,
                "world_size": self.world_size,
            }
            
            if additional_data:
                training_state.update(additional_data)
            
            with open(checkpoint_path / "training_state.json", 'w') as f:
                json.dump(training_state, f, indent=2)
            
            # Update metadata
            self._update_metadata(checkpoint_name, metrics, is_best)
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"✅ Checkpoint saved successfully: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            # Clean up partial checkpoint
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path, ignore_errors=True)
            raise
    
    def _is_lora_model(self, model: nn.Module) -> bool:
        """Check if model is using LoRA adapters."""
        return hasattr(model, 'peft_config') or any(
            'lora' in name.lower() for name, _ in model.named_parameters()
        )
    
    def _save_lora_model(self, model: nn.Module, checkpoint_path: Path) -> None:
        """Save LoRA model adapters."""
        try:
            if hasattr(model, 'save_pretrained'):
                # PEFT model with save_pretrained method
                model.save_pretrained(checkpoint_path / "lora_adapters")
            else:
                # Manual LoRA parameter saving
                lora_params = {}
                for name, param in model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        lora_params[name] = param.cpu().clone()
                
                torch.save(lora_params, checkpoint_path / "lora_adapters.pth")
            
            logger.info("LoRA adapters saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save LoRA adapters: {e}")
            raise
    
    def _save_full_model(self, model: nn.Module, checkpoint_path: Path) -> None:
        """Save full model weights."""
        try:
            if self.use_sharding and self.world_size > 1:
                self._save_sharded_model(model, checkpoint_path)
            else:
                # Standard single-file save
                model_state = model.state_dict()
                torch.save(model_state, checkpoint_path / "model.pth")
            
            logger.info("Full model weights saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save full model: {e}")
            raise
    
    def _save_sharded_model(self, model: nn.Module, checkpoint_path: Path) -> None:
        """Save model with sharding for large models."""
        try:
            from torch.distributed import get_rank, get_world_size
            
            model_state = model.state_dict()
            
            # Split state dict across shards
            shard_size = len(model_state) // self.world_size
            start_idx = self.rank * shard_size
            end_idx = start_idx + shard_size if self.rank < self.world_size - 1 else len(model_state)
            
            items = list(model_state.items())
            shard_state = dict(items[start_idx:end_idx])
            
            # Save shard
            shard_path = checkpoint_path / f"model_shard_{self.rank:02d}.pth"
            torch.save(shard_state, shard_path)
            
            # Save shard metadata (rank 0 only)
            if self.rank == 0:
                shard_metadata = {
                    "total_shards": self.world_size,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "shard_info": [
                        {"rank": i, "file": f"model_shard_{i:02d}.pth"}
                        for i in range(self.world_size)
                    ]
                }
                with open(checkpoint_path / "sharding_metadata.json", 'w') as f:
                    json.dump(shard_metadata, f, indent=2)
            
            logger.info(f"Model shard {self.rank} saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save sharded model: {e}")
            raise
    
    def _save_optimizer(self, optimizer: Optimizer, checkpoint_path: Path) -> None:
        """Save optimizer state."""
        try:
            optimizer_state = optimizer.state_dict()
            torch.save(optimizer_state, checkpoint_path / "optimizer.pth")
            logger.info("Optimizer state saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save optimizer: {e}")
            raise
    
    def _update_metadata(
        self,
        checkpoint_name: str,
        metrics: Dict[str, float],
        is_best: bool
    ) -> None:
        """Update checkpoint metadata."""
        checkpoint_info = {
            "name": checkpoint_name,
            "path": str(self.checkpoint_dir / checkpoint_name),
            "timestamp": time.time(),
            "metrics": metrics,
        }
        
        # Add to checkpoint list
        self.metadata["checkpoints"].append(checkpoint_info)
        self.metadata["latest_checkpoint"] = checkpoint_name
        
        # Update best checkpoint
        if is_best:
            self.metadata["best_checkpoint"] = checkpoint_name
            self.metadata["best_metric"] = metrics
        
        # Save metadata
        self._save_metadata()
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to maintain max_checkpoints limit."""
        if len(self.metadata["checkpoints"]) <= self.max_checkpoints:
            return
        
        # Sort by timestamp
        checkpoints = sorted(
            self.metadata["checkpoints"],
            key=lambda x: x["timestamp"]
        )
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info["path"])
            if checkpoint_path.exists():
                try:
                    shutil.rmtree(checkpoint_path)
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint_path}: {e}")
        
        # Update metadata
        self.metadata["checkpoints"] = checkpoints[-self.max_checkpoints:]
        self._save_metadata()
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint (if None, loads latest)
            model: Model to load weights into
            optimizer: Optimizer to load state into
            load_optimizer: Whether to load optimizer state
            strict: Whether to strictly enforce parameter matching
            
        Returns:
            Dict containing loaded training state
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoints found")
        
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            # Load training state
            training_state_path = checkpoint_path / "training_state.json"
            if training_state_path.exists():
                with open(training_state_path, 'r') as f:
                    training_state = json.load(f)
            else:
                training_state = {}
            
            # Load model
            if model is not None:
                self._load_model_weights(model, checkpoint_path, training_state, strict)
            
            # Load optimizer
            if optimizer is not None and load_optimizer:
                self._load_optimizer_state(optimizer, checkpoint_path)
            
            logger.info(f"✅ Checkpoint loaded successfully: {checkpoint_path}")
            return training_state
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def _load_model_weights(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        training_state: Dict[str, Any],
        strict: bool
    ) -> None:
        """Load model weights from checkpoint."""
        model_type = training_state.get("model_type", "full")
        
        if model_type == "lora":
            self._load_lora_weights(model, checkpoint_path, strict)
        else:
            self._load_full_model_weights(model, checkpoint_path, strict)
    
    def _load_lora_weights(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        strict: bool
    ) -> None:
        """Load LoRA adapter weights."""
        try:
            lora_adapters_path = checkpoint_path / "lora_adapters"
            lora_file_path = checkpoint_path / "lora_adapters.pth"
            
            if lora_adapters_path.exists() and hasattr(model, 'load_adapter'):
                # PEFT model with load_adapter method
                model.load_adapter(lora_adapters_path)
            elif lora_file_path.exists():
                # Manual LoRA parameter loading
                lora_params = torch.load(lora_file_path, map_location='cpu')
                model.load_state_dict(lora_params, strict=False)
            else:
                raise FileNotFoundError("No LoRA adapters found in checkpoint")
            
            logger.info("LoRA adapters loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LoRA adapters: {e}")
            raise
    
    def _load_full_model_weights(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        strict: bool
    ) -> None:
        """Load full model weights."""
        try:
            # Check for sharded model
            sharding_metadata_path = checkpoint_path / "sharding_metadata.json"
            
            if sharding_metadata_path.exists():
                self._load_sharded_model_weights(model, checkpoint_path, strict)
            else:
                # Standard single-file load
                model_path = checkpoint_path / "model.pth"
                if model_path.exists():
                    state_dict = torch.load(model_path, map_location='cpu')
                    model.load_state_dict(state_dict, strict=strict)
                else:
                    raise FileNotFoundError("No model weights found in checkpoint")
            
            logger.info("Full model weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load full model weights: {e}")
            raise
    
    def _load_sharded_model_weights(
        self,
        model: nn.Module,
        checkpoint_path: Path,
        strict: bool
    ) -> None:
        """Load sharded model weights."""
        try:
            # Load sharding metadata
            with open(checkpoint_path / "sharding_metadata.json", 'r') as f:
                shard_metadata = json.load(f)
            
            # Load all shards
            full_state_dict = {}
            for shard_info in shard_metadata["shard_info"]:
                shard_path = checkpoint_path / shard_info["file"]
                if shard_path.exists():
                    shard_state = torch.load(shard_path, map_location='cpu')
                    full_state_dict.update(shard_state)
            
            # Load into model
            model.load_state_dict(full_state_dict, strict=strict)
            logger.info("Sharded model weights loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load sharded model weights: {e}")
            raise
    
    def _load_optimizer_state(self, optimizer: Optimizer, checkpoint_path: Path) -> None:
        """Load optimizer state."""
        try:
            optimizer_path = checkpoint_path / "optimizer.pth"
            if optimizer_path.exists():
                optimizer_state = torch.load(optimizer_path, map_location='cpu')
                optimizer.load_state_dict(optimizer_state)
                logger.info("Optimizer state loaded successfully")
            else:
                logger.warning("No optimizer state found in checkpoint")
                
        except Exception as e:
            logger.error(f"Failed to load optimizer state: {e}")
            raise
    
    def find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint."""
        if self.metadata["latest_checkpoint"]:
            latest_path = self.checkpoint_dir / self.metadata["latest_checkpoint"]
            if latest_path.exists():
                return str(latest_path)
        
        # Fallback: search filesystem
        return find_latest_checkpoint(self.checkpoint_dir)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get the best checkpoint path."""
        if self.metadata["best_checkpoint"]:
            best_path = self.checkpoint_dir / self.metadata["best_checkpoint"]
            if best_path.exists():
                return str(best_path)
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return self.metadata["checkpoints"].copy()


# Standalone utility functions
def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    checkpoint_path: Union[str, Path],
    save_optimizer: bool = True,
    additional_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Simple checkpoint saving function.
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        step: Current step
        metrics: Training metrics
        checkpoint_path: Path to save checkpoint
        save_optimizer: Whether to save optimizer state
        additional_data: Additional data to save
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Create temporary checkpoint manager
    temp_manager = CheckpointManager(
        checkpoint_dir=checkpoint_path.parent,
        max_checkpoints=1000,  # Don't clean up
    )
    
    temp_manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        metrics=metrics,
        save_optimizer=save_optimizer,
        additional_data=additional_data,
    )


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Optional[nn.Module] = None,
    optimizer: Optional[Optimizer] = None,
    load_optimizer: bool = True,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Simple checkpoint loading function.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        load_optimizer: Whether to load optimizer state
        strict: Whether to strictly enforce parameter matching
        
    Returns:
        Dict containing training state
    """
    checkpoint_path = Path(checkpoint_path)
    
    # Create temporary checkpoint manager
    temp_manager = CheckpointManager(
        checkpoint_dir=checkpoint_path.parent,
    )
    
    return temp_manager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        load_optimizer=load_optimizer,
        strict=strict,
    )


def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for checkpoint directories with naming pattern
    checkpoint_pattern = re.compile(r'checkpoint_epoch_(\d+)_step_(\d+)')
    
    latest_checkpoint = None
    latest_step = -1
    
    for item in checkpoint_dir.iterdir():
        if item.is_dir():
            match = checkpoint_pattern.match(item.name)
            if match:
                step = int(match.group(2))
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = str(item)
    
    return latest_checkpoint