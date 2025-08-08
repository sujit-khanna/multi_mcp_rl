"""
Streaming Data Loader: Efficient data loading for GRPO training

This module implements memory-efficient data loading with curriculum learning,
caching, and proper batching for multi-turn tool use tasks.
"""

import json
import logging
import math
import os
import random
import hashlib
from collections import Counter, OrderedDict
from functools import lru_cache
from typing import Dict, List, Optional, Any, Iterator, Tuple, Set, Callable
import warnings

import numpy as np

# Optional dependencies
try:
    import pyarrow.parquet as pq
    import pyarrow as pa
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False
    warnings.warn("PyArrow not available. Parquet support disabled.")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU cache implementation for task caching.
    
    This provides memory-efficient caching of recently accessed tasks
    to avoid repeated disk reads while maintaining bounded memory usage.
    """
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to end if present."""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting oldest if at capacity."""
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Remove oldest
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)


class CurriculumSampler:
    """
    Curriculum learning sampler for progressive difficulty.
    
    This sampler starts with easier tasks and gradually increases difficulty
    based on training progress. It supports both sampling with and without
    replacement and tracks which tasks have been seen.
    
    Args:
        initial_distribution: Initial distribution of complexities (easy, medium, hard)
        final_distribution: Final target distribution
        transition_epochs: Number of epochs to transition from initial to final
        track_seen_tasks: Whether to track and avoid repeating tasks
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        initial_distribution: Dict[str, float] = {"easy": 0.3, "medium": 0.5, "hard": 0.2},
        final_distribution: Dict[str, float] = {"easy": 0.1, "medium": 0.4, "hard": 0.5},
        transition_epochs: int = 10,
        track_seen_tasks: bool = True,
        seed: int = 42,
    ):
        self.initial_distribution = initial_distribution
        self.final_distribution = final_distribution
        self.transition_epochs = transition_epochs
        self.track_seen_tasks = track_seen_tasks
        self.seed = seed
        
        # Validate distributions sum to 1.0
        for dist in [initial_distribution, final_distribution]:
            total = sum(dist.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"Distribution must sum to 1.0, got {total}")
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Track seen tasks
        self.seen_tasks: Set[str] = set()
        
        logger.info(f"CurriculumSampler initialized: {initial_distribution} -> {final_distribution}")
    
    def get_current_distribution(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Get current curriculum distribution based on training progress."""
        
        if total_epochs <= 0 or epoch >= total_epochs:
            return self.final_distribution.copy()
        
        # Calculate transition progress (0.0 to 1.0)
        transition_progress = min(epoch / self.transition_epochs, 1.0)
        
        # Interpolate between initial and final distributions
        current_dist = {}
        for complexity in self.initial_distribution:
            initial_prob = self.initial_distribution[complexity]
            final_prob = self.final_distribution.get(complexity, 0.0)
            current_prob = initial_prob + transition_progress * (final_prob - initial_prob)
            current_dist[complexity] = max(current_prob, 0.0)
        
        # Normalize to ensure sum = 1.0
        total = sum(current_dist.values())
        if total > 0:
            current_dist = {k: v / total for k, v in current_dist.items()}
        
        return current_dist
    
    def sample_complexity(self, epoch: int, total_epochs: int) -> str:
        """Sample a complexity level based on current curriculum."""
        
        current_dist = self.get_current_distribution(epoch, total_epochs)
        
        # Sample complexity based on current distribution
        complexities = list(current_dist.keys())
        probabilities = list(current_dist.values())
        
        return np.random.choice(complexities, p=probabilities)
    
    def mark_task_seen(self, task_id: str) -> None:
        """Mark a task as seen."""
        if self.track_seen_tasks:
            self.seen_tasks.add(task_id)
    
    def is_task_seen(self, task_id: str) -> bool:
        """Check if task has been seen."""
        return self.track_seen_tasks and task_id in self.seen_tasks
    
    def reset_seen_tasks(self) -> None:
        """Reset the set of seen tasks."""
        self.seen_tasks.clear()
        logger.info("Reset seen tasks for curriculum sampler")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum sampler statistics."""
        return {
            "seen_tasks": len(self.seen_tasks),
            "initial_distribution": self.initial_distribution,
            "final_distribution": self.final_distribution,
            "transition_epochs": self.transition_epochs,
        }


class TaskBatcher:
    """
    Smart task batcher that balances episodes by expected length.
    
    This batcher groups tasks to create batches with balanced total expected
    episode length, improving training efficiency by reducing variance in
    batch processing time.
    
    Args:
        expected_turns: Expected number of turns for each complexity level
        target_total_turns: Target total turns per batch
        max_batch_size: Maximum number of tasks per batch
    """
    
    def __init__(
        self,
        expected_turns: Dict[str, Tuple[int, int]] = {
            "easy": (3, 5),
            "medium": (5, 8), 
            "hard": (8, 12),
        },
        target_total_turns: int = 64,
        max_batch_size: int = 16,
    ):
        self.expected_turns = expected_turns
        self.target_total_turns = target_total_turns
        self.max_batch_size = max_batch_size
        
        # Calculate average expected turns for each complexity
        self.avg_turns = {}
        for complexity, (min_turns, max_turns) in expected_turns.items():
            self.avg_turns[complexity] = (min_turns + max_turns) / 2
        
        logger.info(f"TaskBatcher initialized: target_turns={target_total_turns}, "
                   f"max_batch={max_batch_size}")
    
    def create_batch(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create a batch of tasks balanced by expected episode length.
        
        Args:
            tasks: List of available tasks
            
        Returns:
            List of tasks forming a balanced batch
        """
        
        if not tasks:
            return []
        
        # Sort tasks by expected turns (shortest first for greedy packing)
        tasks_with_turns = []
        for task in tasks:
            complexity = task.get("task_metadata", {}).get("complexity", "medium")
            expected = self.avg_turns.get(complexity, self.avg_turns["medium"])
            tasks_with_turns.append((task, expected))
        
        # Sort by expected turns (ascending)
        tasks_with_turns.sort(key=lambda x: x[1])
        
        # Greedy bin packing to target total turns
        batch = []
        current_turns = 0.0
        
        for task, expected_turns in tasks_with_turns:
            if (len(batch) >= self.max_batch_size or 
                current_turns + expected_turns > self.target_total_turns * 1.2):
                break
            
            batch.append(task)
            current_turns += expected_turns
            
            # Stop if we've reached a good target
            if current_turns >= self.target_total_turns * 0.8:
                break
        
        # Ensure minimum batch size
        if len(batch) == 0 and tasks:
            batch = [tasks[0]]
        
        return batch
    
    def estimate_batch_turns(self, batch: List[Dict[str, Any]]) -> float:
        """Estimate total turns for a batch."""
        
        total_turns = 0.0
        for task in batch:
            complexity = task.get("task_metadata", {}).get("complexity", "medium")
            expected = self.avg_turns.get(complexity, self.avg_turns["medium"])
            total_turns += expected
        
        return total_turns


class StreamingDataset:
    """
    Memory-efficient streaming dataset for task data.
    
    This dataset reads tasks from JSON/JSONL or Parquet files without loading
    everything into memory. It supports curriculum learning, caching, filtering,
    and distributed training with sharding.
    
    Args:
        file_path: Path to data file (JSON, JSONL, or Parquet)
        cache_size: Size of LRU cache for recently accessed tasks
        shard_id: Shard ID for distributed training (0-based)
        num_shards: Total number of shards for distributed training
        filter_fn: Optional function to filter tasks
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        file_path: str,
        cache_size: int = 1000,
        shard_id: int = 0,
        num_shards: int = 1,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        seed: int = 42,
    ):
        self.file_path = file_path
        self.cache_size = cache_size
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.filter_fn = filter_fn
        self.seed = seed
        
        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Determine file format
        self.file_format = self._detect_file_format(file_path)
        
        # Initialize cache
        self.cache = LRUCache(capacity=cache_size)
        
        # Task index for streaming
        self._task_index = None
        self._statistics = None
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Build task index
        self._build_task_index()
        
        logger.info(f"StreamingDataset initialized: {file_path} ({self.file_format}), "
                   f"shard {shard_id}/{num_shards}, {len(self._task_index)} tasks")
    
    def _detect_file_format(self, file_path: str) -> str:
        """Detect file format from extension."""
        
        if file_path.endswith('.parquet'):
            if not HAS_PYARROW:
                raise RuntimeError("PyArrow required for Parquet support")
            return 'parquet'
        elif file_path.endswith('.jsonl'):
            return 'jsonl'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            # Default to JSON
            return 'json'
    
    def _build_task_index(self) -> None:
        """Build index of tasks for efficient access."""
        
        logger.info("Building task index...")
        
        self._task_index = []
        self._statistics = {
            "total_tasks": 0,
            "complexity_counts": Counter(),
            "category_counts": Counter(),
        }
        
        if self.file_format == 'parquet':
            self._build_parquet_index()
        else:
            self._build_json_index()
        
        # Apply sharding
        if self.num_shards > 1:
            self._apply_sharding()
        
        logger.info(f"Task index built: {len(self._task_index)} tasks after sharding")
        logger.info(f"Complexity distribution: {dict(self._statistics['complexity_counts'])}")
    
    def _build_json_index(self) -> None:
        """Build index for JSON/JSONL files."""
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if self.file_format == 'jsonl':
                # JSONL format - one task per line
                for line_no, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        task = json.loads(line)
                        self._process_task_for_index(task, line_no)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_no}: {e}")
                        continue
            else:
                # JSON format - single array or multiple objects
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        # Array of tasks
                        for i, task in enumerate(data):
                            self._process_task_for_index(task, i)
                    else:
                        # Single task
                        self._process_task_for_index(data, 0)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON file: {e}")
                    raise
    
    def _build_parquet_index(self) -> None:
        """Build index for Parquet files."""
        
        parquet_file = pq.ParquetFile(self.file_path)
        
        for batch_no, batch in enumerate(parquet_file.iter_batches()):
            # Convert batch to Python objects
            batch_dict = batch.to_pydict()
            
            # Assuming the parquet has columns matching task structure
            num_rows = len(batch_dict[list(batch_dict.keys())[0]])
            
            for row_idx in range(num_rows):
                # Reconstruct task from row
                task = {}
                for col_name, col_data in batch_dict.items():
                    task[col_name] = col_data[row_idx]
                
                global_idx = batch_no * batch.num_rows + row_idx
                self._process_task_for_index(task, global_idx)
    
    def _process_task_for_index(self, task: Dict[str, Any], index: int) -> None:
        """Process a single task for indexing."""
        
        # Apply filter if provided
        if self.filter_fn and not self.filter_fn(task):
            return
        
        # Extract metadata
        task_metadata = task.get("task_metadata", {})
        task_id = task_metadata.get("task_id", f"task_{index}")
        complexity = task_metadata.get("complexity", "unknown")
        category = task_metadata.get("category", "unknown")
        
        # Create index entry
        index_entry = {
            "task_id": task_id,
            "file_index": index,
            "complexity": complexity,
            "category": category,
            "estimated_turns": self._estimate_task_turns(complexity),
        }
        
        self._task_index.append(index_entry)
        
        # Update statistics
        self._statistics["total_tasks"] += 1
        self._statistics["complexity_counts"][complexity] += 1
        self._statistics["category_counts"][category] += 1
    
    def _estimate_task_turns(self, complexity: str) -> int:
        """Estimate number of turns for a task based on complexity."""
        
        turn_estimates = {
            "easy": 4,
            "medium": 6,
            "hard": 10,
        }
        
        return turn_estimates.get(complexity, 6)
    
    def _apply_sharding(self) -> None:
        """Apply sharding to task index for distributed training."""
        
        if self.num_shards <= 1:
            return
        
        # Use deterministic sharding based on task_id hash
        sharded_index = []
        for entry in self._task_index:
            task_hash = int(hashlib.md5(entry["task_id"].encode()).hexdigest(), 16)
            if task_hash % self.num_shards == self.shard_id:
                sharded_index.append(entry)
        
        logger.info(f"Sharding: {len(self._task_index)} -> {len(sharded_index)} tasks "
                   f"for shard {self.shard_id}/{self.num_shards}")
        
        self._task_index = sharded_index
    
    def _load_task_by_index(self, file_index: int) -> Optional[Dict[str, Any]]:
        """Load a specific task by its file index."""
        
        try:
            if self.file_format == 'parquet':
                return self._load_parquet_task(file_index)
            else:
                return self._load_json_task(file_index)
        except Exception as e:
            logger.error(f"Failed to load task at index {file_index}: {e}")
            return None
    
    def _load_json_task(self, file_index: int) -> Optional[Dict[str, Any]]:
        """Load task from JSON/JSONL file."""
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            if self.file_format == 'jsonl':
                # JSONL - read specific line
                for line_no, line in enumerate(f):
                    if line_no == file_index:
                        return json.loads(line.strip())
                return None
            else:
                # JSON - load and index
                data = json.load(f)
                if isinstance(data, list) and 0 <= file_index < len(data):
                    return data[file_index]
                elif file_index == 0:
                    return data
                return None
    
    def _load_parquet_task(self, file_index: int) -> Optional[Dict[str, Any]]:
        """Load task from Parquet file."""
        
        # This is simplified - in practice you'd want more efficient random access
        parquet_file = pq.ParquetFile(self.file_path)
        
        current_index = 0
        for batch in parquet_file.iter_batches():
            batch_size = batch.num_rows
            if current_index <= file_index < current_index + batch_size:
                # Found the right batch
                batch_dict = batch.to_pydict()
                row_idx = file_index - current_index
                
                # Reconstruct task
                task = {}
                for col_name, col_data in batch_dict.items():
                    task[col_name] = col_data[row_idx]
                
                return task
            
            current_index += batch_size
        
        return None
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific task by ID."""
        
        # Check cache first
        cached_task = self.cache.get(task_id)
        if cached_task is not None:
            return cached_task
        
        # Find in index
        for entry in self._task_index:
            if entry["task_id"] == task_id:
                task = self._load_task_by_index(entry["file_index"])
                if task:
                    self.cache.put(task_id, task)
                return task
        
        return None
    
    def get_tasks_by_complexity(self, complexity: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get tasks filtered by complexity level."""
        
        matching_entries = [
            entry for entry in self._task_index 
            if entry["complexity"] == complexity
        ]
        
        if limit:
            matching_entries = matching_entries[:limit]
        
        tasks = []
        for entry in matching_entries:
            task = self.get_task(entry["task_id"])
            if task:
                tasks.append(task)
        
        return tasks
    
    def get_random_tasks(self, count: int, complexity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get random tasks, optionally filtered by complexity."""
        
        if complexity:
            available_entries = [
                entry for entry in self._task_index 
                if entry["complexity"] == complexity
            ]
        else:
            available_entries = self._task_index
        
        if not available_entries:
            return []
        
        # Sample random entries
        sample_size = min(count, len(available_entries))
        sampled_entries = random.sample(available_entries, sample_size)
        
        tasks = []
        for entry in sampled_entries:
            task = self.get_task(entry["task_id"])
            if task:
                tasks.append(task)
        
        return tasks
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        
        return {
            **self._statistics,
            "cache_size": self.cache.size(),
            "cache_capacity": self.cache.capacity,
            "shard_info": {
                "shard_id": self.shard_id,
                "num_shards": self.num_shards,
                "tasks_in_shard": len(self._task_index),
            }
        }
    
    def __len__(self) -> int:
        """Get number of tasks in dataset."""
        return len(self._task_index)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all tasks in dataset."""
        
        for entry in self._task_index:
            task = self.get_task(entry["task_id"])
            if task:
                yield task


class TaskDataLoader:
    """
    Main data loader interface combining streaming, curriculum, and batching.
    
    This class provides the primary interface for loading task data during
    training. It combines the StreamingDataset, CurriculumSampler, and
    TaskBatcher to provide efficient, curriculum-aware data loading.
    
    Args:
        dataset: StreamingDataset instance
        curriculum_sampler: CurriculumSampler for progressive difficulty
        task_batcher: TaskBatcher for episode length balancing
        shuffle: Whether to shuffle tasks within each complexity level
    """
    
    def __init__(
        self,
        dataset: StreamingDataset,
        curriculum_sampler: Optional[CurriculumSampler] = None,
        task_batcher: Optional[TaskBatcher] = None,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.curriculum_sampler = curriculum_sampler or CurriculumSampler()
        self.task_batcher = task_batcher or TaskBatcher()
        self.shuffle = shuffle
        
        # Validate dataset has tasks
        if len(dataset) == 0:
            raise ValueError("Dataset is empty")
        
        logger.info(f"TaskDataLoader initialized with {len(dataset)} tasks")
    
    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a random batch of tasks."""
        
        tasks = self.dataset.get_random_tasks(batch_size * 2)  # Get extra for batching
        if not tasks:
            return []
        
        if self.shuffle:
            random.shuffle(tasks)
        
        # Create balanced batch
        batch = self.task_batcher.create_batch(tasks)
        
        # Ensure we don't exceed requested batch size
        return batch[:batch_size]
    
    def get_curriculum_batch(
        self,
        batch_size: int,
        epoch: int,
        total_epochs: int,
    ) -> List[Dict[str, Any]]:
        """Get a curriculum-aware batch of tasks."""
        
        # Determine complexity distribution for current epoch
        target_distribution = self.curriculum_sampler.get_current_distribution(epoch, total_epochs)
        
        # Calculate target counts for each complexity
        target_counts = {}
        total_weight = sum(target_distribution.values())
        
        for complexity, weight in target_distribution.items():
            target_count = max(1, int(batch_size * weight / total_weight))
            target_counts[complexity] = target_count
        
        # Collect tasks for each complexity level
        batch_tasks = []
        for complexity, count in target_counts.items():
            complexity_tasks = self.dataset.get_tasks_by_complexity(complexity, count * 2)
            
            if complexity_tasks:
                if self.shuffle:
                    random.shuffle(complexity_tasks)
                
                # Take required number
                selected_tasks = complexity_tasks[:count]
                batch_tasks.extend(selected_tasks)
                
                # Mark tasks as seen
                for task in selected_tasks:
                    task_id = task.get("task_metadata", {}).get("task_id", "unknown")
                    self.curriculum_sampler.mark_task_seen(task_id)
        
        # Create balanced batch
        if batch_tasks:
            batch = self.task_batcher.create_batch(batch_tasks)
            return batch[:batch_size]
        
        # Fallback to random batch
        return self.get_batch(batch_size)
    
    def get_epoch_iterator(
        self,
        batch_size: int,
        epoch: int,
        total_epochs: int,
        use_curriculum: bool = True,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Get iterator for a complete epoch."""
        
        # Estimate number of batches per epoch
        total_tasks = len(self.dataset)
        batches_per_epoch = max(1, total_tasks // batch_size)
        
        for batch_idx in range(batches_per_epoch):
            if use_curriculum:
                batch = self.get_curriculum_batch(batch_size, epoch, total_epochs)
            else:
                batch = self.get_batch(batch_size)
            
            if batch:
                yield batch
            else:
                logger.warning(f"Empty batch at epoch {epoch}, batch {batch_idx}")
                break
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics."""
        
        dataset_stats = self.dataset.get_statistics()
        curriculum_stats = self.curriculum_sampler.get_statistics()
        
        return {
            "dataset": dataset_stats,
            "curriculum": curriculum_stats,
            "batching": {
                "target_total_turns": self.task_batcher.target_total_turns,
                "max_batch_size": self.task_batcher.max_batch_size,
                "expected_turns": self.task_batcher.expected_turns,
            }
        }


# Utility functions for data loading

def create_data_loader(
    data_path: str,
    cache_size: int = 1000,
    batch_size: int = 8,
    num_shards: int = 1,
    shard_id: int = 0,
    seed: int = 42,
) -> TaskDataLoader:
    """
    Factory function to create a complete data loader setup.
    
    Args:
        data_path: Path to data file
        cache_size: Size of task cache
        batch_size: Target batch size
        num_shards: Number of distributed training shards
        shard_id: Current shard ID
        seed: Random seed
        
    Returns:
        Configured TaskDataLoader instance
    """
    
    # Create dataset
    dataset = StreamingDataset(
        file_path=data_path,
        cache_size=cache_size,
        shard_id=shard_id,
        num_shards=num_shards,
        seed=seed,
    )
    
    # Create curriculum sampler
    curriculum_sampler = CurriculumSampler(seed=seed)
    
    # Create task batcher
    task_batcher = TaskBatcher(
        target_total_turns=batch_size * 8,  # Estimate 8 turns per task on average
        max_batch_size=batch_size,
    )
    
    # Create data loader
    data_loader = TaskDataLoader(
        dataset=dataset,
        curriculum_sampler=curriculum_sampler,
        task_batcher=task_batcher,
        shuffle=True,
    )
    
    return data_loader


def validate_task_format(task: Dict[str, Any]) -> bool:
    """
    Validate that a task has the expected format.
    
    Args:
        task: Task dictionary to validate
        
    Returns:
        True if task format is valid
    """
    
    required_fields = ["task_metadata", "prompt", "reward_spec"]
    
    # Check required top-level fields
    for field in required_fields:
        if field not in task:
            return False
    
    # Check task_metadata structure
    task_metadata = task["task_metadata"]
    if not isinstance(task_metadata, dict):
        return False
    
    required_metadata = ["task_id"]
    for field in required_metadata:
        if field not in task_metadata:
            return False
    
    # Check prompt format (should be list of message dicts)
    prompt = task["prompt"]
    if not isinstance(prompt, list):
        return False
    
    for message in prompt:
        if not isinstance(message, dict) or "role" not in message or "content" not in message:
            return False
    
    # Check reward_spec structure
    reward_spec = task["reward_spec"]
    if not isinstance(reward_spec, dict):
        return False
    
    return True


if __name__ == "__main__":
    # Example usage and testing
    print("StreamingDataLoader module loaded successfully!")
    
    # Test LRU cache
    cache = LRUCache(capacity=3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.put("d", 4)  # Should evict "a"
    
    print(f"Cache test: a={cache.get('a')}, d={cache.get('d')}")  # a=None, d=4
    
    # Test curriculum sampler
    sampler = CurriculumSampler()
    dist_early = sampler.get_current_distribution(0, 10)
    dist_late = sampler.get_current_distribution(9, 10)
    
    print(f"Curriculum test: early={dist_early}, late={dist_late}")
    
    # Test task validation
    valid_task = {
        "task_metadata": {"task_id": "test_001", "complexity": "easy"},
        "prompt": [{"role": "user", "content": "Hello"}],
        "reward_spec": {"ground_truth": {}},
    }
    
    print(f"Task validation: {validate_task_format(valid_task)}")  # Should be True