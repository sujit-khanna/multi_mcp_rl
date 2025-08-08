#!/usr/bin/env python3
"""
Environment Adapter for GRPO Training
=====================================

This module provides integration between the existing MCPToolEnvironment
and the GRPO training infrastructure, allowing external policies to be used
with the environment.

Key Components:
- PolicyAdapter: Wraps training policy to work with MCPToolEnvironment
- SharedToolManagerEnvironment: Environment wrapper with shared tool manager
- TaskDataLoader: Loads and formats tasks from data/processed/train.json
- AsyncToSyncBridge: Handles async environment steps in sync training

Author: SkyRL Tool Agent Team
Date: 2025-08-02
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Union, Iterator
from dataclasses import dataclass, field
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

# Import existing environment components
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from environments.mcp_tool_environment import MCPToolEnvironment, BaseTextEnvStepOutput
from environments.real_tool_manager import RealMCPToolManager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PolicyInterface:
    """Interface for external policy to interact with environment"""
    generate_action: Callable[[List[Dict[str, str]]], str]
    reset: Optional[Callable[[], None]] = None
    update_state: Optional[Callable[[Dict[str, Any]], None]] = None


class PolicyAdapter:
    """
    Adapts external training policy to work with MCPToolEnvironment
    
    This removes dependency on PolicyIntegratedEnvironment by allowing
    any external policy to be used with the environment.
    """
    
    def __init__(self, policy_fn: Union[Callable, PolicyInterface]):
        """
        Initialize policy adapter
        
        Args:
            policy_fn: Either a callable that takes conversation history and returns action,
                      or a PolicyInterface with additional methods
        """
        if isinstance(policy_fn, PolicyInterface):
            self.policy = policy_fn
        else:
            # Wrap simple callable in interface
            self.policy = PolicyInterface(generate_action=policy_fn)
        
        self.conversation_history: List[Dict[str, str]] = []
        
    def get_action(self, observation: Union[str, List[Dict[str, str]]]) -> str:
        """
        Get action from policy given observation
        
        Args:
            observation: Either raw text or conversation history
            
        Returns:
            Action string containing reasoning and/or tool calls
        """
        # Update conversation history
        if isinstance(observation, str):
            # Convert string observation to message format
            self.conversation_history.append({
                "role": "user",
                "content": observation
            })
        elif isinstance(observation, list):
            # Extend with new messages
            self.conversation_history.extend(observation)
        
        # Generate action from policy
        action = self.policy.generate_action(self.conversation_history)
        
        # Add action to history
        self.conversation_history.append({
            "role": "assistant",
            "content": action
        })
        
        return action
    
    def reset(self):
        """Reset adapter state"""
        self.conversation_history = []
        if self.policy.reset:
            self.policy.reset()


class SharedToolManagerEnvironment(MCPToolEnvironment):
    """
    Extended MCPToolEnvironment that uses a shared tool manager
    
    This prevents creating new tool connections for each environment instance,
    which is critical for training efficiency.
    """
    
    def __init__(
        self,
        task_data: Dict[str, Any],
        shared_tool_manager: Optional[RealMCPToolManager] = None,
        policy_adapter: Optional[PolicyAdapter] = None
    ):
        """
        Initialize environment with shared resources
        
        Args:
            task_data: Task specification
            shared_tool_manager: Shared tool manager instance
            policy_adapter: External policy adapter
        """
        super().__init__(task_data)
        
        # Use shared tool manager if provided
        if shared_tool_manager:
            self.tool_manager = shared_tool_manager
            self.available_tools = self.tool_manager.get_available_tools()
            logger.info(f"Using shared tool manager with {len(self.available_tools)} tools")
        
        # Store policy adapter for potential use
        self.policy_adapter = policy_adapter
        
    async def initialize_tools(self):
        """Override to prevent re-initialization if using shared manager"""
        if self.tool_manager is None:
            # Only initialize if no shared manager
            await super().initialize_tools()
        else:
            logger.debug("Skipping tool initialization - using shared manager")


class AsyncToSyncBridge:
    """
    Bridge to handle async environment operations in sync training loop
    
    This allows the async MCPToolEnvironment to work with synchronous
    training code like PyTorch's training loops.
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop = None
        self.thread = None
        self._setup_event_loop()
        
    def _setup_event_loop(self):
        """Setup dedicated event loop in separate thread"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        # Wait for loop to be ready
        while self.loop is None:
            import time
            time.sleep(0.01)
    
    def run_async(self, coro):
        """Run async coroutine and return result synchronously"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
    
    def cleanup(self):
        """Cleanup event loop and thread"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)
        self.executor.shutdown(wait=True)


class TaskDataLoader:
    """
    Loads and formats tasks from processed training data
    
    Handles conversion from stored format to format expected by environment
    and training loop.
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize data loader
        
        Args:
            data_path: Path to processed training data JSON
        """
        self.data_path = Path(data_path)
        self.tasks = self._load_tasks()
        logger.info(f"Loaded {len(self.tasks)} tasks from {self.data_path}")
        
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load tasks from JSON file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            tasks = json.load(f)
        
        # Validate and format tasks
        formatted_tasks = []
        for i, task in enumerate(tasks):
            try:
                formatted_task = self._format_task(task, i)
                formatted_tasks.append(formatted_task)
            except Exception as e:
                logger.warning(f"Failed to format task {i}: {e}")
        
        return formatted_tasks
    
    def _format_task(self, task: Dict[str, Any], index: int) -> Dict[str, Any]:
        """
        Format task for environment compatibility
        
        Ensures task has all required fields and proper structure
        """
        # Extract or create task metadata
        task_metadata = task.get("task_metadata", {})
        if "task_id" not in task_metadata:
            task_metadata["task_id"] = task.get("extra_info", {}).get("task_id", f"task_{index}")
        if "complexity" not in task_metadata:
            task_metadata["complexity"] = task.get("extra_info", {}).get("complexity", "medium")
        if "category" not in task_metadata:
            task_metadata["category"] = task.get("extra_info", {}).get("category", "general")
        
        # Ensure prompt is properly formatted
        prompt = task.get("prompt", [])
        if not isinstance(prompt, list):
            prompt = [{"role": "user", "content": str(prompt)}]
        
        # Format reward specification
        reward_spec = task.get("reward_spec", {})
        if "ground_truth" not in reward_spec:
            reward_spec["ground_truth"] = {}
        
        # Add success criteria if missing
        ground_truth = reward_spec["ground_truth"]
        if "success_criteria" not in ground_truth:
            ground_truth["success_criteria"] = {
                "expected_tools": ground_truth.get("expected_tools", []),
                "min_tool_calls": 1,
                "min_reasoning_score": 60
            }
        
        # Return formatted task
        return {
            "task_metadata": task_metadata,
            "prompt": prompt,
            "reward_spec": reward_spec,
            "extra_info": task.get("extra_info", {}),
            "env_class": task.get("env_class", "mcp_tool_environment")
        }
    
    def get_task_iterator(self, shuffle: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Get iterator over tasks
        
        Args:
            shuffle: Whether to shuffle tasks
            
        Returns:
            Iterator over task dictionaries
        """
        tasks = self.tasks.copy()
        
        if shuffle:
            import random
            random.shuffle(tasks)
        
        for task in tasks:
            yield task
    
    def get_batch(self, batch_size: int, shuffle: bool = True) -> List[Dict[str, Any]]:
        """
        Get batch of tasks
        
        Args:
            batch_size: Number of tasks to return
            shuffle: Whether to shuffle before sampling
            
        Returns:
            List of task dictionaries
        """
        tasks = self.tasks.copy()
        
        if shuffle:
            import random
            random.shuffle(tasks)
        
        return tasks[:batch_size]


class EnvironmentAdapter:
    """
    Main adapter class that integrates all components
    
    This is the primary interface for using MCPToolEnvironment with
    external training policies.
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        shared_tool_manager: Optional[RealMCPToolManager] = None,
        use_async_bridge: bool = True
    ):
        """
        Initialize environment adapter
        
        Args:
            data_path: Path to training data
            shared_tool_manager: Optional shared tool manager
            use_async_bridge: Whether to use async-to-sync bridge
        """
        self.data_loader = TaskDataLoader(data_path)
        self.shared_tool_manager = shared_tool_manager
        self.async_bridge = AsyncToSyncBridge() if use_async_bridge else None
        
        # Initialize shared tool manager if not provided
        if self.shared_tool_manager is None:
            logger.info("Initializing shared tool manager...")
            self._initialize_shared_tools()
    
    def _initialize_shared_tools(self):
        """Initialize shared tool manager"""
        self.shared_tool_manager = RealMCPToolManager()
        
        if self.async_bridge:
            # Initialize asynchronously
            self.async_bridge.run_async(self.shared_tool_manager.initialize())
        else:
            # Run in new event loop
            loop = asyncio.new_event_loop()
            loop.run_until_complete(self.shared_tool_manager.initialize())
            loop.close()
    
    def create_environment(
        self,
        task_data: Dict[str, Any],
        policy_fn: Optional[Union[Callable, PolicyInterface]] = None
    ) -> SharedToolManagerEnvironment:
        """
        Create environment instance with shared resources
        
        Args:
            task_data: Task specification
            policy_fn: Optional external policy function
            
        Returns:
            Configured environment instance
        """
        # Create policy adapter if provided
        policy_adapter = PolicyAdapter(policy_fn) if policy_fn else None
        
        # Create environment with shared resources
        env = SharedToolManagerEnvironment(
            task_data=task_data,
            shared_tool_manager=self.shared_tool_manager,
            policy_adapter=policy_adapter
        )
        
        return env
    
    def step_environment(
        self,
        env: SharedToolManagerEnvironment,
        action: str
    ) -> BaseTextEnvStepOutput:
        """
        Execute environment step, handling async operations
        
        Args:
            env: Environment instance
            action: Action to execute
            
        Returns:
            Step output with observations, reward, done flag
        """
        if self.async_bridge and hasattr(env.step, '__call__'):
            # Check if step might have async operations
            try:
                # Try direct step first
                return env.step(action)
            except RuntimeError as e:
                if "asyncio" in str(e).lower():
                    # Handle async operation
                    return self.async_bridge.run_async(self._async_step(env, action))
                raise
        else:
            # Direct synchronous step
            return env.step(action)
    
    async def _async_step(self, env: SharedToolManagerEnvironment, action: str):
        """Async wrapper for environment step"""
        # Ensure tools are initialized
        if env.tool_manager is None:
            await env.initialize_tools()
        return env.step(action)
    
    def create_training_iterator(
        self,
        policy_fn: Union[Callable, PolicyInterface],
        batch_size: int = 1,
        max_episodes_per_task: int = 1,
        shuffle: bool = True
    ) -> Iterator[Dict[str, Any]]:
        """
        Create iterator for training that yields episode data
        
        Args:
            policy_fn: Policy function to use
            batch_size: Number of tasks to process in parallel
            max_episodes_per_task: Episodes to run per task
            shuffle: Whether to shuffle tasks
            
        Yields:
            Episode data dictionaries with trajectories and rewards
        """
        for task_batch in self._batch_tasks(batch_size, shuffle):
            # Process batch of tasks
            batch_results = []
            
            for task_data in task_batch:
                for episode in range(max_episodes_per_task):
                    episode_data = self._run_episode(task_data, policy_fn)
                    batch_results.append(episode_data)
            
            yield {
                "episodes": batch_results,
                "batch_size": len(batch_results)
            }
    
    def _batch_tasks(self, batch_size: int, shuffle: bool) -> Iterator[List[Dict[str, Any]]]:
        """Create batches of tasks"""
        tasks = list(self.data_loader.get_task_iterator(shuffle=shuffle))
        
        for i in range(0, len(tasks), batch_size):
            yield tasks[i:i + batch_size]
    
    def _run_episode(
        self,
        task_data: Dict[str, Any],
        policy_fn: Union[Callable, PolicyInterface]
    ) -> Dict[str, Any]:
        """
        Run single episode and collect trajectory
        
        Args:
            task_data: Task specification
            policy_fn: Policy to use
            
        Returns:
            Episode data with trajectory and rewards
        """
        # Create environment and policy adapter
        env = self.create_environment(task_data, policy_fn)
        policy_adapter = env.policy_adapter or PolicyAdapter(policy_fn)
        
        # Initialize trajectory storage
        trajectory = {
            "task_id": task_data["task_metadata"]["task_id"],
            "observations": [],
            "actions": [],
            "rewards": [],
            "done": False,
            "info": []
        }
        
        # Get initial observation
        initial_prompt = task_data["prompt"]
        observation = initial_prompt
        
        # Run episode
        done = False
        total_reward = 0.0
        
        while not done:
            # Get action from policy
            action = policy_adapter.get_action(observation)
            trajectory["actions"].append(action)
            
            # Step environment
            step_output = self.step_environment(env, action)
            
            # Handle different output formats
            if hasattr(step_output, '__dict__'):
                # Object with attributes
                observation = step_output.observations
                reward = step_output.reward
                done = step_output.done
                info = step_output.metadata
            else:
                # Dictionary format
                observation = step_output.get("observations", "")
                reward = step_output.get("reward", 0.0)
                done = step_output.get("done", False)
                info = step_output.get("metadata", {})
            
            # Store step data
            trajectory["observations"].append(observation)
            trajectory["rewards"].append(reward)
            trajectory["info"].append(info)
            total_reward += reward
        
        trajectory["done"] = done
        trajectory["total_reward"] = total_reward
        
        # Add final metrics
        trajectory["final_metrics"] = {
            "task_completed": env.state.task_completed,
            "total_tool_calls": len(env.state.tool_calls_made),
            "turn_count": env.state.turn_count,
            "reward_breakdown": env._get_reward_breakdown()
        }
        
        return trajectory
    
    def cleanup(self):
        """Cleanup resources"""
        if self.async_bridge:
            self.async_bridge.cleanup()
        if self.shared_tool_manager:
            if hasattr(self.shared_tool_manager, 'cleanup'):
                asyncio.run(self.shared_tool_manager.cleanup())


def create_environment_adapter(
    data_path: str = "data/processed/train.json",
    use_shared_tools: bool = True
) -> EnvironmentAdapter:
    """
    Factory function to create configured environment adapter
    
    Args:
        data_path: Path to training data
        use_shared_tools: Whether to use shared tool manager
        
    Returns:
        Configured EnvironmentAdapter instance
    """
    # Resolve data path
    if not Path(data_path).is_absolute():
        # Make relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / data_path
    
    # Create adapter
    adapter = EnvironmentAdapter(
        data_path=data_path,
        shared_tool_manager=None if not use_shared_tools else RealMCPToolManager(),
        use_async_bridge=True
    )
    
    return adapter


if __name__ == "__main__":
    # Test the adapter
    logging.basicConfig(level=logging.INFO)
    
    # Create adapter
    adapter = create_environment_adapter()
    
    # Test with dummy policy
    def dummy_policy(conversation: List[Dict[str, str]]) -> str:
        """Simple test policy"""
        return "<think>Testing the adapter</think>\n<tool_call>{\"name\": \"fmp_get_quote\", \"arguments\": {\"symbol\": \"AAPL\"}}</tool_call>"
    
    # Get a task and run episode
    task = adapter.data_loader.get_batch(1)[0]
    logger.info(f"Testing with task: {task['task_metadata']['task_id']}")
    
    episode_data = adapter._run_episode(task, dummy_policy)
    logger.info(f"Episode completed: reward={episode_data['total_reward']:.3f}")
    
    # Cleanup
    adapter.cleanup()