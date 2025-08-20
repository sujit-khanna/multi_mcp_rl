"""
Trajectory Collector: Parallel rollout collection for GRPO training

This module implements efficient parallel collection of trajectories from multiple
MCPToolEnvironment instances for training with Group Relative Policy Optimization.
"""

import asyncio
import copy
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any, Callable, Tuple
import warnings
import numpy as np

import torch
import numpy as np

# Import path adjustment for MCPToolEnvironment
import sys
import os
from pathlib import Path

# Add the skyrl_tool_agent root directory to path
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent  # training -> skyrl_tool_agent
environments_dir = root_dir / "environments"

if str(environments_dir) not in sys.path:
    sys.path.insert(0, str(environments_dir))

try:
    from mcp_tool_environment import MCPToolEnvironment
    # Use the real tool manager from environments directory
    from real_tool_manager import RealMCPToolManager as SimpleSharedManager
    ENVIRONMENT_IMPORTS_AVAILABLE = True
except ImportError as e:
    # Only warn if we can't find the specific modules, not for general import issues
    if "mcp_tool_environment" in str(e) or "RealMCPToolManager" in str(e):
        warnings.warn(f"Could not import environment modules (may be expected during testing): {e}")
    MCPToolEnvironment = None
    SimpleSharedManager = None
    ENVIRONMENT_IMPORTS_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EpisodeResult:
    """
    Container for a single episode result with all trajectory data.
    
    This matches the format expected by the GRPO trainer and is compatible
    with PolicyIntegratedEnvironment output format.
    """
    
    def __init__(
        self,
        task_id: str,
        trajectory: List[Dict[str, Any]],
        total_reward: float = 0.0,
        turns: int = 0,
        task_completed: bool = False,
        tools_used: int = 0,
        expected_tools: Optional[List[str]] = None,
        success: bool = False,
        reward_breakdown: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
        execution_time: float = 0.0,
        initial_prompt: Optional[List[Dict[str, str]]] = None,
    ):
        self.task_id = task_id
        self.trajectory = trajectory
        self.total_reward = total_reward
        self.turns = turns
        self.task_completed = task_completed
        self.tools_used = tools_used
        self.expected_tools = expected_tools or []
        self.success = success
        self.reward_breakdown = reward_breakdown or {}
        self.error = error
        self.execution_time = execution_time
        self.initial_prompt = initial_prompt or []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        return {
            "task_id": self.task_id,
            "trajectory": self.trajectory,
            "total_reward": self.total_reward,
            "turns": self.turns,
            "task_completed": self.task_completed,
            "tools_used": self.tools_used,
            "expected_tools": self.expected_tools,
            "success": self.success,
            "reward_breakdown": self.reward_breakdown,
            "error": self.error,
            "execution_time": self.execution_time,
            "initial_prompt": self.initial_prompt,
        }
    
    @property
    def rewards(self) -> List[float]:
        """Extract individual step rewards from trajectory."""
        return [turn.get("reward", 0.0) for turn in self.trajectory if "reward" in turn]
    
    @property  
    def actions(self) -> List[str]:
        """Extract actions from trajectory."""
        return [turn.get("action", "") for turn in self.trajectory if "action" in turn]
    
    @property
    def states(self) -> List[Any]:
        """Extract states from trajectory.""" 
        return [turn.get("state", []) for turn in self.trajectory if "state" in turn]

    def is_valid(self) -> bool:
        """Check if episode result is valid for training."""
        return (
            self.error is None and
            len(self.trajectory) > 0 and
            self.turns > 0 and
            all("action" in turn and "reward" in turn for turn in self.trajectory)
        )


class TrajectoryCollector:
    """
    Parallel trajectory collector for GRPO training.
    
    This class manages multiple MCPToolEnvironment instances to collect
    trajectories in parallel, using a shared tool manager for efficiency.
    It handles the complete pipeline from task specification to formatted
    trajectory data ready for GRPO training.
    
    Key Features:
    - Parallel episode collection using asyncio
    - Shared tool manager for efficient resource usage
    - Proper async/sync handling for tool execution
    - Comprehensive error handling and retry logic
    - Memory-efficient trajectory formatting
    - Detailed logging and monitoring
    
    Args:
        policy: QwenPolicy instance for action generation
        env_factory: Factory function to create MCPToolEnvironment instances
        num_parallel_envs: Number of parallel environments to run
        shared_tool_manager: Shared tool manager instance
        max_episode_length: Maximum number of turns per episode
        retry_failed_episodes: Whether to retry failed episodes
        collect_log_probs: Whether to collect log probabilities during rollout
    """
    
    def __init__(
        self,
        policy,  # QwenPolicy instance
        env_factory: Callable[[Dict[str, Any]], Any],  # Function that creates MCPToolEnvironment
        num_parallel_envs: int = 1,  # CRITICAL FIX: Reduced from 4 to 1 to prevent deadlock
        shared_tool_manager: Any = None,  # SimpleSharedManager instance
        max_episode_length: int = 15,
        retry_failed_episodes: bool = True,
        collect_log_probs: bool = True,
        executor_max_workers: int = 1,  # CRITICAL FIX: Reduced from 8 to 1 to prevent deadlock
    ):
        """Initialize the trajectory collector."""
        
        self.policy = policy
        self.env_factory = env_factory
        self.num_parallel_envs = num_parallel_envs
        self.shared_tool_manager = shared_tool_manager
        self.max_episode_length = max_episode_length
        self.retry_failed_episodes = retry_failed_episodes
        self.collect_log_probs = collect_log_probs
        
        # Thread pool executor for synchronous tool execution
        self.executor = ThreadPoolExecutor(max_workers=executor_max_workers)
        
        # Semaphore to limit concurrent environments
        self.env_semaphore = asyncio.Semaphore(num_parallel_envs)
        
        # Statistics tracking
        self.total_episodes_collected = 0
        self.total_failed_episodes = 0
        self.total_retry_attempts = 0
        self.collection_times = []
        
        # Ensure policy is in evaluation mode for rollouts
        self.policy.enable_eval_mode()
        
        logger.info(f"TrajectoryCollector initialized with {num_parallel_envs} parallel environments")
    
    async def collect_batch(
        self,
        tasks: List[Dict[str, Any]],
        batch_timeout: float = 3600.0,  # 1 hour timeout per batch
    ) -> List[EpisodeResult]:
        """
        Collect a batch of trajectories in parallel.
        
        This is the main method for trajectory collection. It takes a list of
        task specifications and returns a list of completed episode results.
        
        Args:
            tasks: List of task dictionaries with task_metadata, prompt, reward_spec
            batch_timeout: Maximum time to wait for batch completion
            
        Returns:
            List of EpisodeResult objects
        """
        
        if not tasks:
            logger.warning("No tasks provided for batch collection")
            return []
        
        start_time = time.time()
        logger.info(f"Starting batch collection of {len(tasks)} tasks")
        
        # Create semaphore for this batch
        batch_semaphore = asyncio.Semaphore(self.num_parallel_envs)
        
        # Create collection tasks
        collection_tasks = []
        for i, task_data in enumerate(tasks):
            collection_task = asyncio.create_task(
                self._collect_single_episode_with_semaphore(
                    task_data, batch_semaphore, episode_id=i
                )
            )
            collection_tasks.append(collection_task)
        
        # Wait for all tasks to complete with timeout
        try:
            episode_results = await asyncio.wait_for(
                asyncio.gather(*collection_tasks, return_exceptions=True),
                timeout=batch_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Batch collection timed out after {batch_timeout} seconds")
            # Cancel remaining tasks
            for task in collection_tasks:
                if not task.done():
                    task.cancel()
            episode_results = []
        
        # Process results and handle exceptions
        valid_results = []
        for i, result in enumerate(episode_results):
            if isinstance(result, Exception):
                logger.error(f"Episode {i} failed with exception: {result}")
                # Create failed episode result
                task_id = tasks[i].get("task_metadata", {}).get("task_id", f"task_{i}")
                failed_result = EpisodeResult(
                    task_id=task_id,
                    trajectory=[],
                    error=str(result),
                    initial_prompt=tasks[i].get("prompt", [])
                )
                valid_results.append(failed_result)
            elif isinstance(result, EpisodeResult):
                valid_results.append(result)
            else:
                logger.warning(f"Unexpected result type for episode {i}: {type(result)}")
        
        # Update statistics
        collection_time = time.time() - start_time
        self.collection_times.append(collection_time)
        self.total_episodes_collected += len(valid_results)
        
        # Log batch statistics
        successful_episodes = sum(1 for result in valid_results if result.is_valid())
        failed_episodes = len(valid_results) - successful_episodes
        
        logger.info(f"Batch collection completed in {collection_time:.2f}s: "
                   f"{successful_episodes} successful, {failed_episodes} failed")
        
        return valid_results
    
    async def _collect_single_episode_with_semaphore(
        self,
        task_data: Dict[str, Any],
        semaphore: asyncio.Semaphore,
        episode_id: int = 0,
    ) -> EpisodeResult:
        """Collect single episode with semaphore for concurrency control."""
        
        async with semaphore:
            return await self._collect_single_episode(task_data, episode_id)
    
    async def _collect_single_episode(
        self,
        task_data: Dict[str, Any],
        episode_id: int = 0,
        max_retries: int = 2,
    ) -> EpisodeResult:
        """
        Collect a single episode trajectory.
        
        This method handles the complete episode collection pipeline:
        1. Initialize environment with task data
        2. Run episode loop with policy actions
        3. Collect trajectory data at each turn
        4. Format result for GRPO training
        
        Args:
            task_data: Task specification dictionary
            episode_id: Unique identifier for this episode
            max_retries: Maximum number of retry attempts
            
        Returns:
            EpisodeResult with complete trajectory data
        """
        
        start_time = time.time()
        task_id = task_data.get("task_metadata", {}).get("task_id", f"episode_{episode_id}")
        
        for attempt in range(max_retries + 1):
            try:
                # Create environment
                env = await self._create_environment(task_data)
                if env is None:
                    raise RuntimeError("Failed to create environment")
                
                # Run episode
                trajectory_data = await self._run_episode(env, task_data, task_id)
                
                # Create episode result
                result = self._format_episode_result(
                    task_data=task_data,
                    trajectory_data=trajectory_data,
                    execution_time=time.time() - start_time
                )
                
                # Cleanup environment
                await self._cleanup_environment(env)
                
                return result
                
            except Exception as e:
                logger.error(f"Episode {task_id} attempt {attempt + 1} failed: {e}")
                if attempt < max_retries and self.retry_failed_episodes:
                    self.total_retry_attempts += 1
                    logger.info(f"Retrying episode {task_id} (attempt {attempt + 2})")
                    await asyncio.sleep(1.0)  # Brief delay before retry
                else:
                    # Final failure
                    self.total_failed_episodes += 1
                    return EpisodeResult(
                        task_id=task_id,
                        trajectory=[],
                        error=f"Failed after {attempt + 1} attempts: {str(e)}",
                        execution_time=time.time() - start_time,
                        initial_prompt=task_data.get("prompt", [])
                    )
    
    async def _create_environment(self, task_data: Dict[str, Any]) -> Optional[Any]:
        """Create and initialize MCPToolEnvironment."""
        
        try:
            # Run environment creation in executor since it may involve sync operations
            env = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.env_factory,
                task_data
            )
            
            # Set shared tool manager if provided
            if self.shared_tool_manager is not None:
                env.tool_manager = self.shared_tool_manager
            
            return env
            
        except Exception as e:
            logger.error(f"Failed to create environment: {e}")
            return None
    
    async def _run_episode(
        self,
        env: Any,  # MCPToolEnvironment instance
        task_data: Dict[str, Any],
        task_id: str,
    ) -> Dict[str, Any]:
        """
        Run a complete episode and collect trajectory data.
        
        This method implements the core episode loop:
        1. Get initial observation from environment
        2. Generate action using policy
        3. Execute action in environment
        4. Collect rewards and observations
        5. Repeat until done or max length reached
        
        Args:
            env: MCPToolEnvironment instance
            task_data: Task specification
            task_id: Task identifier
            
        Returns:
            Dictionary with trajectory data and episode statistics
        """
        
        logger.info(f"🎮 Starting episode run for task {task_id}")
        logger.info(f"   Max episode length: {self.max_episode_length}")
        
        trajectory = []
        total_reward = 0.0
        turn = 0
        done = False
        
        # Get initial prompt from task data
        initial_prompt = task_data.get("prompt", [])
        
        # Initialize conversation history
        conversation_history = copy.deepcopy(initial_prompt)
        
        try:
            while not done and turn < self.max_episode_length:
                turn += 1
                logger.info(f"🔄 Episode {task_id} - Turn {turn}/{self.max_episode_length}")
                
                # Generate action using policy
                logger.info(f"⚡ Generating action for turn {turn}...")
                action = await self._generate_action(conversation_history)
                logger.info(f"📝 Generated action length: {len(action)} chars")
                logger.info(f"   Action preview: {action[:100]}...")
                
                # CRITICAL FIX: Check for repetitive generation and terminate early
                if self._is_repetitive_generation(action):
                    logger.warning(f"⚠️ Detected repetitive generation, terminating episode early")
                    done = True
                    break
                
                # Execute action in environment
                logger.info(f"🔧 Executing action in environment...")
                step_result = await self._execute_environment_step(env, action)
                logger.info(f"✅ Environment step completed")
                
                # Extract step data
                observation = step_result.get("observation", "")
                reward = step_result.get("reward", 0.0)
                done = step_result.get("done", False)
                step_metadata = step_result.get("metadata", {})
                
                # Update conversation history
                # Add assistant action
                conversation_history.append({
                    "role": "assistant",
                    "content": action
                })
                
                # Add observation if present
                if observation:
                    conversation_history.append({
                        "role": "user", 
                        "content": observation
                    })
                
                # Collect log probabilities if requested
                log_prob = None
                if self.collect_log_probs:
                    # CRITICAL FIX: Use sample-time logprobs from vLLM if available
                    # This is essential for proper PPO ratio computation
                    sample_time_logprob = None
                    if hasattr(self.policy, 'get_last_sample_logprobs'):
                        sample_logprobs = self.policy.get_last_sample_logprobs()
                        # vLLM generates one response at a time, so we always use index 0
                        if sample_logprobs and len(sample_logprobs) > 0:
                            # Get the logprobs for the FIRST (and only) response
                            turn_logprobs = sample_logprobs[0]  # Always index 0 for single generation
                            if turn_logprobs:
                                # Sum the logprobs for all tokens in this action
                                sample_time_logprob = sum(turn_logprobs) if isinstance(turn_logprobs, list) else turn_logprobs
                                logger.info(f"✅ Using sample-time logprob for turn {turn}: {sample_time_logprob:.4f}")
                            else:
                                logger.warning(f"⚠️ Empty logprobs list for turn {turn}")
                        else:
                            logger.warning(f"⚠️ No sample logprobs available from vLLM for turn {turn}")
                    
                    if sample_time_logprob is not None:
                        log_prob = sample_time_logprob
                    else:
                        # Fallback: compute with current model (NOT ideal for PPO)
                        if turn == 0:  # Only warn once per episode
                            logger.warning("No sample-time logprobs available, computing with current model (PPO ratio will be ~1.0!)")
                        try:
                            # Clear GPU cache before computing log probs to prevent OOM
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            
                            # Get log probability for this action
                            states = [conversation_history[:-2]]  # State before action
                            actions = [action]
                            
                            # Compute without gradients to save memory
                            with torch.no_grad():
                                log_probs = self.policy.compute_log_probs(states, actions)
                                log_prob = log_probs[0].item() if len(log_probs) > 0 else None
                        except torch.cuda.OutOfMemoryError:
                            logger.warning("CUDA OOM when computing log probabilities, using default value")
                            log_prob = -1.0  # Default value for OOM case
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as e:
                            logger.warning(f"Failed to compute log probabilities: {e}")
                            log_prob = -1.0  # Default value for other errors
                
                # Create turn data
                turn_data = {
                    "turn": turn,
                    "action": action,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "metadata": {
                        **step_metadata,
                        "log_prob": log_prob,
                        "conversation_length": len(conversation_history),
                    }
                }
                
                trajectory.append(turn_data)
                total_reward += reward
                
                # Check for early termination conditions
                if done:
                    logger.debug(f"Episode {task_id} completed at turn {turn}")
                    break
                
                # Check for critical failures
                if step_metadata.get("critical_failure", False):
                    logger.warning(f"Episode {task_id} terminated due to critical failure")
                    done = True
                    break
        
        except Exception as e:
            logger.error(f"Error during episode execution for {task_id}: {e}")
            # Add error information to last turn if available
            if trajectory:
                trajectory[-1]["metadata"]["execution_error"] = str(e)
        
        # Get final environment state
        try:
            env_state = getattr(env, "state", None)
            task_completed = getattr(env_state, "task_completed", False) if env_state else False
            tool_calls_made = getattr(env_state, "tool_calls_made", []) if env_state else []
            tools_used = len(tool_calls_made) if tool_calls_made else len([
                t for t in trajectory if "<tool_call>" in t.get("action", "")
            ])
            
            # Get reward breakdown from final state
            reward_components = getattr(env, "reward_components", None)
            reward_breakdown = {}
            if reward_components:
                reward_breakdown = {
                    "total": getattr(reward_components, "total", total_reward),
                    "task_completion": getattr(reward_components, "task_completion", 0.0),
                    "reasoning_quality": getattr(reward_components, "reasoning_quality", 0.0),
                    "tool_efficiency": getattr(reward_components, "tool_efficiency", 0.0),
                }
        except Exception as e:
            logger.warning(f"Failed to extract environment state: {e}")
            task_completed = False
            tools_used = 0
            reward_breakdown = {"total": total_reward}
        
        return {
            "trajectory": trajectory,
            "total_reward": total_reward,
            "turns": turn,
            "task_completed": task_completed,
            "tools_used": tools_used,
            "reward_breakdown": reward_breakdown,
            "conversation_history": conversation_history,
        }
    
    async def _generate_action(self, conversation_history: List[Dict[str, str]]) -> str:
        """Generate action using policy."""
        
        try:
            # Run policy generation in executor with timeout to avoid blocking
            states = [conversation_history]
            logger.info("🎯 TrajectoryCollector._generate_action starting...")
            logger.info(f"   Conversation history length: {len(conversation_history)} messages")
            logger.info(f"   Executor max workers: {self.executor._max_workers}")
            
            # Add timeout to prevent infinite hangs
            try:
                logger.info("📞 Submitting policy.generate_action to executor...")
                start_time = time.time()
                actions = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self.policy.generate_action,
                        states
                    ),
                    timeout=600.0  # 10 minute timeout for policy generation
                )
                execution_time = time.time() - start_time
                logger.info(f"✅ Policy generation completed in {execution_time:.2f} seconds")
                logger.info(f"   Generated {len(actions)} actions")
            except asyncio.TimeoutError:
                logger.error("⏰ Policy generation timed out after 600 seconds")
                return "ERROR: Policy generation timed out"
            
            if actions and len(actions) > 0:
                return actions[0]
            else:
                logger.warning("Policy returned empty action, using fallback")
                return "I need to think about this task."
                
        except Exception as e:
            logger.error(f"Error generating action: {e}")
            return f"Error generating action: {str(e)}"
    
    def _is_repetitive_generation(self, text: str) -> bool:
        """Detect if the generated text contains excessive repetition."""
        if len(text) < 100:
            return False
        
        # Check for repeated phrases (common in broken generation)
        words = text.split()
        if len(words) < 20:
            return False
        
        # Look for repeated sequences of 3+ words
        for i in range(len(words) - 6):
            phrase = ' '.join(words[i:i+3])
            rest_text = ' '.join(words[i+3:])
            if rest_text.count(phrase) >= 3:  # Same phrase repeated 3+ times
                logger.debug(f"Detected repetitive phrase: '{phrase}'")
                return True
        
        # Check for high ratio of repeated tokens
        unique_words = set(words)
        if len(words) > 50 and len(unique_words) / len(words) < 0.3:  # Less than 30% unique words
            logger.debug(f"Low word diversity: {len(unique_words)}/{len(words)} = {len(unique_words)/len(words):.2f}")
            return True
        
        return False
    
    async def _execute_environment_step(
        self,
        env: Any,
        action: str,
    ) -> Dict[str, Any]:
        """Execute action in environment and return step result."""
        
        try:
            # Check if policy indicates this action was forced
            forced_action = False
            if hasattr(self.policy, 'last_forced_mask') and self.policy.last_forced_mask:
                # Use the last forced mask entry (most recent action)
                if len(self.policy.last_forced_mask) > 0:
                    forced_action = self.policy.last_forced_mask[-1]
            
            # Set forced action flag on environment before step
            if hasattr(env, '_last_action_was_forced'):
                env._last_action_was_forced = forced_action
            
            # Run environment step in executor since it may involve sync tool calls
            step_result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                env.step,
                action
            )
            
            # Ensure step_result is a dictionary
            if not isinstance(step_result, dict):
                # Try to convert from tuple format (observation, reward, done, info)
                if isinstance(step_result, (tuple, list)) and len(step_result) >= 3:
                    observation, reward, done = step_result[:3]
                    info = step_result[3] if len(step_result) > 3 else {}
                    step_result = {
                        "observation": observation,
                        "reward": reward,
                        "done": done,
                        "metadata": info
                    }
                else:
                    # Fallback for unexpected format
                    step_result = {
                        "observation": str(step_result),
                        "reward": 0.0,
                        "done": False,
                        "metadata": {}
                    }
            
            return step_result
            
        except Exception as e:
            logger.error(f"Error executing environment step: {e}")
            return {
                "observation": f"Error executing action: {str(e)}",
                "reward": -1.0,  # Negative reward for errors
                "done": True,
                "metadata": {"error": str(e), "critical_failure": True}
            }
    
    def _format_episode_result(
        self,
        task_data: Dict[str, Any],
        trajectory_data: Dict[str, Any],
        execution_time: float,
    ) -> EpisodeResult:
        """Format trajectory data into EpisodeResult."""
        
        task_metadata = task_data.get("task_metadata", {})
        reward_spec = task_data.get("reward_spec", {})
        
        # Extract expected tools from success criteria
        expected_tools = []
        success_criteria = reward_spec.get("ground_truth", {})
        if "expected_tools" in success_criteria:
            expected_tools = success_criteria["expected_tools"]
        
        # Determine success based on task completion and reward
        success = (
            trajectory_data.get("task_completed", False) and
            trajectory_data.get("total_reward", 0.0) > 0.0
        )
        
        return EpisodeResult(
            task_id=task_metadata.get("task_id", "unknown"),
            trajectory=trajectory_data.get("trajectory", []),
            total_reward=trajectory_data.get("total_reward", 0.0),
            turns=trajectory_data.get("turns", 0),
            task_completed=trajectory_data.get("task_completed", False),
            tools_used=trajectory_data.get("tools_used", 0),
            expected_tools=expected_tools,
            success=success,
            reward_breakdown=trajectory_data.get("reward_breakdown", {}),
            execution_time=execution_time,
            initial_prompt=task_data.get("prompt", []),
        )
    
    async def _cleanup_environment(self, env: Any) -> None:
        """Clean up environment resources."""
        
        try:
            # Run cleanup in executor if environment has cleanup method
            if hasattr(env, "cleanup"):
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    env.cleanup
                )
            elif hasattr(env, "close"):
                await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    env.close
                )
        except Exception as e:
            logger.warning(f"Error during environment cleanup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        
        avg_collection_time = np.mean(self.collection_times) if self.collection_times else 0.0
        success_rate = (
            (self.total_episodes_collected - self.total_failed_episodes) / 
            self.total_episodes_collected
        ) if self.total_episodes_collected > 0 else 0.0
        
        return {
            "total_episodes_collected": self.total_episodes_collected,
            "total_failed_episodes": self.total_failed_episodes,
            "total_retry_attempts": self.total_retry_attempts,
            "success_rate": success_rate,
            "avg_collection_time": avg_collection_time,
            "parallel_envs": self.num_parallel_envs,
            "max_episode_length": self.max_episode_length,
        }
    
    async def cleanup(self) -> None:
        """Cleanup collector resources."""
        
        logger.info("Cleaning up TrajectoryCollector")
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"Final collection statistics: {stats}")
    
    def __repr__(self) -> str:
        """String representation of the collector."""
        return (f"TrajectoryCollector(parallel_envs={self.num_parallel_envs}, "
                f"max_length={self.max_episode_length}, collected={self.total_episodes_collected})")


# Utility functions for trajectory collection

async def collect_trajectories_for_training(
    policy,
    env_factory: Callable,
    tasks: List[Dict[str, Any]],
    num_parallel_envs: int = 4,
    shared_tool_manager: Any = None,
    max_episode_length: int = 15,
) -> List[EpisodeResult]:
    """
    Convenience function to collect trajectories for training.
    
    Args:
        policy: QwenPolicy instance
        env_factory: Function to create environments
        tasks: List of task specifications
        num_parallel_envs: Number of parallel environments
        shared_tool_manager: Shared tool manager instance
        max_episode_length: Maximum episode length
        
    Returns:
        List of EpisodeResult objects
    """
    
    collector = TrajectoryCollector(
        policy=policy,
        env_factory=env_factory,
        num_parallel_envs=num_parallel_envs,
        shared_tool_manager=shared_tool_manager,
        max_episode_length=max_episode_length,
    )
    
    try:
        results = await collector.collect_batch(tasks)
        return results
    finally:
        await collector.cleanup()


def convert_episode_results_to_grpo_trajectories(
    episode_results: List[EpisodeResult]
) -> List[Any]:  # List[Trajectory] from grpo_trainer
    """
    Convert EpisodeResult objects to GRPO Trajectory objects.
    
    Args:
        episode_results: List of EpisodeResult objects
        
    Returns:
        List of Trajectory objects ready for GRPO training
    """
    
    # Import here to avoid circular imports
    from ..core.grpo_trainer import Trajectory
    
    grpo_trajectories = []
    
    for episode in episode_results:
        if not episode.is_valid():
            continue
        
        # Extract states (conversation states at each turn)
        states = []
        actions = []
        rewards = []
        dones = []
        
        for turn_data in episode.trajectory:
            # Reconstruct conversation state before this action
            # This is a simplified version - in practice you'd need the full conversation history
            state = [{"role": "user", "content": "Task prompt"}]  # Placeholder
            
            states.append(state)
            actions.append(turn_data["action"])
            rewards.append(turn_data["reward"])
            dones.append(turn_data["done"])
        
        if states and actions:
            trajectory = Trajectory(
                task_id=episode.task_id,
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
            )
            grpo_trajectories.append(trajectory)
    
    return grpo_trajectories


if __name__ == "__main__":
    # Example usage and testing
    print("TrajectoryCollector module loaded successfully!")
    
    # Test episode result creation
    test_episode = EpisodeResult(
        task_id="test_001",
        trajectory=[{
            "turn": 1,
            "action": "I need to analyze this task.",
            "observation": "Task analysis complete.",
            "reward": 1.0,
            "done": True,
            "metadata": {}
        }],
        total_reward=1.0,
        turns=1,
        initial_prompt=[{"role": "user", "content": "Test task"}],
        success=True,
    )
    
    print(f"Test episode: {test_episode.task_id}, valid: {test_episode.is_valid()}")