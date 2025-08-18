#!/usr/bin/env python3
"""
MCPToolEnvironment: SkyRL Environment for Multi-Turn Tool Use
============================================================

A text-based environment for training RL agents on multi-turn tool-calling tasks
using MCP (Model Context Protocol) servers.

Key Features:
- Multi-turn conversation support with persistent context
- Real tool execution via MCP servers
- Enhanced reward system with intermediate progress rewards
- State-based and response-based evaluation
- Curriculum learning support
- Error injection for robustness training

Author: SkyRL Tool Agent Team
Date: 2024-01-29
"""

import os
import re
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import sys
from pathlib import Path

import logging
# Add SkyRL paths (using local installation)
project_root = Path(__file__).parent.parent
skyrl_gym_path = str(project_root / "SkyRL" / "skyrl-gym")
if skyrl_gym_path not in sys.path:
    sys.path.insert(0, skyrl_gym_path)

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, MessageType, ConversationType
SKYRL_AVAILABLE = True


# SkyRL imports
# try:
#     # Try to import from the actual SkyRL installation
#     import sys
#     from pathlib import Path
    
#     # Add SkyRL paths
#     skyrl_gym_path = str(Path.home() / "Desktop/ongoing_projects/cooking_time_rl/SkyRL/skyrl-gym")
#     if skyrl_gym_path not in sys.path:
#         sys.path.insert(0, skyrl_gym_path)
    
#     from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, MessageType, ConversationType
#     SKYRL_AVAILABLE = True
    
# except ImportError:
#     # Fallback for development without skyrl installed
#     from typing import Dict, List, Any
    
#     MessageType = Dict[str, str]
#     ConversationType = List[MessageType]
    
#     class BaseTextEnv:
#         def __init__(self):
#             self.turns = 0
#             self.max_turns = 1
#             self.tool_groups = []
#             self.tool_to_toolgroup = {}
        
#         def step(self, action: str):
#             raise NotImplementedError
        
#         def init_tool_groups(self, tool_groups=[]):
#             self.tool_groups = tool_groups
#             self.tool_to_toolgroup = {}
    
#     class BaseTextEnvStepOutput:
#         def __init__(self, observations, reward, done, metadata, postprocessed_action=None):
#             self.observations = observations
#             self.reward = reward
#             self.done = done
#             self.metadata = metadata
#             self.postprocessed_action = postprocessed_action
    
#     SKYRL_AVAILABLE = False

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)


@dataclass
class EnvironmentState:
    """Tracks the current state of the environment"""
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    tool_calls_made: List[Dict[str, Any]] = field(default_factory=list)
    turn_count: int = 0
    task_completed: bool = False
    error_count: int = 0
    reasoning_blocks: List[str] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    system_state_changes: List[Dict[str, Any]] = field(default_factory=list)
    total_reward: float = 0.0


@dataclass
class RewardComponents:
    """Breakdown of reward components for transparency"""
    task_completion: float = 0.0
    reasoning_quality: float = 0.0
    tool_efficiency: float = 0.0
    error_recovery: float = 0.0
    intermediate_progress: float = 0.0
    state_correctness: float = 0.0
    response_correctness: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate weighted total reward"""
        weights = {
            'task_completion': 0.4,    # Increased - completion is most important
            'reasoning_quality': 0.15,
            'tool_efficiency': 0.15,
            'error_recovery': 0.1,
            'intermediate_progress': 0.05,  # Reduced to avoid constant rewards
            'state_correctness': 0.1,
            'response_correctness': 0.05
        }
        
        total = 0.0
        for component, weight in weights.items():
            total += getattr(self, component) * weight
        
        # Add penalty for forced actions (set externally)
        forced_penalty = getattr(self, 'forced_action_penalty', 0.0)
        total -= forced_penalty
        
        return min(1.0, max(0.0, total))


class MCPToolEnvironment(BaseTextEnv):
    """
    SkyRL environment for multi-turn tool use with MCP servers
    """
    
    def __init__(self, task_data: Dict[str, Any]):
        """
        Initialize the environment with task data
        
        Args:
            task_data: Dictionary containing task specification in SkyRL format
        """
        # Initialize parent BaseTextEnv
        super().__init__()
        
        # Task specification
        self.task_id = task_data.get("task_metadata", {}).get("task_id", "unknown")
        self.user_query = self._extract_user_query(task_data)
        self.complexity = task_data.get("task_metadata", {}).get("complexity", 
                                       task_data.get("extra_info", {}).get("complexity", "medium"))
        self.category = task_data.get("task_metadata", {}).get("category", "general")
        self.ground_truth = task_data.get("reward_spec", {}).get("ground_truth", {})
        
        # Extract success criteria
        self.success_criteria = self.ground_truth.get("success_criteria", {})
        self.expected_tools = self.success_criteria.get("expected_tools", [])
        self.min_reasoning_score = self.success_criteria.get("min_reasoning_score", 60)
        self.min_tool_calls = self.success_criteria.get("min_tool_calls", 1)
        
        # Environment configuration
        self.max_turns = self._get_max_turns_by_complexity()
        self.state = EnvironmentState()
        self.reward_components = RewardComponents()
        
        # Tool management (will be initialized asynchronously)
        self.tool_manager = None
        self.available_tools = []
        
        # Evaluation tracking
        self.state_evaluator = StateBasedEvaluator(self.ground_truth)
        self.response_evaluator = ResponseBasedEvaluator(self.ground_truth)
        
        # Error injection configuration - DISABLED by default for deterministic training
        self.error_injection_enabled = task_data.get("extra_info", {}).get("enable_errors", False)
        self.error_injection_rate = 0.2 if self.complexity == "hard" else 0.1
        
        logger.info(f"Initialized MCPToolEnvironment for task {self.task_id} (complexity: {self.complexity})")
    
    def _extract_user_query(self, task_data: Dict[str, Any]) -> str:
        """Extract user query from task data"""
        prompt = task_data.get("prompt", [])
        for msg in prompt:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    
    def _get_max_turns_by_complexity(self) -> int:
        """Get maximum turns based on task complexity"""
        complexity_limits = {
            "easy": 5,     # Reduced for faster training
            "medium": 8,   # Reduced for faster training
            "hard": 10     # Reduced for faster training
        }
        return complexity_limits.get(self.complexity, 6)
    
    async def initialize_tools(self):
        """Initialize MCP tool manager asynchronously"""
        from .simple_shared_manager import SimpleSharedMCPToolManager
        self.tool_manager = SimpleSharedMCPToolManager()
        await self.tool_manager.initialize()
        self.available_tools = list(self.tool_manager.available_tools.keys())
        logger.info(f"Initialized {len(self.available_tools)} tools")
    
    def step(self, action: str) -> BaseTextEnvStepOutput:
        """
        Process agent action and return environment response
        
        Args:
            action: Agent's response containing reasoning and/or tool calls
            
        Returns:
            BaseTextEnvStepOutput with observations, reward, done flag, and metadata
        """
        # Increment turn count
        self.state.turn_count += 1
        
        # Parse action
        reasoning_blocks, tool_calls, natural_response = self._parse_action(action)
        
        # Update conversation history
        self.state.conversation_history.append({
            "role": "assistant",
            "content": action
        })
        
        # Process reasoning blocks
        if reasoning_blocks:
            self.state.reasoning_blocks.extend(reasoning_blocks)
            self._evaluate_reasoning_quality(reasoning_blocks)
        
        # Execute tool calls and collect observations
        observations = []
        tool_results = []
        
        for tool_call in tool_calls:
            result = self._execute_tool_call(tool_call)
            tool_results.append(result)
            observations.append(self._format_tool_response(result))
            
            # Track tool usage
            self.state.tool_calls_made.append({
                "turn": self.state.turn_count,
                "tool": tool_call.get("name"),
                "parameters": tool_call.get("arguments"),
                "result": result
            })
        
        # Compute intermediate rewards
        self._compute_intermediate_rewards()
        
        # Check if task is completed
        task_completed = self._check_task_completion(natural_response, tool_results)
        
        # Determine if episode is done
        done = (
            task_completed or 
            self.state.turn_count >= self.max_turns or
            self._check_critical_failure()
        )
        
        # Final reward computation if episode is done
        if done:
            self._compute_final_rewards()
        
        # Update total reward in state
        self.state.total_reward += self.reward_components.total
        
        # Format observations
        observation_text = self._format_observations(observations, done)
        
        # Prepare metadata
        metadata = {
            "turn_count": self.state.turn_count,
            "tool_calls_this_turn": len(tool_calls),
            "total_tool_calls": len(self.state.tool_calls_made),
            "reasoning_blocks": len(reasoning_blocks),
            "reward_breakdown": self._get_reward_breakdown(),
            "task_completed": task_completed
        }
        
        # Format observations as OpenAI API messages for SkyRL
        if SKYRL_AVAILABLE:
            # Convert to OpenAI message format required by SkyRL
            skyrl_observations = []
            
            # Add system message if needed
            if self.state.turn_count == 1:
                skyrl_observations.append({
                    "role": "system",
                    "content": f"You are helping with task: {self.user_query}"
                })
            
            # Add the current turn observation
            skyrl_observations.append({
                "role": "user" if self.state.turn_count % 2 == 1 else "assistant",
                "content": observation_text
            })
            
            return {
                "observations": skyrl_observations,
                "reward": self.reward_components.total,
                "done": done,
                "metadata": metadata,
                "postprocessed_action": action
            }
        else:
            # Fallback for non-SkyRL usage
            return BaseTextEnvStepOutput(
                observations=observation_text,
                reward=self.reward_components.total,
                done=done,
                metadata=metadata
            )
    
    def _parse_action(self, action: str) -> Tuple[List[str], List[Dict], str]:
        """
        Parse agent action into components
        
        Returns:
            Tuple of (reasoning_blocks, tool_calls, natural_response)
        """
        # Extract reasoning blocks
        reasoning_pattern = r'<think>(.*?)</think>'
        reasoning_blocks = re.findall(reasoning_pattern, action, re.DOTALL)
        
        # Extract tool calls
        tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
        tool_call_texts = re.findall(tool_call_pattern, action, re.DOTALL)
        
        tool_calls = []
        for tool_call_text in tool_call_texts:
            try:
                tool_call = json.loads(tool_call_text.strip())
                # CRITICAL FIX: Normalize tool call arguments to handle common aliases
                tool_call = self._normalize_tool_call_args(tool_call)
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call: {tool_call_text}")
        
        # Extract natural response (everything not in special tags)
        natural_response = re.sub(reasoning_pattern, '', action)
        natural_response = re.sub(tool_call_pattern, '', natural_response)
        natural_response = natural_response.strip()
        
        return reasoning_blocks, tool_calls, natural_response
    
    def _normalize_tool_call_args(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        CRITICAL FIX: Normalize tool call arguments to handle common aliases
        This fixes the "No tool calls found" issue by converting near-miss calls into valid ones
        """
        if "arguments" not in tool_call:
            return tool_call
            
        tool_name = tool_call.get("name", "")
        args = tool_call["arguments"]
        
        # Define argument aliases for common tools
        arg_aliases = {
            "execute_python": {
                "python_code": "code",
                "script": "code", 
                "python": "code"
            },
            "tavily_search": {
                "search_query": "query",
                "q": "query",
                "search": "query"
            },
            "polygon_get_aggs": {
                "ticker": "symbol",
                "stock": "symbol"
            },
            "fmp_get_quote": {
                "ticker": "symbol",
                "stock": "symbol"
            },
            "send_slack_message": {
                "msg": "message",
                "text": "message"
            }
        }
        
        if tool_name in arg_aliases:
            aliases = arg_aliases[tool_name]
            normalized_args = {}
            
            for arg_key, arg_value in args.items():
                # Check if this argument key has an alias mapping
                normalized_key = aliases.get(arg_key, arg_key)
                normalized_args[normalized_key] = arg_value
            
            tool_call["arguments"] = normalized_args
            
        return tool_call
    
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call via MCP
        
        Args:
            tool_call: Dictionary with 'name' and 'arguments'
            
        Returns:
            Result dictionary with 'success', 'content', and optional 'error'
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        # Check if tool exists
        if not self.tool_manager or tool_name not in self.available_tools:
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "content": None
            }
        
        # Error injection for robustness training
        if self.error_injection_enabled and self._should_inject_error():
            return self._inject_tool_error(tool_name)
        
        # Execute tool (synchronously for now)
        try:
            # Note: In real implementation, this would be async
            result = self.tool_manager.execute_tool_sync(tool_name, arguments)
            
            # Track state changes if applicable
            if self._modifies_state(tool_name):
                self.state.system_state_changes.append({
                    "turn": self.state.turn_count,
                    "tool": tool_name,
                    "change_type": "write",
                    "details": result
                })
            
            return {
                "success": True,
                "content": result,
                "error": None
            }
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def _should_inject_error(self) -> bool:
        """Determine if error should be injected based on configuration"""
        # Deterministic: use turn count instead of random
        if not self.error_injection_enabled:
            return False
        # Inject error on specific turns for deterministic behavior
        return (self.state.turn_count % 10) == 7  # Error on turn 7, 17, 27, etc.
    
    def _inject_tool_error(self, tool_name: str) -> Dict[str, Any]:
        """Inject a realistic error for robustness training"""
        error_types = [
            ("Network timeout", "Request timed out after 30 seconds"),
            ("Rate limit exceeded", "API rate limit reached. Please try again later."),
            ("Service unavailable", "503 Service Temporarily Unavailable"),
            ("Invalid response", "Received malformed response from API")
        ]
        
        # Deterministic: use turn count to select error type
        error_index = self.state.turn_count % len(error_types)
        error_type, error_message = error_types[error_index]
        logger.info(f"Injecting error for {tool_name}: {error_type}")
        
        return {
            "success": False,
            "error": error_message,
            "content": None
        }
    
    def _modifies_state(self, tool_name: str) -> bool:
        """Check if tool modifies system state"""
        state_modifying_tools = [
            "send_slack_message",
            "execute_python",
            "create_file",
            "delete_file",
            "update_watchlist"
        ]
        return tool_name in state_modifying_tools
    
    def _format_tool_response(self, result: Dict[str, Any]) -> str:
        """Format tool execution result for observation"""
        if result["success"]:
            content = result["content"]
            # Truncate very long responses
            if isinstance(content, str) and len(content) > 1000:
                content = content[:1000] + "... (truncated)"
            return f"<tool_response>\n{content}\n</tool_response>"
        else:
            return f"<tool_error>\n{result['error']}\n</tool_error>"
    
    def _evaluate_reasoning_quality(self, reasoning_blocks: List[str]):
        """Evaluate the quality of reasoning blocks"""
        total_score = 0.0
        
        for block in reasoning_blocks:
            # Check for key reasoning indicators
            score = 0.0
            
            # Problem analysis
            if any(phrase in block.lower() for phrase in ['need to', 'task requires', 'user wants']):
                score += 0.2
            
            # Tool selection rationale
            if any(phrase in block.lower() for phrase in ['use', 'tool', 'because', 'since']):
                score += 0.2
            
            # Planning steps
            if any(phrase in block.lower() for phrase in ['first', 'then', 'next', 'finally']):
                score += 0.2
            
            # Error consideration
            if any(phrase in block.lower() for phrase in ['if', 'error', 'fail', 'alternative']):
                score += 0.2
            
            # Clear structure
            if len(block.split('\n')) > 2:
                score += 0.2
            
            total_score += score
        
        # Normalize by number of blocks
        if reasoning_blocks:
            avg_score = total_score / len(reasoning_blocks)
            self.reward_components.reasoning_quality = avg_score
    
    def _compute_intermediate_rewards(self):
        """
        Compute intermediate rewards for progress towards goal
        
        This addresses reward signal sparsity by providing feedback
        even when the task is not yet complete.
        """
        progress_score = 0.0
        
        # Reward for using expected tools
        if self.expected_tools:
            tools_used = [tc["tool"] for tc in self.state.tool_calls_made]
            expected_used = sum(1 for tool in self.expected_tools if tool in tools_used)
            progress_score += 0.3 * (expected_used / len(self.expected_tools))
        
        # Reward for successful tool calls
        successful_calls = sum(1 for tc in self.state.tool_calls_made 
                              if tc["result"].get("success", False))
        if self.state.tool_calls_made:
            progress_score += 0.3 * (successful_calls / len(self.state.tool_calls_made))
        
        # Reward for reasoning depth
        if self.state.reasoning_blocks:
            progress_score += 0.2 * min(1.0, len(self.state.reasoning_blocks) / 3)
        
        # Reward for error recovery
        if self.state.error_count > 0:
            recovery_attempts = sum(1 for tc in self.state.tool_calls_made 
                                   if "retry" in str(tc).lower() or "alternative" in str(tc).lower())
            if recovery_attempts > 0:
                progress_score += 0.2
        
        self.reward_components.intermediate_progress = progress_score
    
    def _check_task_completion(self, natural_response: str, tool_results: List[Dict]) -> bool:
        """
        Check if the task has been completed successfully
        
        Uses both state-based and response-based evaluation
        """
        # Basic completion check
        if not natural_response and not tool_results:
            return False
        
        # State-based evaluation
        state_correct = self.state_evaluator.evaluate(
            self.state.system_state_changes,
            self.state.turn_count
        )
        self.reward_components.state_correctness = 1.0 if state_correct else 0.0
        
        # Response-based evaluation
        response_correct = self.response_evaluator.evaluate(
            self.state.tool_calls_made,
            tool_results,
            natural_response
        )
        self.reward_components.response_correctness = 1.0 if response_correct else 0.0
        
        # Task is complete if both evaluations pass
        task_completed = state_correct and response_correct
        
        # Also check if minimum requirements are met
        if task_completed:
            task_completed = (
                len(self.state.tool_calls_made) >= self.min_tool_calls and
                self.reward_components.reasoning_quality >= (self.min_reasoning_score / 100)
            )
        
        self.state.task_completed = task_completed
        return task_completed
    
    def _check_critical_failure(self) -> bool:
        """Check if a critical failure has occurred"""
        # Too many consecutive errors
        if self.state.error_count >= 3:
            return True
        
        # Agent explicitly gives up
        last_message = self.state.conversation_history[-1]["content"] if self.state.conversation_history else ""
        if any(phrase in last_message.lower() for phrase in ["i give up", "cannot complete", "impossible"]):
            return True
        
        return False
    
    def _compute_final_rewards(self):
        """Compute final rewards when episode ends"""
        # Task completion reward
        self.reward_components.task_completion = 1.0 if self.state.task_completed else 0.0
        
        # Tool efficiency reward
        if self.expected_tools and self.state.tool_calls_made:
            # Penalize unnecessary tool calls
            expected_count = len(self.expected_tools)
            actual_count = len(self.state.tool_calls_made)
            efficiency = 1.0 - (abs(actual_count - expected_count) / max(expected_count, actual_count))
            self.reward_components.tool_efficiency = max(0.0, efficiency)
        
        # Error recovery reward
        if self.state.error_count > 0:
            # Check if errors were recovered from
            final_success = self.state.task_completed
            recovery_demonstrated = any("retry" in str(tc) or "alternative" in str(tc) 
                                       for tc in self.state.tool_calls_made)
            if final_success and recovery_demonstrated:
                self.reward_components.error_recovery = 1.0
            elif recovery_demonstrated:
                self.reward_components.error_recovery = 0.5
    
    def _format_observations(self, observations: List[str], done: bool) -> str:
        """Format observations for the agent"""
        obs_parts = []
        
        # Add tool responses
        obs_parts.extend(observations)
        
        # Add turn counter
        obs_parts.append(f"\n[Turn {self.state.turn_count}/{self.max_turns}]")
        
        # Add completion message if done
        if done:
            if self.state.task_completed:
                obs_parts.append("\n✅ Task completed successfully!")
            elif self.state.turn_count >= self.max_turns:
                obs_parts.append("\n⏱️ Maximum turns reached.")
            else:
                obs_parts.append("\n❌ Task failed or terminated.")
        
        return "\n".join(obs_parts)
    
    def _get_reward_breakdown(self) -> Dict[str, float]:
        """Get detailed reward breakdown for transparency"""
        return {
            "task_completion": self.reward_components.task_completion,
            "reasoning_quality": self.reward_components.reasoning_quality,
            "tool_efficiency": self.reward_components.tool_efficiency,
            "error_recovery": self.reward_components.error_recovery,
            "intermediate_progress": self.reward_components.intermediate_progress,
            "state_correctness": self.reward_components.state_correctness,
            "response_correctness": self.reward_components.response_correctness,
            "total": self.reward_components.total
        }
    
    def reset(self):
        """Reset environment to initial state"""
        self.state = EnvironmentState()
        self.reward_components = RewardComponents()
        logger.info(f"Environment reset for task {self.task_id}")


class StateBasedEvaluator:
    """
    Evaluates correctness based on system state changes
    """
    
    def __init__(self, ground_truth: Dict[str, Any]):
        self.expected_state = ground_truth.get("expected_state", {})
        self.required_changes = ground_truth.get("required_state_changes", [])
    
    def evaluate(self, state_changes: List[Dict], turn: int) -> bool:
        """
        Check if system state matches expectations
        
        Returns True if state is correct, False otherwise
        """
        if not self.required_changes:
            # No state requirements defined - this should be stricter
            # Only return True if some state changes were actually made
            return len(state_changes) > 0
        
        # Check if all required changes were made
        for required_change in self.required_changes:
            if not self._find_matching_change(required_change, state_changes):
                return False
        
        return True
    
    def _find_matching_change(self, required: Dict, actual_changes: List[Dict]) -> bool:
        """Find if a required state change was made"""
        for change in actual_changes:
            if (change.get("tool") == required.get("tool") and
                change.get("change_type") == required.get("change_type")):
                # Additional validation could go here
                return True
        return False


class ResponseBasedEvaluator:
    """
    Evaluates correctness based on execution path and responses
    """
    
    def __init__(self, ground_truth: Dict[str, Any]):
        self.minimal_execution_path = ground_truth.get("minimal_execution_path", [])
        self.required_outputs = ground_truth.get("required_outputs", [])
    
    def evaluate(self, tool_calls: List[Dict], results: List[Dict], response: str) -> bool:
        """
        Check if execution path and outputs match requirements
        
        Returns True if response is correct, False otherwise
        """
        # If no requirements defined, be strict - require actual tool usage for correctness
        if not self.minimal_execution_path and not self.required_outputs:
            # No specific requirements, but should require at least one successful tool call
            # for response correctness (otherwise it's just natural language)
            return len(tool_calls) > 0 and any(r.get("success", False) for r in results if r)
        
        # Check minimal execution path
        if self.minimal_execution_path:
            executed_tools = [tc["tool"] for tc in tool_calls]
            for required_tool in self.minimal_execution_path:
                if required_tool not in executed_tools:
                    return False
        
        # Check required outputs in response
        if self.required_outputs:
            for required_output in self.required_outputs:
                if required_output.lower() not in response.lower():
                    return False
        
        return True


# Environment registration function for SkyRL
def make_mcp_tool_env(task_data: Dict[str, Any]) -> MCPToolEnvironment:
    """Factory function to create MCPToolEnvironment instances"""
    return MCPToolEnvironment(task_data)

# Register environment with SkyRL if available
if SKYRL_AVAILABLE:
    try:
        from skyrl_gym.envs.registration import register, registry
        
        # IDEMPOTENT GUARD: Check if already registered
        if "mcp_tool_env" not in registry:
            register(
                id="mcp_tool_env",
                entry_point="environments.mcp_tool_environment:make_mcp_tool_env"
            )
            logger.info("✅ MCPToolEnvironment registered with SkyRL")
        else:
            logger.debug("🔄 MCPToolEnvironment already registered with SkyRL")
        
    except ImportError:
        logger.debug("Could not register environment with SkyRL registry (SkyRL not available)")
        
    except Exception as e:
        # Silently handle already registered case
        if "already registered" in str(e):
            logger.debug(f"📝 Environment already registered: {e}")
        else:
            logger.warning(f"Environment registration failed: {e}")
else:
    logger.info("SkyRL not available - environment running in standalone mode")