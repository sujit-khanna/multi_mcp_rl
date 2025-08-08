#!/usr/bin/env python3
"""
Policy-Integrated MCPToolEnvironment: SkyRL Environment with GPT-4o-mini Policy
==============================================================================

Extends MCPToolEnvironment to include an integrated policy model (GPT-4o-mini) 
that generates realistic agent actions for RL training.

Key Features:
- GPT-4o-mini as the action-generating policy
- Multi-turn conversation with policy-generated actions
- Real tool execution via MCP servers
- Curriculum learning with policy guidance
- Error injection and recovery training

Author: SkyRL Tool Agent Team
Date: 2024-01-29
"""

import os
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import openai

# Import base environment
from .mcp_tool_environment import (
    MCPToolEnvironment, 
    SKYRL_AVAILABLE,
    BaseTextEnvStepOutput
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for the integrated policy model"""
    model_name: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    enable_reasoning: bool = True
    enable_tool_calling: bool = True
    use_openrouter: bool = False
    openrouter_api_key: Optional[str] = None
    openrouter_site_url: Optional[str] = None
    openrouter_site_name: Optional[str] = None
    system_prompt_template: str = """You are a deep-thinking AI assistant that demonstrates high-quality step-by-step reasoning before and after using tools, including intelligent error handling and recovery strategies.

Available tools: {tools}
Current task: {task}

CRITICAL REASONING REQUIREMENTS:
1. Every response must start with <think> tags containing detailed analysis (100+ words)
2. Break down the user's request into specific components
3. Explain WHY you choose specific tools and their order
4. Plan your approach with clear dependencies and sequencing
5. After tool responses, analyze results and plan next steps
6. **DEMONSTRATE ERROR RECOVERY** when tools fail or return errors

ERROR HANDLING REQUIREMENTS:
- When a tool fails or returns an error, analyze WHY it failed
- Consider alternative tools or approaches
- Show strategic thinking about backup plans
- Demonstrate graceful degradation when some data is unavailable
- Document your recovery strategy clearly

GOOD REASONING EXAMPLE:
<think>
The user wants me to "find AAPL stock price and recent AI news."

Analysis of request:
1. Current stock price for Apple (AAPL) - requires real-time financial data
2. Recent news about Apple's AI initiatives - requires current web search

Tool selection rationale:
- For stock price: I'll use fmp_get_quote because it provides real-time quotes with detailed metrics
- For AI news: tavily_search is optimal for recent web content with good summarization

Approach strategy:
1. First get AAPL quote to establish current market context
2. Then search for "Apple AI news" to find recent developments
3. Finally synthesize both pieces of information for a comprehensive response

These can be executed sequentially since the stock context might inform how I interpret the news.
</think>

Your reasoning must be:
- Specific and analytical (not generic)
- At least 100 words explaining your thought process
- Include why you chose specific tools
- Show clear planning and sequencing
- Demonstrate understanding of the request
- **Show error recovery strategies when tools fail**

TOOL CALL FORMAT:
To use a tool, you MUST use this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

IMPORTANT: After your <think> block, you MUST make actual tool calls using the format above.

BAD REASONING EXAMPLES (NEVER DO THIS):
❌ <think>I need to get information about AAPL.</think>
❌ <think>Making tool calls to gather information.</think>  
❌ <think>Let me search for this data.</think>
❌ <think>Using tools to help the user.</think>"""


class PolicyIntegratedEnvironment(MCPToolEnvironment):
    """
    MCPToolEnvironment with integrated GPT-4o-mini policy for action generation
    """
    
    def __init__(self, task_data: Dict[str, Any], policy_config: Optional[PolicyConfig] = None):
        """
        Initialize the policy-integrated environment
        
        Args:
            task_data: Dictionary containing task specification in SkyRL format
            policy_config: Configuration for the policy model
        """
        # Initialize parent environment
        super().__init__(task_data)
        
        # Policy configuration
        self.policy_config = policy_config or PolicyConfig()
        
        # Set up API client
        if self.policy_config.use_openrouter:
            # Use OpenRouter
            openrouter_key = self.policy_config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
            if not openrouter_key:
                raise ValueError("OpenRouter API key is required. Set OPENROUTER_API_KEY environment variable or provide in policy_config.")
            
            self.openai_client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=openrouter_key
            )
            # Only include headers if values are provided
            self.extra_headers = {}
            if self.policy_config.openrouter_site_url:
                self.extra_headers["HTTP-Referer"] = self.policy_config.openrouter_site_url
            if self.policy_config.openrouter_site_name:
                self.extra_headers["X-Title"] = self.policy_config.openrouter_site_name
            
            logger.info(f"Using OpenRouter with model: {self.policy_config.model_name}")
        else:
            # Use standard OpenAI
            api_key = self.policy_config.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or provide in policy_config.")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            self.extra_headers = None
        
        # Policy state
        self.policy_conversation: List[Dict[str, str]] = []
        self.last_policy_action: Optional[str] = None
        
        logger.info(f"Initialized PolicyIntegratedEnvironment with {self.policy_config.model_name}")
    
    def generate_policy_action(self, observation: str = "") -> str:
        """
        Generate an action using the integrated GPT-4o-mini policy
        
        Args:
            observation: Current observation from the environment
            
        Returns:
            Generated action string with reasoning and/or tool calls
        """
        try:
            # Build conversation context
            messages = self._build_policy_messages(observation)
            
            # Generate response using OpenAI API
            if self.policy_config.use_openrouter and self.extra_headers:
                response = self.openai_client.chat.completions.create(
                    model=self.policy_config.model_name,
                    messages=messages,
                    temperature=self.policy_config.temperature,
                    max_tokens=self.policy_config.max_tokens,
                    extra_headers=self.extra_headers
                )
            else:
                response = self.openai_client.chat.completions.create(
                    model=self.policy_config.model_name,
                    messages=messages,
                    temperature=self.policy_config.temperature,
                    max_tokens=self.policy_config.max_tokens
                )
            
            action = response.choices[0].message.content
            self.last_policy_action = action
            
            # Update policy conversation
            if observation:
                self.policy_conversation.append({"role": "user", "content": observation})
            self.policy_conversation.append({"role": "assistant", "content": action})
            
            logger.debug(f"Generated policy action: {action[:100]}...")
            return action
            
        except Exception as e:
            logger.error(f"Policy action generation failed: {e}")
            # Fallback to simple action
            return self._generate_fallback_action()
    
    def _build_policy_messages(self, observation: str) -> List[Dict[str, str]]:
        """Build message context for policy model"""
        messages = []
        
        # System prompt
        available_tools = list(self.available_tools.keys()) if hasattr(self, 'available_tools') else []
        system_prompt = self.policy_config.system_prompt_template.format(
            tools=available_tools,
            task=self.user_query
        )
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history (last few turns to avoid context limit)
        recent_conversation = self.policy_conversation[-6:] if len(self.policy_conversation) > 6 else self.policy_conversation
        messages.extend(recent_conversation)
        
        # Current observation
        if observation:
            messages.append({"role": "user", "content": observation})
        elif self.state.turn_count == 0:
            # Initial turn - present the task
            messages.append({"role": "user", "content": self.user_query})
        
        return messages
    
    def _generate_fallback_action(self) -> str:
        """Generate a ReAct-style fallback action when policy fails"""
        if self.expected_tools and self.state.turn_count == 1:
            tool_name = self.expected_tools[0]
            return f"""<think>
I need to approach this task systematically. Looking at the user's request: {self.user_query[:100]}...

Task Analysis:
1. The request requires me to gather specific information
2. I have access to {len(self.expected_tools)} relevant tools: {', '.join(self.expected_tools)}
3. The most appropriate starting point is {tool_name} as it directly addresses the core information need

Strategy:
- Start with {tool_name} to establish baseline data
- Analyze the results to determine if additional tools are needed
- Provide comprehensive response based on gathered information

This approach ensures systematic data gathering while maintaining efficiency.
</think>

I'll help you with this task. Let me start by gathering the necessary information using the most appropriate tool for your request.

<tool_call>
{{"name": "{tool_name}", "arguments": {{"query": "{self.user_query[:50]}..." if len(self.user_query) > 50 else self.user_query}}}}
</tool_call>"""
        else:
            return """<think>
I need to understand what the user is asking for and how I can best help them. Let me analyze their request and determine the most appropriate approach to provide valuable assistance.

Without specific tool requirements, I should focus on providing thoughtful analysis and clear communication about how I can help.
</think>

I understand your request. Let me think through this systematically and provide you with the most helpful response possible based on the information available."""
    
    def step_with_policy(self, observation: str = "") -> BaseTextEnvStepOutput:
        """
        Take a step using the integrated policy to generate actions
        
        Args:
            observation: Optional observation to provide to policy
            
        Returns:
            BaseTextEnvStepOutput with policy-generated action results
        """
        # Generate action using policy
        action = self.generate_policy_action(observation)
        
        # Execute the action in the environment
        return self.step(action)
    
    def run_policy_episode(self, max_turns: Optional[int] = None) -> Dict[str, Any]:
        """
        Run a complete episode using the integrated policy
        
        Args:
            max_turns: Override max turns for this episode
            
        Returns:
            Episode results with trajectory and metrics
        """
        if max_turns:
            original_max_turns = self.max_turns
            self.max_turns = max_turns
        
        episode_trajectory = []
        episode_reward = 0.0
        observation = ""
        
        logger.info(f"Starting policy episode for task: {self.task_id}")
        
        try:
            while not self._is_episode_done():
                # Generate and execute action
                output = self.step_with_policy(observation)
                
                # Extract observation for next turn
                if SKYRL_AVAILABLE and isinstance(output, dict):
                    observation_messages = output.get("observations", [])
                    observation = observation_messages[-1].get("content", "") if observation_messages else ""
                    reward = output.get("reward", 0.0)
                    done = output.get("done", False)
                    metadata = output.get("metadata", {})
                else:
                    observation = output.observations if hasattr(output, 'observations') else ""
                    reward = output.reward if hasattr(output, 'reward') else 0.0
                    done = output.done if hasattr(output, 'done') else False
                    metadata = output.metadata if hasattr(output, 'metadata') else {}
                
                # Record trajectory step
                step_data = {
                    "turn": self.state.turn_count,
                    "action": self.last_policy_action,
                    "observation": observation,
                    "reward": reward,
                    "done": done,
                    "metadata": metadata
                }
                episode_trajectory.append(step_data)
                episode_reward += reward
                
                if done:
                    break
            
            # Episode results
            episode_results = {
                "task_id": self.task_id,
                "trajectory": episode_trajectory,
                "total_reward": episode_reward,
                "turns": self.state.turn_count,
                "task_completed": self.state.task_completed,
                "tools_used": len(self.state.tool_calls_made),
                "expected_tools": self.expected_tools,
                "success_rate": 1.0 if self.state.task_completed else 0.0,
                "reward_breakdown": self._get_reward_breakdown()
            }
            
            logger.info(f"Policy episode completed: {self.state.turn_count} turns, "
                       f"reward: {episode_reward:.3f}, success: {self.state.task_completed}")
            
            return episode_results
            
        except Exception as e:
            logger.error(f"Policy episode failed: {e}")
            return {
                "task_id": self.task_id,
                "error": str(e),
                "trajectory": episode_trajectory,
                "total_reward": episode_reward,
                "turns": self.state.turn_count,
                "task_completed": False
            }
        
        finally:
            # Restore original max turns if modified
            if max_turns:
                self.max_turns = original_max_turns
    
    def _is_episode_done(self) -> bool:
        """Check if the current episode should end"""
        return (
            self.state.task_completed or 
            self.state.turn_count >= self.max_turns or
            self._check_critical_failure()
        )
    
    def run_batch_episodes(self, task_batch: List[Dict[str, Any]], 
                          episodes_per_task: int = 1) -> List[Dict[str, Any]]:
        """
        Run multiple episodes across a batch of tasks
        
        Args:
            task_batch: List of task data dictionaries
            episodes_per_task: Number of episodes to run per task
            
        Returns:
            List of episode results
        """
        all_results = []
        
        for task_data in task_batch:
            for episode_num in range(episodes_per_task):
                # Create new environment instance for each episode
                env = PolicyIntegratedEnvironment(task_data, self.policy_config)
                
                # Initialize tools if available
                if hasattr(self, 'tool_manager') and self.tool_manager:
                    env.tool_manager = self.tool_manager
                    env.available_tools = self.available_tools
                
                # Run episode
                results = env.run_policy_episode()
                results["episode_num"] = episode_num
                all_results.append(results)
        
        return all_results
    
    def reset(self):
        """Reset environment and policy state"""
        super().reset()
        self.policy_conversation = []
        self.last_policy_action = None
        logger.info(f"PolicyIntegratedEnvironment reset for task {self.task_id}")


# Factory function for SkyRL integration
def make_policy_integrated_env(task_data: Dict[str, Any], 
                              policy_config: Optional[PolicyConfig] = None) -> PolicyIntegratedEnvironment:
    """Factory function to create PolicyIntegratedEnvironment instances"""
    return PolicyIntegratedEnvironment(task_data, policy_config)


# Register with SkyRL if available
if SKYRL_AVAILABLE:
    try:
        # Try to register with SkyRL - handle potential API changes gracefully
        import sys
        from pathlib import Path
        skyrl_gym_path = str(Path.home() / "Desktop/ongoing_projects/cooking_time_rl/SkyRL/skyrl-gym")
        if skyrl_gym_path in sys.path:
            from skyrl_gym.envs import registration
            if hasattr(registration, 'register'):
                try:
                    registration.register(
                        id="policy_integrated_mcp_env",
                        entry_point="environments.policy_integrated_environment:make_policy_integrated_env"
                    )
                    logger.info("✅ PolicyIntegratedEnvironment registered with SkyRL")
                except TypeError:
                    # Handle case where register doesn't accept description parameter
                    logger.info("✅ PolicyIntegratedEnvironment registration attempted (API differences handled)")
        
    except ImportError:
        logger.info("SkyRL registration not available - PolicyIntegratedEnvironment running in standalone mode")
        
    except Exception as e:
        logger.info(f"SkyRL registration note: {e}")
else:
    logger.info("SkyRL not available - PolicyIntegratedEnvironment running in standalone mode")