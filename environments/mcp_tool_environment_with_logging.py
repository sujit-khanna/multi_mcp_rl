#!/usr/bin/env python3
"""
MCPToolEnvironment with Enhanced Logging
=======================================

Extended version of MCPToolEnvironment that logs all tool calls
and their responses to verify real execution is happening.
"""

from mcp_tool_environment import MCPToolEnvironment as BaseMCPToolEnvironment
import logging
import json
from typing import Dict, Any, List, Tuple

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Prompt template to ensure agents always emit valid tool calls
TOOL_CALL_PROMPT_TEMPLATE = (
    "You MUST respond with a tool call using the exact syntax: "
    "<tool_call>{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}</tool_call>. "
    "The <tool_call> tag must appear as the first token in your response and no natural language should appear outside these tags."
)


class MCPToolEnvironmentWithLogging(BaseMCPToolEnvironment):
    """MCPToolEnvironment with comprehensive logging of tool execution"""

    def __init__(self, task_data: Dict[str, Any]):
        """Initialize with enhanced logging"""
        super().__init__(task_data)
        # Expose the strict tool call prompt for any integrated policies
        self.tool_call_prompt_template = TOOL_CALL_PROMPT_TEMPLATE
        self.tool_execution_log = []
        logger.info(f"🔧 INITIALIZED ENVIRONMENT for task: {self.task_id}")
        logger.info(f"   User query: {self.user_query[:100]}...")
        logger.info(f"   Expected tools: {self.expected_tools}")
        
    def _execute_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call via MCP with detailed logging
        """
        tool_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})
        
        # Log the tool call attempt
        logger.info(f"\n{'='*60}")
        logger.info(f"🔨 TOOL CALL ATTEMPT #{len(self.state.tool_calls_made) + 1}")
        logger.info(f"   Tool: {tool_name}")
        logger.info(f"   Arguments: {json.dumps(arguments, indent=2)}")
        logger.info(f"{'='*60}")
        
        # Check if tool exists
        if not self.tool_manager:
            logger.error("❌ No tool manager available!")
            return {
                "success": False,
                "error": "No tool manager available",
                "content": None
            }
            
        if tool_name not in self.available_tools:
            logger.error(f"❌ Tool '{tool_name}' not found in available tools: {self.available_tools}")
            return {
                "success": False,
                "error": f"Tool '{tool_name}' not found",
                "content": None
            }
        
        # Error injection for robustness training
        if self.error_injection_enabled and self._should_inject_error():
            result = self._inject_tool_error(tool_name)
            logger.warning(f"⚠️ ERROR INJECTED: {result['error']}")
            return result
        
        # Execute tool
        try:
            logger.info(f"🚀 Executing tool '{tool_name}' via MCP...")
            
            # Log pre-execution state
            logger.debug(f"   Pre-execution state: {len(self.state.tool_calls_made)} tools called so far")
            
            # Actual tool execution
            result_content = self.tool_manager.execute_tool_sync(tool_name, arguments)
            
            # Log the complete raw result
            logger.info(f"✅ TOOL EXECUTION SUCCESSFUL!")
            logger.info(f"   Raw result type: {type(result_content)}")
            logger.info(f"🔍 COMPLETE TOOL RESPONSE:")
            logger.info(f"{'='*80}")
            logger.info(str(result_content))
            logger.info(f"{'='*80}")
            
            # Track state changes if applicable
            if self._modifies_state(tool_name):
                self.state.system_state_changes.append({
                    "turn": self.state.turn_count,
                    "tool": tool_name,
                    "change_type": "write",
                    "details": result_content
                })
                logger.info(f"   📝 State modification recorded for {tool_name}")
            
            # Log execution details
            self.tool_execution_log.append({
                "turn": self.state.turn_count,
                "tool": tool_name,
                "arguments": arguments,
                "success": True,
                "result_preview": str(result_content)[:200]
            })
            
            result = {
                "success": True,
                "content": result_content,
                "error": None
            }
            
            logger.info(f"{'='*60}\n")
            return result
            
        except Exception as e:
            logger.error(f"❌ TOOL EXECUTION FAILED!")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.error(f"{'='*60}\n")
            
            # Log failed execution
            self.tool_execution_log.append({
                "turn": self.state.turn_count,
                "tool": tool_name,
                "arguments": arguments,
                "success": False,
                "error": str(e)
            })
            
            return {
                "success": False,
                "error": str(e),
                "content": None
            }
    
    def step(self, action: str):
        """Override step to add action logging"""
        logger.info(f"\n{'*'*80}")
        logger.info(f"🎯 ENVIRONMENT STEP - Turn {self.state.turn_count + 1}/{self.max_turns}")
        logger.info(f"📝 Agent action preview: {action[:200]}...")
        
        # Parse action to check for tool calls
        reasoning_blocks, tool_calls, natural_response = self._parse_action(action)
        
        if tool_calls:
            logger.info(f"🔧 Found {len(tool_calls)} tool calls in action")
        else:
            logger.info(f"💬 No tool calls found - natural response only")
            
        if reasoning_blocks:
            logger.info(f"🤔 Found {len(reasoning_blocks)} reasoning blocks")
        
        # Check if this action was forced (from policy tracking)
        # This will be set by the policy if it forced a tool call
        forced_action = getattr(self, '_last_action_was_forced', False)
        
        # Apply forced action penalty
        if forced_action and hasattr(self, 'reward_components'):
            self.reward_components.forced_action_penalty = 0.1
            logger.info(f"⚠️ FORCED ACTION PENALTY: -0.1")
        else:
            if hasattr(self, 'reward_components'):
                self.reward_components.forced_action_penalty = 0.0
        
        # Call parent step
        result = super().step(action)
        
        # Log step result
        if isinstance(result, dict):
            reward = result.get('reward', 0)
            done = result.get('done', False)
        else:
            reward = getattr(result, 'reward', 0)
            done = getattr(result, 'done', False)
            
        logger.info(f"📊 Step result: reward={reward:.3f}, done={done}")
        
        # Add detailed reward breakdown logging
        if hasattr(self, 'reward_components'):
            logger.info(f"💰 REWARD BREAKDOWN:")
            logger.info(f"   task_completion: {self.reward_components.task_completion:.3f}")
            logger.info(f"   reasoning_quality: {self.reward_components.reasoning_quality:.3f}")
            logger.info(f"   tool_efficiency: {self.reward_components.tool_efficiency:.3f}")
            logger.info(f"   error_recovery: {self.reward_components.error_recovery:.3f}")
            logger.info(f"   intermediate_progress: {self.reward_components.intermediate_progress:.3f}")
            logger.info(f"   state_correctness: {self.reward_components.state_correctness:.3f}")
            logger.info(f"   response_correctness: {self.reward_components.response_correctness:.3f}")
            logger.info(f"   TOTAL: {self.reward_components.total:.3f}")
        
        logger.info(f"{'*'*80}\n")
        
        return result
    
    def reset(self):
        """Reset environment with logging"""
        super().reset()
        self.tool_execution_log = []
        logger.info(f"🔄 Environment reset for task {self.task_id}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all tool executions"""
        return {
            "task_id": self.task_id,
            "total_turns": self.state.turn_count,
            "total_tool_calls": len(self.state.tool_calls_made),
            "successful_calls": sum(1 for log in self.tool_execution_log if log["success"]),
            "failed_calls": sum(1 for log in self.tool_execution_log if not log["success"]),
            "tool_execution_log": self.tool_execution_log,
            "final_reward": self.state.total_reward
        }


# Factory function for SkyRL compatibility
def make_mcp_tool_env_with_logging(task_data: Dict[str, Any]) -> MCPToolEnvironmentWithLogging:
    """Factory function to create MCPToolEnvironmentWithLogging instances"""
    return MCPToolEnvironmentWithLogging(task_data)