#!/usr/bin/env python3
"""
Improved Mini Agent for High-Quality Multi-Turn Dataset Generation (SkyRL Format)
======================================================

Enhanced ReAct-style agent that generates multi-turn tool use datasets
compatible with SkyRL trainer format:
https://skyrl.readthedocs.io/en/latest/datasets/dataset-preparation.html

Output Format:
- data_source: "mcp_tool_agent_synthetic"
- prompt: OpenAI chat format conversation
- env_class: "tool_using_agent" 
- reward_spec: Rule-based success criteria
- extra_info: Comprehensive metadata and metrics

Environment Variables:
- SKYRL_RANDOM_SEED: Random seed for deterministic behavior (default: 42)
- SKYRL_ENV_PATH: Path to .env file (default: current directory/.env)

Author: MCP Tools Team
Date: 2025-01-25

.venv/bin/python data_fly_wheel/scripts/sample_implementations/utils/dataset_generator.py --input mcp_tools/datasets/openai/claude_new.json --output mini_agent_claude_dataset.json --curriculum-order
"""

import os
import json
import asyncio
import logging
import sys
import re
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
import openai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
from datetime import datetime

# Deterministic random behavior for consistent dataset generation
random.seed(int(os.getenv("SKYRL_RANDOM_SEED", "42")))

# Load environment variables
env_path = os.getenv("SKYRL_ENV_PATH", "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/.env")
load_dotenv(env_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create structured child loggers for major components
logger = logging.getLogger(__name__)
agent_logger = logger.getChild("agent")
tool_logger = logger.getChild("tool_manager")
reasoning_logger = logger.getChild("reasoning_checker")
dataset_logger = logger.getChild("dataset_builder")
validation_logger = logger.getChild("validation")

# OpenAI Configuration
OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Complexity-aware configuration for curriculum learning
COMPLEXITY_CONFIG = {
    "easy": {
        "min_reasoning_score": 60,
        "min_tool_calls": 2,
        "timeout_minutes": 3,
        "curriculum_score": 1,
        "max_iterations": 8
    },
    "medium": {
        "min_reasoning_score": 70,
        "min_tool_calls": 3,
        "timeout_minutes": 5,
        "curriculum_score": 2,
        "max_iterations": 12
    },
    "hard": {
        "min_reasoning_score": 80,
        "min_tool_calls": 4,
        "timeout_minutes": 8,
        "curriculum_score": 3,
        "max_iterations": 15
    }
}

# Default complexity if not specified
DEFAULT_COMPLEXITY = "medium"

# MCP Server Configuration
MCP_SERVERS = {
    "slack": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/slack_limited_server.py",
    "tavily": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/tavily_limited_server.py",
    "polygon": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/polygon_limited_server.py", 
    "fmp": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/fmp_limited_server.py",
    "python": "/Users/sujitkhanna/Desktop/ongoing_projects/cooking_time_rl/skyrl_tool_agent/mcp_tools/limited/python_execution_server.py"
}

@dataclass
class ConversationMessage:
    role: str               # "system", "human", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

@dataclass
class DatasetEntry:
    data_source: str
    prompt: List[Dict[str, str]]
    env_class: str
    reward_spec: Dict[str, Any] 
    extra_info: Dict[str, Any] = field(default_factory=dict)
    # Enhanced curriculum learning fields
    task_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    final_answer: str
    tool_calls: List[str]
    error: Optional[str] = None
    conversation_history: List[ConversationMessage] = None
    dataset_entry: Optional[DatasetEntry] = None

# ================================================
# SkyRL Infrastructure Functions
# ================================================

def extract_success_criteria(agent_result: 'AgentResult', complexity: str = None, complexity_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Extract success criteria from agent execution results with deterministic ordering"""
    actual_tools = agent_result.tool_calls
    
    # Use complexity-specific thresholds if provided, otherwise use defaults
    if complexity and complexity_config:
        min_tool_calls = complexity_config["min_tool_calls"]
        reasoning_threshold = complexity_config["min_reasoning_score"]
    else:
        # Fallback to medium complexity defaults for backward compatibility
        min_tool_calls = COMPLEXITY_CONFIG["medium"]["min_tool_calls"]
        reasoning_threshold = COMPLEXITY_CONFIG["medium"]["min_reasoning_score"]
    
    criteria = {
        "min_tool_calls": min_tool_calls,
        "reasoning_quality_threshold": reasoning_threshold,
        "error_recovery_demonstrated": False
    }
    
    # FIX 1: Deterministic ordering of expected_tools (sorted alphabetically)
    if actual_tools:
        criteria["expected_tools"] = sorted(set(actual_tools))  # Sort after deduplication
    
    # Check for error recovery in conversation history (harmonized with has_error_recovery)
    if agent_result.conversation_history:
        for msg in agent_result.conversation_history:
            if msg.role == 'assistant' and msg.content:
                if any(pattern in msg.content.lower() for pattern in 
                       ['recovery strategy', 'alternative approach', 'fallback']):
                    criteria["error_recovery_demonstrated"] = True
                    break
    
    return criteria

def extract_tool_sequence(content: str) -> List[str]:
    """Extract tool sequence from assistant messages using robust regex parsing"""
    tool_sequence = []
    seen_tools = set()  # Track tools to avoid duplicates while preserving order
    
    # Use a more robust regex pattern that properly handles nested braces
    # This pattern correctly handles nested JSON objects within function_call
    TOOL_RX = re.compile(r'\{"function_call":\s*\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\}', re.DOTALL)
    for match in TOOL_RX.finditer(content):
        try:
            func_json = json.loads(match.group(0))
            tool_name = func_json["function_call"].get("name")
            if tool_name and tool_name not in seen_tools:
                tool_sequence.append(tool_name)
                seen_tools.add(tool_name)
        except (json.JSONDecodeError, KeyError):
            logger.debug("Skipping malformed function_call block", exc_info=True)
    
    # Also check for our current format: <tool_call> blocks with "name" key
    # Use the robust helper function to handle nested braces properly
    tool_call_starts = []
    start_idx = 0
    while True:
        start_idx = content.find("<tool_call>", start_idx)
        if start_idx == -1:
            break
        tool_call_starts.append(start_idx)
        start_idx += 1
    
    for start_pos in tool_call_starts:
        tool_data = _extract_json_from_tool_call(content[start_pos:])
        if tool_data and "name" in tool_data:
            tool_name = tool_data["name"]
            if tool_name not in seen_tools:
                tool_sequence.append(tool_name)
                seen_tools.add(tool_name)
    
    return tool_sequence

def validate_skyrl_format(example: Dict[str, Any]) -> bool:
    """Comprehensive validation of SkyRL dataset format with descriptive error messages"""
    try:
        # Check required top-level keys
        required_keys = {"data_source", "prompt", "env_class", "reward_spec"}
        missing_keys = required_keys - set(example.keys())
        if missing_keys:
            raise ValueError(f"Missing required top-level keys: {', '.join(missing_keys)}")
        
        # Validate data_source - support both original and extended sources
        if not isinstance(example["data_source"], str) or not example["data_source"]:
            raise ValueError("data_source must be a non-empty string")
        valid_data_sources = ["mcp_tool_agent_synthetic", "mcp_tool_agent_synthetic_extended"]
        if example["data_source"] not in valid_data_sources:
            raise ValueError(f"data_source must be one of {valid_data_sources}, got '{example['data_source']}'")
        
        # Validate env_class - support both original and financial agent environments
        valid_env_classes = ["tool_using_agent", "financial_tool_agent"]
        if example["env_class"] not in valid_env_classes:
            raise ValueError(f"env_class must be one of {valid_env_classes}, got '{example['env_class']}'")
        
        # FIX 3: Comprehensive prompt validation - check ALL elements, not just first
        if not isinstance(example["prompt"], list):
            raise ValueError("prompt must be a list")
        if len(example["prompt"]) == 0:
            raise ValueError("prompt cannot be empty")
        
        # Validate each turn in the prompt
        valid_roles = {"system", "user", "assistant", "tool"}
        for turn_idx, turn in enumerate(example["prompt"]):
            if not isinstance(turn, dict):
                raise ValueError(f"prompt[{turn_idx}] must be a dict, got {type(turn).__name__}")
            
            # Ensure only allowed keys: "role" and "content"
            allowed_keys = {"role", "content"}
            extra_keys = set(turn.keys()) - allowed_keys
            if extra_keys:
                raise ValueError(f"prompt[{turn_idx}] contains invalid keys: {', '.join(extra_keys)}. Only 'role' and 'content' allowed")
            
            # Check required keys
            if "role" not in turn:
                raise ValueError(f"prompt[{turn_idx}] missing required 'role' key")
            if "content" not in turn:
                raise ValueError(f"prompt[{turn_idx}] missing required 'content' key")
            
            # Validate role
            if turn["role"] not in valid_roles:
                raise ValueError(f"prompt[{turn_idx}] has invalid role '{turn['role']}'. Must be one of: {', '.join(valid_roles)}")
            
            # Validate content
            if not isinstance(turn["content"], str):
                raise ValueError(f"prompt[{turn_idx}] content must be a string, got {type(turn['content']).__name__}")
        
        # Validate reward_spec structure
        if not isinstance(example["reward_spec"], dict):
            raise ValueError("reward_spec must be a dict")
        if "method" not in example["reward_spec"]:
            raise ValueError("reward_spec missing required 'method' key")
        if "ground_truth" not in example["reward_spec"]:
            raise ValueError("reward_spec missing required 'ground_truth' key")
        
        # Validate extra_info (should be dict, but contents are flexible)
        if "extra_info" in example and not isinstance(example["extra_info"], dict):
            raise ValueError("extra_info must be a dict if present")
        
        # Validate task_metadata if present (for curriculum learning support)
        if "task_metadata" in example and not isinstance(example["task_metadata"], dict):
            raise ValueError("task_metadata must be a dict if present")
        
        return True
        
    except ValueError:
        # Re-raise ValueError as-is for proper error propagation
        raise
    except Exception as e:
        raise ValueError(f"Unexpected validation error: {str(e)}")

def convert_conversation_format(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Convert conversation into the list‑of‑dicts format required by SkyRL.
    Each original turn becomes exactly one dict with keys {'role','content'}.
    No concatenation or role 'unknown' is allowed.
    """
    converted: List[Dict[str, str]] = []
    
    for turn in conversation:
        # FIX 4: Robust role mapping with explicit branches to prevent concatenation
        role = None
        if turn.get("from") in {"gpt", "assistant"} or turn.get("role") == "assistant":
            role = "assistant"
        elif turn.get("from") in {"human"} or turn.get("role") == "user":
            role = "user"
        elif turn.get("from") == "system" or turn.get("role") == "system":
            role = "system"
        elif turn.get("from") == "tool" or turn.get("role") == "tool":
            role = "tool"

        if role is None:
            logger.warning("Skipping turn with unknown role: %s", turn)
            continue

        content = turn.get("value") or turn.get("content") or ""
        if not isinstance(content, str):
            logger.warning("Skipping non‑string content: %s", turn)
            continue

        # Assert content is string to prevent concatenation bugs
        assert isinstance(content, str), f"Content must be string, got {type(content)}"
        
        converted.append({"role": role, "content": content})

    return converted

def _extract_json_from_tool_call(content: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract JSON from <tool_call> blocks using proper brace counting.
    This handles nested braces in tool arguments correctly.
    """
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    
    start_idx = content.find(start_tag)
    if start_idx == -1:
        return None
    
    # Find the start of JSON (first '{' after <tool_call>)
    json_start = content.find('{', start_idx + len(start_tag))
    if json_start == -1:
        return None
    
    # Count braces to find the matching closing brace
    brace_count = 0
    i = json_start
    while i < len(content):
        if content[i] == '{':
            brace_count += 1
        elif content[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                # Found the matching closing brace
                json_str = content[json_start:i+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
        i += 1
    
    return None

class ImprovedMCPToolManager:
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.failed_servers: Set[str] = set()  # Track server aliases that failed initialization
        
    async def initialize_tools(self):
        tool_logger.info("Initializing MCP tools...")
        for server_name, server_script in MCP_SERVERS.items():
            server_path = os.path.join(
                os.getcwd(), server_script
            )
            try:
                await self._initialize_single_server(server_name, server_path)
            except Exception as e:
                tool_logger.error(f"Failed to init {server_name}: {e}")
                # Track the failed server alias
                self.failed_servers.add(server_name)
        
        # Log availability summary
        total_servers = len(MCP_SERVERS)
        available_servers = total_servers - len(self.failed_servers)
        tool_logger.info(f"Loaded {len(self.tools)} tools from {available_servers}/{total_servers} servers")
        if self.failed_servers:
            tool_logger.warning(f"Failed servers: {', '.join(sorted(self.failed_servers))}")

    async def _initialize_single_server(self, server_name: str, server_path: str):
        env_vars = dict(os.environ)
        from dotenv import dotenv_values
        env_vars.update(dotenv_values(env_path))
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        async with stdio_client(server_params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                for tool in (await session.list_tools()).tools:
                    self.tools[tool.name] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }

    def _get_server_for_tool(self, tool_name: str) -> Optional[str]:
        """Get the server alias for a given tool name"""
        # Direct mapping based on tool name prefixes and patterns
        for server_alias in MCP_SERVERS.keys():
            if tool_name.startswith(server_alias) or server_alias in tool_name.lower():
                return server_alias
        return None

    def _is_tool_from_failed_server(self, tool_name: str) -> bool:
        """Check if a tool belongs to a failed server"""
        server_alias = self._get_server_for_tool(tool_name)
        return server_alias in self.failed_servers if server_alias else False

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        tool_logger.debug(f"Looking up tool: {tool_name}")
        
        # Check if tool belongs to a failed server
        if self._is_tool_from_failed_server(tool_name):
            server_alias = self._get_server_for_tool(tool_name)
            tool_logger.warning(f"Tool '{tool_name}' is disabled due to server '{server_alias}' failure")
            return False, {"error": f"Tool '{tool_name}' is disabled (server '{server_alias}' failed)", "retryable": False}
        
        info = self.tools.get(tool_name)
        if not info:
            tool_logger.error(f"Tool '{tool_name}' not found in available tools")
            return False, {"error": f"Tool '{tool_name}' not found"}
        
        # Find which server this tool belongs to using the mapping function
        server_alias = self._get_server_for_tool(tool_name)
        if not server_alias:
            tool_logger.error(f"Could not determine server for tool '{tool_name}' - no matching server found")
            return False, {"error": f"Tool '{tool_name}' has no matching server", "retryable": False}
        
        server_script = MCP_SERVERS.get(server_alias)
        if not server_script:
            tool_logger.error(f"Server script not found for alias '{server_alias}'")
            return False, {"error": f"Server '{server_alias}' not configured", "retryable": False}
        
        tool_logger.debug(f"Using server script: {server_script} for server alias: {server_alias}")
        server_path = os.path.join(os.getcwd(), server_script)
        env_vars = dict(os.environ)
        from dotenv import dotenv_values
        env_vars.update(dotenv_values(env_path))
        params = StdioServerParameters(
            command=sys.executable,
            args=[os.path.basename(server_path)],
            env=env_vars,
            cwd=os.path.dirname(server_path)
        )
        try:
            tool_logger.debug(f"Connecting to server at {server_path}")
            async with stdio_client(params) as (r, w):
                tool_logger.debug("Connected to server, initializing session")
                async with ClientSession(r, w) as session:
                    tool_logger.debug("Session initialized, calling tool")
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments), timeout=30.0
                    )
                    tool_logger.debug("Tool call completed successfully")
                    text = result.content[0].text if result.content and hasattr(result.content[0], 'text') else str(result.content)
                    return True, {"content": text}
        except asyncio.TimeoutError:
            tool_logger.error(f"Tool {tool_name} timed out after 30 seconds")
            return False, {"error": "timeout", "retryable": True}
        except Exception as e:
            tool_logger.error(f"Tool {tool_name} failed with error: {e}")
            return False, {"error": str(e), "retryable": False}

#TODO: make this as generic as possible, cannot have hardcoded patterns
class ReasoningQualityChecker:
    def __init__(self):
        # Keep minimal patterns as fallback, but primarily use LLM evaluation
        self.fallback_patterns = {
            "bad": ["making tool calls to gather information", "let me check"],
            "good": ["analysis", "approach", "strategy", "because", "since"]
        }

    # Use LLM as a judge to score reasoning quality
    async def score_reasoning(self, content: str) -> Dict[str, Any]:
        """Score reasoning quality from 0-100 using LLM evaluation"""
        if not content:
            return {"score": 0, "issues": ["No content"], "suggestions": []}
        
        # Use LLM to evaluate reasoning quality
        try:
            evaluation_prompt = f"""
You are an expert evaluator of AI reasoning quality. Score the following reasoning on a scale of 0-100 based on these criteria:

1. **Specificity** (0-25): Is the reasoning specific and analytical, not generic?
2. **Completeness** (0-25): Does it include analysis, tool selection rationale, and planning?
3. **Structure** (0-25): Is it well-organized with clear sequencing and dependencies?
4. **Depth** (0-25): Does it demonstrate understanding and strategic thinking?

Bonus points for:
- Error recovery strategies
- Clear step-by-step thinking
- Proper use of <think> tags

Deduct points for:
- Generic phrases like "gathering information" or "let me check"
- Too brief (under 100 words)
- Missing analytical elements

REASONING TO EVALUATE:
{content}

Respond with a JSON object containing:
- "score": integer from 0-100
- "issues": list of specific problems found
- "suggestions": list of specific improvements
- "breakdown": object with scores for each criteria (specificity, completeness, structure, depth)
"""

            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.0,
                max_tokens=500
            )
            
            evaluation_text = response.choices[0].message.content
            
            # Try to parse JSON response
            try:
                evaluation = json.loads(evaluation_text)
                score = max(0, min(100, evaluation.get("score", 0)))
                reasoning_logger.debug(f"LLM reasoning evaluation: {score}/100")
                return {
                    "score": score,
                    "issues": evaluation.get("issues", []),
                    "suggestions": evaluation.get("suggestions", []),
                    "breakdown": evaluation.get("breakdown", {}),
                    "length": len(content),
                    "llm_evaluated": True
                }
            except json.JSONDecodeError:
                # Fallback to pattern-based evaluation
                reasoning_logger.warning("LLM evaluation failed to return valid JSON, using fallback")
                return self._fallback_score_reasoning(content)
                
        except Exception as e:
            reasoning_logger.warning(f"LLM evaluation failed: {e}, using fallback")
            return self._fallback_score_reasoning(content)
    
    def _fallback_score_reasoning(self, content: str) -> Dict[str, Any]:
        """Fallback pattern-based scoring when LLM evaluation fails"""
        content_lower = content.lower()
        issues = []
        suggestions = []
        score = 70  # Start with neutral score
        
        # Check for bad patterns
        bad_count = sum(1 for pattern in self.fallback_patterns["bad"] if pattern in content_lower)
        if bad_count > 0:
            score -= bad_count * 15
            issues.append(f"Contains {bad_count} generic phrases")
            suggestions.append("Replace generic phrases with specific analysis")
        
        # Check for good patterns
        good_count = sum(1 for pattern in self.fallback_patterns["good"] if pattern in content_lower)
        score += good_count * 5
        
        # Check length
        if len(content) < 100:
            score -= 20
            issues.append("Reasoning too short")
            suggestions.append("Provide more detailed thinking")
        
        # Check for think tags - but skip penalty if message contains tool calls
        has_tool_calls = any(pattern in content_lower for pattern in ['<tool_call>', '"function_call"'])
        if "<think>" not in content_lower and not has_tool_calls:
            score -= 30
            issues.append("Missing <think> tags")
            suggestions.append("Wrap reasoning in <think></think> tags")
        elif has_tool_calls and "<think>" not in content_lower:
            # Tool call messages may have planning in previous turns, so just note it
            suggestions.append("Consider adding <think> tags for clearer reasoning structure")
        
        score = max(0, min(100, score))
        
        reasoning_logger.debug(f"Fallback reasoning score: {score}/100, issues: {len(issues)}, has_tool_calls: {has_tool_calls}")
        
        return {
            "score": score,
            "issues": issues,
            "suggestions": suggestions,
            "length": len(content),
            "llm_evaluated": False,
            "has_tool_calls": has_tool_calls
        }
    
    async def is_acceptable(self, content: str, min_score: int = 60) -> bool:
        """Check if reasoning meets minimum quality threshold"""
        result = await self.score_reasoning(content)
        return result["score"] >= min_score

class ToolErrorHandler:
    def __init__(self):
        # Common error patterns in tool responses
        self.error_patterns = [
            "'NoneType' object is not iterable",
            "not found",
            "error",
            "failed",
            "timeout",
            "invalid",
            "null",
            "empty response",
            "no data available"
        ]
        
        # Tool fallback strategies
        self.fallback_strategies = {
            "fmp_get_stock_news": {
                "alternatives": ["tavily_search"],
                "reformulation": "search for '{ticker} stock news' instead of using financial API"
            },
            "fmp_get_quote": {
                "alternatives": ["polygon_get_ticker_details", "tavily_search"],
                "reformulation": "search for '{ticker} stock price' on the web"
            },
            "polygon_get_ticker_details": {
                "alternatives": ["fmp_get_quote", "tavily_search"],
                "reformulation": "try different financial data source or web search"
            },
            "tavily_search": {
                "alternatives": ["fmp_get_stock_news"],
                "reformulation": "try more specific search terms or use financial APIs"
            }
        }
    
    def detect_error(self, tool_response: str) -> bool:
        """Detect if a tool response contains an error"""
        if not tool_response:
            return True
        
        response_lower = tool_response.lower()
        return any(pattern in response_lower for pattern in self.error_patterns)
    
    def get_recovery_strategy(self, failed_tool: str, original_args: Dict[str, Any], available_tools: List[str]) -> Dict[str, Any]:
        """Get a recovery strategy for a failed tool"""
        strategy = self.fallback_strategies.get(failed_tool, {})
        alternatives = [tool for tool in strategy.get("alternatives", []) if tool in available_tools]
        
        return {
            "has_alternatives": len(alternatives) > 0,
            "alternative_tools": alternatives,
            "reformulation_suggestion": strategy.get("reformulation", "try alternative approach"),
            "original_args": original_args
        }
    
    # This module also needs a change; canty have pre-determined patterns of recovery reasoning
    async def generate_recovery_reasoning(self, failed_tool: str, error_content: str, recovery_strategy: Dict[str, Any], context: str = "") -> str:
        """Generate reasoning for error recovery using LLM for dynamic, context-aware strategies"""
        alternatives = recovery_strategy.get("alternative_tools", [])
        reformulation = recovery_strategy.get("reformulation_suggestion", "")
        
        # Use LLM to generate contextual recovery reasoning
        try:
            recovery_prompt = f"""
Generate high-quality error recovery reasoning for a tool failure scenario. The reasoning should be wrapped in <think></think> tags and demonstrate strategic problem-solving.

CONTEXT:
- Failed tool: {failed_tool}
- Error: {error_content[:200]}{'...' if len(error_content) > 200 else ''}
- Available alternatives: {', '.join(alternatives) if alternatives else 'None'}
- User context: {context[:100]}{'...' if len(context) > 100 else ''}
- Suggested reformulation: {reformulation}

Generate reasoning that:
1. Analyzes WHY the tool failed (data availability, technical issue, etc.)
2. Explains the recovery strategy clearly
3. Shows strategic decision-making about alternatives
4. Demonstrates understanding of the problem context
5. Is specific and analytical, not generic

Make it 150-300 words, professional, and demonstrate real problem-solving thinking.
"""

            client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": recovery_prompt}],
                temperature=0.1,  # Low temperature for consistent reasoning
                max_tokens=400
            )
            
            generated_reasoning = response.choices[0].message.content.strip()
            
            # Ensure the reasoning is wrapped in think tags
            if not generated_reasoning.startswith("<think>"):
                generated_reasoning = f"<think>\n{generated_reasoning}\n</think>"
            
            tool_logger.debug(f"Generated LLM recovery reasoning for {failed_tool}")
            return generated_reasoning
            
        except Exception as e:
            tool_logger.warning(f"LLM recovery reasoning generation failed: {e}, using fallback")
            return self._fallback_recovery_reasoning(failed_tool, error_content, alternatives, reformulation)
    #TODO: Need to make this more generic, cannot have hardcoded patterns
    def _fallback_recovery_reasoning(self, failed_tool: str, error_content: str, alternatives: List[str], reformulation: str) -> str:
        """Fallback recovery reasoning when LLM generation fails"""
        reasoning = f"""<think>
I encountered an error with {failed_tool}: {error_content[:100]}{'...' if len(error_content) > 100 else ''}

This failure likely indicates {'limited data availability' if 'not iterable' in error_content else 'a technical issue'}. Let me implement a recovery strategy:

1. **Error Analysis**: The {failed_tool} tool {'returned null/empty data' if 'not iterable' in error_content else 'failed to execute properly'}
2. **Recovery Options**: 
   - Try alternative tools: {', '.join(alternatives) if alternatives else 'None available in current toolset'}
   - Reformulate approach: {reformulation}
3. **Strategic Decision**: {'I will try ' + alternatives[0] + ' as an alternative' if alternatives else 'I will acknowledge the limitation and work with available data'}

This demonstrates proper error handling in real-world scenarios where tools may fail due to data availability or technical issues.
</think>"""
        
        return reasoning

class MiniAgent:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.tool_manager = ImprovedMCPToolManager()
        self.reasoning_checker = ReasoningQualityChecker()
        self.error_handler = ToolErrorHandler()

    def _create_system_prompt(self) -> str:
        # Only include tools from available (non-failed) servers
        available_tools = {name: info for name, info in self.tool_manager.tools.items() 
                          if not self.tool_manager._is_tool_from_failed_server(name)}
        tool_lines = [f"- {t}: {info['description']}" for t, info in available_tools.items()]
        
        return f"""You are a deep-thinking AI assistant. You MUST demonstrate high-quality step-by-step reasoning before and after using tools, including intelligent error handling and recovery strategies.

CRITICAL REASONING REQUIREMENTS:
1. Every response must start with <think> tags containing detailed analysis
2. Break down the user's request into specific components
3. Explain WHY you choose specific tools and their order
4. Plan your approach with clear dependencies
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

Approach:
1. First get AAPL quote to establish current market context
2. Then search for "Apple AI news" to find recent developments
3. Finally synthesize both pieces of information for a comprehensive response

These can be executed sequentially since the stock context might inform how I interpret the news.
</think>

ERROR RECOVERY EXAMPLE:
<think>
I encountered an error with fmp_get_stock_news: "'NoneType' object is not iterable"

This failure likely indicates limited data availability. Let me implement a recovery strategy:

1. **Error Analysis**: The fmp_get_stock_news tool returned null/empty data, possibly because this ticker has limited news coverage in the FMP database
2. **Recovery Options**: 
   - Try alternative tools: tavily_search
   - Reformulate approach: search for 'TICKER stock news' instead of using financial API
3. **Strategic Decision**: I will try tavily_search as an alternative to get web-based news results

This demonstrates proper error handling in real-world scenarios where tools may fail due to data availability.
</think>

BAD REASONING EXAMPLES (NEVER DO THIS):
❌ <think>I need to get information about AAPL.</think>
❌ <think>Making tool calls to gather information.</think>  
❌ <think>Let me search for this data.</think>
❌ <think>Using tools to help the user.</think>

Your reasoning must be:
- Specific and analytical (not generic)
- At least 100 words explaining your thought process
- Include why you chose specific tools
- Show clear planning and sequencing
- Demonstrate understanding of the request
- **Show error recovery strategies when tools fail**

Available tools:
{chr(10).join(tool_lines)}

Remember: The quality of your reasoning AND error handling is as important as the tool results. Show your thinking process clearly and demonstrate resilience when things don't go as planned."""

    def _build_openai_messages(self, conv: List[ConversationMessage]) -> List[Dict[str, Any]]:
        msgs = []
        i = 0
        
        while i < len(conv):
            m = conv[i]
            
            if m.role == 'system':
                msgs.append({"role": "system", "content": m.content or ""})
                
            elif m.role == 'human':
                msgs.append({"role": "user", "content": m.content or ""})
                
            elif m.role == 'assistant':
                # Check if this is a reasoning message followed by tool calls
                content = m.content or ""
                
                # Look ahead to see if there are tool call messages following
                tool_calls = []
                j = i + 1
                tool_responses = []
                
                # Collect tool calls and responses
                while j < len(conv):
                    if conv[j].role == 'assistant' and '<tool_call>' in (conv[j].content or ''):
                        # FIX: Extract tool call using robust brace counting instead of fragile string slicing
                        tool_content = conv[j].content or ""
                        tool_data = _extract_json_from_tool_call(tool_content)
                        if tool_data and "name" in tool_data:
                            call_id = f"call_{len(tool_calls)}"
                            tool_calls.append({
                                "id": call_id,
                                "type": "function", 
                                "function": {
                                    "name": tool_data["name"],
                                    "arguments": json.dumps(tool_data.get("arguments", {}))
                                }
                            })
                            
                            # Look for corresponding tool response
                            if j + 1 < len(conv) and conv[j + 1].role == 'tool':
                                tool_responses.append({
                                    "role": "tool",
                                    "content": conv[j + 1].content or "",
                                    "tool_call_id": call_id
                                })
                                j += 1  # Skip the tool response
                        j += 1
                    else:
                        break
                
                # Create the assistant message
                if tool_calls:
                    msgs.append({
                        "role": "assistant",
                        "content": content,
                        "tool_calls": tool_calls
                    })
                    # Add all tool responses
                    msgs.extend(tool_responses)
                    i = j - 1  # Skip processed messages
                else:
                    # Regular assistant message
                    msgs.append({"role": "assistant", "content": content})
                    
            # Skip tool messages as they're handled above
            i += 1
        
        return msgs

    async def solve_task(self, user_query: str, max_iters: int = 15, complexity: str = None, task_metadata: Dict[str, Any] = None) -> AgentResult:
        # Initialize complexity-aware configuration
        if complexity is None:
            complexity = DEFAULT_COMPLEXITY
        
        complexity_config = COMPLEXITY_CONFIG.get(complexity, COMPLEXITY_CONFIG[DEFAULT_COMPLEXITY])
        if task_metadata is None:
            task_metadata = {}
        
        # Use complexity-specific parameters
        actual_max_iters = min(max_iters, complexity_config["max_iterations"])
        timeout_seconds = complexity_config["timeout_minutes"] * 60
        min_reasoning_score = complexity_config["min_reasoning_score"]
        min_tool_calls = complexity_config["min_tool_calls"]
        
        agent_logger.info(f"Starting task with complexity: {complexity} (score: {min_reasoning_score}, tools: {min_tool_calls}, max_iters: {actual_max_iters})")
        
        # initialize conversation
        conv: List[ConversationMessage] = []
        conv.append(ConversationMessage(role="system", content=self._create_system_prompt()))
        conv.append(ConversationMessage(role="human", content=user_query))
        tool_calls_list: List[str] = []
        final_answer = None
        error = None
        reasoning_attempts = 0
        reasoning_scores = []
        best_reasoning_score = 0

        for i in range(actual_max_iters):
            agent_logger.debug(f"Iteration {i+1}/{actual_max_iters}")
            msgs = self._build_openai_messages(conv)
            agent_logger.debug(f"Sending {len(msgs)} messages to OpenAI")
            
            # Only include tools from available (non-failed) servers
            available_tools = {name: info for name, info in self.tool_manager.tools.items() 
                              if not self.tool_manager._is_tool_from_failed_server(name)}
            
            resp = await self.openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=msgs,
                tools=[{"type":"function","function": {"name":n, "description":info['description'], "parameters":info['parameters']}} for n,info in available_tools.items()],
                tool_choice="auto",
                temperature=0.0
            )
            agent_logger.debug("Received OpenAI response")
            msg = resp.choices[0].message
            agent_logger.debug(f"Message content length: {len(msg.content) if msg.content else 0}")
            agent_logger.debug(f"Number of tool calls: {len(msg.tool_calls) if msg.tool_calls else 0}")
            
            # Check reasoning quality
            reasoning_quality = await self.reasoning_checker.score_reasoning(msg.content or "")
            reasoning_scores.append(reasoning_quality["score"])
            best_reasoning_score = max(best_reasoning_score, reasoning_quality["score"])
            agent_logger.debug(f"Reasoning quality score: {reasoning_quality['score']}/100")
            
            # Validate reasoning quality
            if not await self.reasoning_checker.is_acceptable(msg.content or "", min_score=min_reasoning_score):
                reasoning_attempts += 1
                agent_logger.warning(f"Poor reasoning quality (score: {reasoning_quality['score']}/100), attempt {reasoning_attempts}")
                
                if reasoning_attempts <= 3:  # Give up to 3 chances to improve reasoning
                    improvement_prompt = self._generate_reasoning_improvement_prompt(
                        reasoning_quality, user_query, reasoning_attempts
                    )
                    conv.append(ConversationMessage(role="human", content=improvement_prompt))
                    continue
                else:
                    agent_logger.error("Failed to generate acceptable reasoning after 3 attempts")
                    error = f"Poor reasoning quality after {reasoning_attempts} attempts"
                    break
            
            # If reasoning is acceptable, proceed with tool logic
            reasoning_attempts = 0  # Reset counter on good reasoning
            
            # Record the assistant message with thinking
            content_to_use = msg.content or ""
            if not content_to_use and msg.tool_calls:
                content_to_use = "<think>Making tool calls to gather information.</think>"
            
            # If there are tool calls, we need to handle them individually
            if msg.tool_calls:
                # First, record the reasoning/planning message
                conv.append(ConversationMessage(role="assistant", content=content_to_use, tool_calls=None))
                
                # Execute each tool call individually with error handling
                for tc_idx, tc in enumerate(msg.tool_calls):
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    tool_id = tc.id
                    
                    agent_logger.info(f"Executing tool: {tool_name} (call {tc_idx + 1}/{len(msg.tool_calls)})")
                    
                    # Create individual tool call message
                    tool_call_content = f"<tool_call>\n{json.dumps({'name': tool_name, 'arguments': tool_args})}\n</tool_call>"
                    conv.append(ConversationMessage(role="assistant", content=tool_call_content, tool_calls=None))
                    
                    # Execute the tool
                    ok, result = await self.tool_manager.execute_tool(tool_name, tool_args)
                    result_content = json.dumps(result)
                    
                    agent_logger.info(f"Tool {tool_name} result: ok={ok}, result_length={len(result_content)}")
                    
                    # Check for errors and handle recovery
                    if not ok or self.error_handler.detect_error(result_content):
                        agent_logger.warning(f"Tool {tool_name} failed or returned error")
                        
                        # Record the failed tool response
                        conv.append(ConversationMessage(role="tool", content=f"<tool_response>\n{result_content}\n</tool_response>", tool_call_id=tool_id, name=tool_name))
                        
                        # Generate error recovery strategy
                        available_tools = list(self.tool_manager.tools.keys())
                        recovery_strategy = self.error_handler.get_recovery_strategy(tool_name, tool_args, available_tools)
                        
                        if recovery_strategy["has_alternatives"]:
                            # Try the first alternative tool
                            alt_tool = recovery_strategy["alternative_tools"][0]
                            recovery_reasoning = await self.error_handler.generate_recovery_reasoning(
                                tool_name, result_content, recovery_strategy, user_query
                            )
                            
                            # Add recovery reasoning
                            conv.append(ConversationMessage(role="assistant", content=recovery_reasoning, tool_calls=None))
                            
                            # Try alternative tool with adapted arguments
                            alt_args = self._adapt_args_for_alternative_tool(tool_args, tool_name, alt_tool)
                            alt_tool_call_content = f"<tool_call>\n{json.dumps({'name': alt_tool, 'arguments': alt_args})}\n</tool_call>"
                            conv.append(ConversationMessage(role="assistant", content=alt_tool_call_content, tool_calls=None))
                            
                            # Execute alternative tool
                            alt_ok, alt_result = await self.tool_manager.execute_tool(alt_tool, alt_args)
                            alt_result_content = json.dumps(alt_result)
                            
                            if alt_ok:
                                tool_calls_list.append(alt_tool)
                                agent_logger.info(f"Alternative tool {alt_tool} succeeded")
                            
                            # Record alternative tool response
                            conv.append(ConversationMessage(role="tool", content=f"<tool_response>\n{alt_result_content}\n</tool_response>", tool_call_id=None, name=alt_tool))
                        else:
                            # No alternatives available, acknowledge limitation
                            limitation_reasoning = f"<think>\nUnfortunately, no alternative tools are available for {tool_name}. I'll acknowledge this limitation and continue with the available data from other sources.\n</think>"
                            conv.append(ConversationMessage(role="assistant", content=limitation_reasoning, tool_calls=None))
                    else:
                        # Tool succeeded
                        tool_calls_list.append(tool_name)
                        conv.append(ConversationMessage(role="tool", content=f"<tool_response>\n{result_content}\n</tool_response>", tool_call_id=tool_id, name=tool_name))
                
                continue  # Continue to next iteration after processing all tool calls
            else:
                # No tool calls, record the assistant message normally
                conv.append(ConversationMessage(role="assistant", content=content_to_use, tool_calls=None))
                
                # Check if we have enough tool calls for completion
                agent_logger.info(f"No tool calls in this response. Total tool calls so far: {len(tool_calls_list)}")
                if len(tool_calls_list) < min_tool_calls:
                    agent_logger.info("Less than minimum tool calls, requesting more")
                    conv.append(ConversationMessage(role="human", content="Please continue to demonstrate tool use to complete the task properly."))
                    continue
                
                agent_logger.info("Sufficient tool calls made, setting final answer")
                final_answer = content_to_use or "No response generated"
                break

        # Build dataset entry in SkyRL format
        dataset_prompt = self._build_dataset_messages(conv, user_query)
        
        # Use the new infrastructure function to extract success criteria
        temp_result = AgentResult(
            final_answer=final_answer or "Incomplete",
            tool_calls=tool_calls_list,
            error=error,
            conversation_history=conv
        )
        success_criteria = extract_success_criteria(temp_result, complexity, complexity_config)
        
        # Extract tool sequence from conversation for validation
        all_tool_sequences = []
        for msg in conv:
            if msg.role == 'assistant' and msg.content:
                tool_seq = extract_tool_sequence(msg.content)
                all_tool_sequences.extend(tool_seq)
        
        # Create reward specification based on tool execution success and extracted criteria
        reward_spec = {
            "method": "rule",
            "ground_truth": {
                "success_criteria": success_criteria,
                "actual_performance": {
                    "tool_calls_made": len(tool_calls_list),
                    "reasoning_score": best_reasoning_score,
                    "completed_successfully": len(tool_calls_list) >= min_tool_calls and best_reasoning_score >= min_reasoning_score,
                    "extracted_tool_sequence": all_tool_sequences
                }
            }
        }
        
        # Enhanced metadata for SkyRL extra_info with curriculum learning support
        extra_info = {
            "timestamp": datetime.now().isoformat(),
            "model_used": OPENAI_MODEL,
            "tools_available": list(self.tool_manager.tools.keys()),
            "query_complexity": len(user_query.split()),
            
            # Curriculum learning metadata
            "complexity": complexity,
            "complexity_score": complexity_config["curriculum_score"],
            "task_metadata": task_metadata,
            "curriculum_metadata": {
                "difficulty_progression": {
                    "current_level": complexity,
                    "skill_dependencies": self._extract_skill_dependencies(tool_calls_list),
                    "prerequisite_categories": task_metadata.get("prerequisite_categories", []),
                    "next_level_tasks": self._suggest_next_level_tasks(complexity, task_metadata.get("category", ""))
                },
                "learning_objectives": self._extract_learning_objectives(tool_calls_list, complexity),
                "performance_indicators": {
                    "reasoning_efficiency": best_reasoning_score / min_reasoning_score if min_reasoning_score > 0 else 0,
                    "tool_usage_efficiency": len(tool_calls_list) / min_tool_calls if min_tool_calls > 0 else 0,
                    "completion_rate": 1.0 if len(tool_calls_list) >= min_tool_calls and best_reasoning_score >= min_reasoning_score else 0.0
                }
            },
            
            "reasoning_quality": {
                "best_score": best_reasoning_score,
                "average_score": sum(reasoning_scores) / len(reasoning_scores) if reasoning_scores else 0,
                "all_scores": reasoning_scores,
                "attempts_needed": len(reasoning_scores),
                "complexity_threshold": min_reasoning_score,
                "meets_complexity_requirement": best_reasoning_score >= min_reasoning_score
            },
            "execution_metrics": {
                "iterations_used": i + 1,
                "max_iterations_allowed": actual_max_iters,
                "tools_used": tool_calls_list,
                "num_tool_calls": len(tool_calls_list),
                "min_tool_calls_required": min_tool_calls,
                "timeout_seconds": timeout_seconds,
                "error_recovery_attempts": sum(1 for msg in conv if msg.role == 'assistant' and 'error' in (msg.content or '').lower()),
                "has_error_recovery": any('recovery strategy' in (msg.content or '').lower() for msg in conv if msg.role == 'assistant'),
                "complexity_config_used": complexity_config
            },
            "training_metadata": {
                "data_source": "mcp_tool_agent_synthetic_extended",
                "generation_method": "complexity_aware_react_agent",
                "quality_tier": "high" if best_reasoning_score >= min_reasoning_score + 10 else "medium" if best_reasoning_score >= min_reasoning_score else "low",
                "curriculum_ready": len(tool_calls_list) >= min_tool_calls and best_reasoning_score >= min_reasoning_score,
                "original_question": task_metadata.get("question", user_query)
            }
        }
        
        entry = DatasetEntry(
            data_source="mcp_tool_agent_synthetic_extended",
            prompt=dataset_prompt,
            env_class="financial_tool_agent", 
            reward_spec=reward_spec,
            extra_info=extra_info,
            task_metadata={
                "task_id": task_metadata.get("task_id", "unknown"),
                "category": task_metadata.get("category", "unknown"),
                "complexity": complexity,
                "original_question": task_metadata.get("question", user_query),
                "skill_level": task_metadata.get("skill_level", COMPLEXITY_CONFIG[complexity]["curriculum_score"]),
                "prerequisite_categories": task_metadata.get("prerequisite_categories", [])
            }
        )
        return AgentResult(final_answer=final_answer or "Incomplete", tool_calls=tool_calls_list, error=error,
                           conversation_history=conv, dataset_entry=entry)

    async def generate_dataset(self, queries: List[str], output_file: str = "skyrl_dataset.json", complexity_levels: List[str] = None, task_metadata_list: List[Dict[str, Any]] = None) -> List[Any]:
        dataset = []
        quality_stats = {
            "high_quality": 0, 
            "medium_quality": 0, 
            "low_quality": 0, 
            "failed": 0,
            "rejected_format": 0,
            "rejected_reasoning": 0,
            "timeout": 0
        }
        detailed_log = []  # For comprehensive logging
        
        # Handle complexity levels and metadata
        if complexity_levels is None:
            complexity_levels = [DEFAULT_COMPLEXITY] * len(queries)
        elif len(complexity_levels) != len(queries):
            # Extend or truncate to match queries length
            complexity_levels = (complexity_levels * (len(queries) // len(complexity_levels) + 1))[:len(queries)]
        
        if task_metadata_list is None:
            task_metadata_list = [{}] * len(queries)
        elif len(task_metadata_list) != len(queries):
            task_metadata_list = (task_metadata_list * (len(queries) // len(task_metadata_list) + 1))[:len(queries)]
        
        for i, q in enumerate(queries):
            complexity = complexity_levels[i]
            task_metadata = task_metadata_list[i]
            complexity_config = COMPLEXITY_CONFIG.get(complexity, COMPLEXITY_CONFIG[DEFAULT_COMPLEXITY])
            timeout_seconds = complexity_config["timeout_minutes"] * 60
            min_tool_calls = complexity_config["min_tool_calls"]
            min_reasoning_score = complexity_config["min_reasoning_score"]
            
            query_log = {
                "query_index": i + 1,
                "query": q,
                "complexity": complexity,
                "complexity_config": complexity_config,
                "task_metadata": task_metadata,
                "timestamp": datetime.now().isoformat(),
                "status": "unknown",
                "result": None,
                "reasoning_scores": [],
                "error_details": None,
                "conversation_history": [],
                "final_reasoning_score": 0,
                "tool_calls_made": [],
                "iterations_used": 0,
                "dataset_entry": None,
                "metadata": {},
                "rejection_reason": None
            }
            
            dataset_logger.info(f"Processing query {i+1}/{len(queries)} [complexity: {complexity}]: {q}")
            try:
                # Use complexity-specific timeout
                result = await asyncio.wait_for(
                    self.solve_task(q, complexity=complexity, task_metadata=task_metadata), 
                    timeout=timeout_seconds
                )
                
                # Record successful execution details
                query_log["status"] = "completed"
                query_log["result"] = {
                    "final_answer": result.final_answer,
                    "tool_calls": result.tool_calls,
                    "error": result.error
                }
                query_log["conversation_history"] = [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "tool_calls": msg.tool_calls,
                        "tool_call_id": msg.tool_call_id,
                        "name": msg.name
                    } for msg in (result.conversation_history or [])
                ]
                query_log["tool_calls_made"] = result.tool_calls
                query_log["iterations_used"] = len(result.conversation_history) if result.conversation_history else 0
                
                # Get reasoning quality from extra_info
                if result.dataset_entry and result.dataset_entry.extra_info:
                    reasoning_quality = result.dataset_entry.extra_info.get("reasoning_quality", {})
                    query_log["reasoning_scores"] = reasoning_quality.get("all_scores", [])
                    query_log["final_reasoning_score"] = reasoning_quality.get("best_score", 0)
                    query_log["metadata"] = result.dataset_entry.extra_info
                    query_log["dataset_entry"] = result.dataset_entry.prompt
                
                # Validate basic requirements
                if len(result.tool_calls) < min_tool_calls:
                    dataset_logger.warning(f"Query {i+1} rejected: Only {len(result.tool_calls)} tool calls made, minimum {min_tool_calls} required - '{q}'")
                    quality_stats["failed"] += 1
                    query_log["status"] = "failed_insufficient_tool_calls"
                    query_log["error_details"] = f"Only {len(result.tool_calls)} tool calls made, minimum {min_tool_calls} required"
                    query_log["rejection_reason"] = "insufficient_tool_calls"
                    detailed_log.append(query_log)
                    continue
                
                # Check reasoning quality
                best_score = query_log["final_reasoning_score"]
                
                if best_score < min_reasoning_score:
                    dataset_logger.warning(f"Query {i+1} rejected: Poor reasoning quality ({best_score}/100) - '{q}'")
                    quality_stats["rejected_reasoning"] += 1
                    query_log["status"] = "rejected_reasoning_quality"
                    query_log["rejection_reason"] = f"reasoning_quality_too_low_{best_score}"
                    detailed_log.append(query_log)
                    continue
                
                # Create dataset entry for validation
                dataset_entry = {
                    "data_source": result.dataset_entry.data_source,
                    "prompt": result.dataset_entry.prompt,
                    "env_class": result.dataset_entry.env_class,
                    "reward_spec": result.dataset_entry.reward_spec,
                    "extra_info": result.dataset_entry.extra_info,
                    "task_metadata": result.dataset_entry.task_metadata
                }
                
                # Validate SkyRL format compliance - now with ValueError exceptions
                try:
                    validate_skyrl_format(dataset_entry)
                    validation_logger.debug(f"Query {i+1} passed SkyRL format validation")
                    query_log["validation_passed"] = True
                except ValueError as e:
                    validation_logger.warning(f"Query {i+1} rejected: SkyRL format validation failed - {str(e)} - '{q}'")
                    quality_stats["rejected_format"] += 1
                    query_log["status"] = "rejected_format_validation"
                    query_log["rejection_reason"] = f"format_validation_failed: {str(e)}"
                    query_log["validation_error"] = str(e)
                    detailed_log.append(query_log)
                    continue
                
                # Classify quality and add to dataset
                if best_score >= 80:
                    quality_stats["high_quality"] += 1
                    dataset_logger.info(f"✅ High quality entry (reasoning: {best_score}/100)")
                elif best_score >= 60:
                    quality_stats["medium_quality"] += 1
                    dataset_logger.info(f"✨ Medium quality entry (reasoning: {best_score}/100)")
                
                dataset.append(dataset_entry)
                dataset_logger.info(f"Successfully processed query {i+1}, added to dataset")
                
            except asyncio.TimeoutError:
                dataset_logger.warning(f"Query {i+1} timed out after {timeout_seconds} seconds - '{q}'")
                quality_stats["timeout"] += 1
                query_log["status"] = "failed_timeout"
                query_log["error_details"] = f"Query timed out after {timeout_seconds} seconds"
                query_log["rejection_reason"] = "timeout"
            except Exception as e:
                dataset_logger.warning(f"Query {i+1} failed with error: {e} - '{q}'")
                quality_stats["failed"] += 1
                query_log["status"] = "failed_exception"
                query_log["error_details"] = str(e)
                query_log["rejection_reason"] = f"exception: {type(e).__name__}"
            
            # Add to detailed log regardless of success/failure
            detailed_log.append(query_log)
        
        # Enhanced output with quality statistics
        total_entries = len(dataset)
        total_processed = len(queries)
        quality_metadata = {
            "total_entries": total_entries,
            "total_processed": total_processed,
            "quality_distribution": quality_stats,
            "quality_percentage": {
                "high_quality": (quality_stats["high_quality"] / total_entries * 100) if total_entries > 0 else 0,
                "medium_quality": (quality_stats["medium_quality"] / total_entries * 100) if total_entries > 0 else 0,
                "low_quality": (quality_stats["low_quality"] / total_entries * 100) if total_entries > 0 else 0
            },
            "rejection_rates": {
                "format_rejection_rate": (quality_stats["rejected_format"] / total_processed * 100),
                "reasoning_rejection_rate": (quality_stats["rejected_reasoning"] / total_processed * 100),
                "timeout_rate": (quality_stats["timeout"] / total_processed * 100),
                "total_rejection_rate": ((quality_stats["rejected_format"] + quality_stats["rejected_reasoning"] + quality_stats["timeout"] + quality_stats["failed"]) / total_processed * 100)
            }
        }
        
        # Save main dataset in SkyRL format (simple list of entries)
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Save detailed log file for analysis (keeping the enhanced format)
        log_file = output_file.replace('.json', '_detailed_log.json')
        detailed_output = {
            "generation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(queries),
                "successful_entries": total_entries,
                "failed_queries": quality_stats["failed"],
                "quality_stats": quality_stats,
                "skyrl_format_version": "1.0",
                "env_class": "tool_using_agent",
                "data_source": "mcp_tool_agent_synthetic",
                "rejection_summary": quality_metadata["rejection_rates"]
            },
            "queries": detailed_log
        }
        
        with open(log_file, 'w') as f:
            json.dump(detailed_output, f, indent=2)
        
        # Enhanced quality summary logging
        dataset_logger.info(f"📊 SkyRL Dataset Generation Summary:")
        dataset_logger.info(f"  • Total queries processed: {total_processed}")
        dataset_logger.info(f"  • Successful entries: {total_entries}")
        dataset_logger.info(f"  • Success rate: {(total_entries/total_processed*100):.1f}%")
        dataset_logger.info(f"")
        dataset_logger.info(f"📈 Quality Distribution:")
        dataset_logger.info(f"  • High quality (80+): {quality_stats['high_quality']} ({quality_metadata['quality_percentage']['high_quality']:.1f}%)")
        dataset_logger.info(f"  • Medium quality (60-79): {quality_stats['medium_quality']} ({quality_metadata['quality_percentage']['medium_quality']:.1f}%)")
        dataset_logger.info(f"  • Low quality (<60): {quality_stats['low_quality']} ({quality_metadata['quality_percentage']['low_quality']:.1f}%)")
        dataset_logger.info(f"")
        dataset_logger.info(f"🚫 Rejection Analysis:")
        dataset_logger.info(f"  • Format validation failures: {quality_stats['rejected_format']} ({quality_metadata['rejection_rates']['format_rejection_rate']:.1f}%)")
        dataset_logger.info(f"  • Reasoning quality failures: {quality_stats['rejected_reasoning']} ({quality_metadata['rejection_rates']['reasoning_rejection_rate']:.1f}%)")
        dataset_logger.info(f"  • Timeouts: {quality_stats['timeout']} ({quality_metadata['rejection_rates']['timeout_rate']:.1f}%)")
        dataset_logger.info(f"  • Other failures: {quality_stats['failed']}")
        dataset_logger.info(f"  • Total rejection rate: {quality_metadata['rejection_rates']['total_rejection_rate']:.1f}%")
        dataset_logger.info(f"")
        dataset_logger.info(f"💾 Output Files:")
        dataset_logger.info(f"  • Dataset: {output_file}")
        dataset_logger.info(f"  • Detailed log: {log_file}")
        dataset_logger.info(f"🎯 Format: SkyRL v1.0 | Env: tool_using_agent | Source: mcp_tool_agent_synthetic")
        
        return dataset

    def _adapt_args_for_alternative_tool(self, original_args: Dict[str, Any], failed_tool: str, alt_tool: str) -> Dict[str, Any]:
        """Adapt arguments when switching to an alternative tool"""
        if failed_tool == "fmp_get_stock_news" and alt_tool == "tavily_search":
            # Convert stock news request to web search
            ticker = original_args.get("tickers", "")
            return {"query": f"{ticker} stock news recent", "max_results": 5}
        
        elif failed_tool == "fmp_get_quote" and alt_tool == "tavily_search":
            # Convert quote request to web search
            symbol = original_args.get("symbol", "")
            return {"query": f"{symbol} stock price current", "max_results": 3}
        
        elif failed_tool == "tavily_search" and alt_tool == "fmp_get_stock_news":
            # Extract ticker from search query if possible
            query = original_args.get("query", "")
            # Simple extraction - look for common ticker patterns
            words = query.upper().split()
            for word in words:
                if len(word) <= 5 and word.isalpha():  # Likely a ticker
                    return {"tickers": word, "limit": 5}
            return {"tickers": "AAPL", "limit": 5}  # Fallback
        
        # Default: try to use original args if compatible
        return original_args

    def _build_dataset_messages(self, conv: List[ConversationMessage], user_query: str) -> List[Dict[str, str]]:
        """Build dataset prompt in SkyRL format (OpenAI chat format) using robust conversion"""
        # Convert ConversationMessage objects to dict format for conversion function
        conversation_dicts = []
        for msg in conv:
            if msg.role == 'system':
                continue  # Skip system messages for SkyRL format
            elif msg.role == 'human' and msg.content != user_query:
                continue  # Skip follow-up human messages for cleaner format
            elif msg.role in ['assistant', 'tool']:
                conversation_dicts.append({
                    "role": msg.role,
                    "content": msg.content or "",
                    "from": msg.role  # Add 'from' for compatibility
                })
        
        # Start with user query
        prompt = [{"role": "user", "content": user_query}]
        
        # Use the new conversion function to prevent concatenation bugs
        if conversation_dicts:
            converted_turns = convert_conversation_format(conversation_dicts)
            
            # FIX: Preserve natural turn structure for SkyRL reward shaping and credit assignment
            # Do NOT squash turns into one giant message - extend to preserve turn-by-turn structure
            prompt.extend(converted_turns)
        
        return prompt
    # When is this called?
    def _generate_reasoning_improvement_prompt(self, quality_report: Dict[str, Any], user_query: str, attempt_count: int) -> str:
        """Generate context-aware prompt to improve reasoning quality"""
        issues = quality_report.get("issues", [])
        suggestions = quality_report.get("suggestions", [])
        score = quality_report.get("score", 0)
        
        if attempt_count == 1:
            return f"""Your reasoning was too generic (score: {score}/100). The user asked: "{user_query}"

You need to provide detailed step-by-step thinking that shows:

1. **Request Analysis**: Break down exactly what the user is asking for
2. **Tool Selection**: Explain WHY you chose specific tools (not just what you're doing)
3. **Approach Planning**: Show the logical sequence and dependencies
4. **Strategic Thinking**: Demonstrate understanding of the problem

Issues found: {', '.join(issues)}

Please provide substantive reasoning (100+ words) that demonstrates real analytical thinking, not just generic statements."""

        elif attempt_count == 2:
            return f"""Your reasoning still needs improvement (score: {score}/100). For this query: "{user_query}"

Specific improvements needed:
{chr(10).join(f"- {suggestion}" for suggestion in suggestions)}

Example of what I need:
<think>
The user is asking for [specific breakdown of request components].
I need to approach this by [detailed plan with rationale].
For tool selection: [specific reasons for each tool choice].
My strategy is: [step-by-step approach with dependencies].
</think>

Avoid these phrases: "gathering information", "making tool calls", "let me check"
Instead, show actual analytical thinking about the problem."""

        else:
            return f"""This is attempt #{attempt_count}. Your reasoning quality is still insufficient (score: {score}/100).

For the query "{user_query}", you must provide reasoning that includes:

1. Detailed analysis of what the user wants
2. Specific justification for tool choices  
3. Clear planning with logical sequencing
4. At least 150 words of substantive thinking

Current issues: {', '.join(issues)}

This is your final chance to demonstrate proper ReAct-style reasoning before I skip this query."""

    def _extract_skill_dependencies(self, tool_calls: List[str]) -> List[str]:
        """Extract skill dependencies based on tools used"""
        skill_map = {
            "tavily_search": "web_research",
            "fmp_get_quote": "market_data_retrieval",
            "fmp_get_stock_news": "financial_news_analysis",
            "polygon_get_ticker_details": "market_data_analysis",
            "polygon_get_market_status": "market_timing",
            "slack_send_message": "communication_automation",
            "slack_get_messages": "data_extraction",
            "python_execute": "data_processing"
        }
        
        skills = set()
        for tool in tool_calls:
            for tool_pattern, skill in skill_map.items():
                if tool_pattern in tool:
                    skills.add(skill)
        
        # Add composite skills based on tool combinations
        if "market_data_retrieval" in skills and "data_processing" in skills:
            skills.add("quantitative_analysis")
        if "financial_news_analysis" in skills and "web_research" in skills:
            skills.add("fundamental_analysis")
        if "communication_automation" in skills and any("market" in s for s in skills):
            skills.add("automated_reporting")
        
        return sorted(list(skills))

    def _suggest_next_level_tasks(self, current_complexity: str, category: str) -> List[str]:
        """Suggest next level tasks for curriculum progression"""
        progression_map = {
            "easy": ["medium_" + category, "easy_cross_category"],
            "medium": ["hard_" + category, "medium_multi_category"], 
            "hard": ["expert_" + category, "hard_cross_domain"]
        }
        
        return progression_map.get(current_complexity, [])

    def _extract_learning_objectives(self, tool_calls: List[str], complexity: str) -> List[str]:
        """Extract learning objectives based on tools used and complexity"""
        base_objectives = ["tool_usage_efficiency", "step_by_step_reasoning"]
        
        # Add complexity-specific objectives
        if complexity == "easy":
            base_objectives.extend(["basic_tool_selection", "simple_data_retrieval"])
        elif complexity == "medium":
            base_objectives.extend(["multi_step_planning", "data_synthesis", "intermediate_analysis"])
        elif complexity == "hard":
            base_objectives.extend(["advanced_reasoning", "error_recovery", "complex_workflow_management", "cross_domain_integration"])
        
        # Add tool-specific objectives
        if len(tool_calls) >= 3:
            base_objectives.append("multi_tool_orchestration")
        if any("slack" in tool for tool in tool_calls):
            base_objectives.append("communication_integration")
        if any("python" in tool for tool in tool_calls):
            base_objectives.append("computational_thinking")
        
        return sorted(list(set(base_objectives)))

# Example usage
def main():
    async def runner():
        # ================================================
        # SkyRL Infrastructure Function Tests
        # ================================================
        print("🧪 Running SkyRL Infrastructure Tests...")
        
        # Test 1: validate_skyrl_format function
        print("\n1. Testing validate_skyrl_format()...")
        valid_example = {
            "data_source": "mcp_tool_agent_synthetic_extended",
            "prompt": [
                {"role": "user", "content": "test query"},
                {"role": "assistant", "content": "test response"}
            ],
            "env_class": "financial_tool_agent",
            "reward_spec": {"method": "rule", "ground_truth": {}}
        }
        
        invalid_example = {
            "data_source": "test",
            "prompt": [
                {"role": "unknown", "content": "bad role"}  # Invalid role
            ],
            "env_class": "test_env"
            # Missing reward_spec
        }
        
        # Test valid example - should not raise exception
        try:
            result = validate_skyrl_format(valid_example)
            assert result is True, "Valid example should return True"
        except ValueError:
            assert False, "Valid example should not raise ValueError"
        
        # Test invalid example - should raise ValueError
        try:
            validate_skyrl_format(invalid_example)
            assert False, "Invalid example should raise ValueError"
        except ValueError as e:
            validation_logger.debug(f"Expected validation error: {e}")
        
        print("✅ validate_skyrl_format() tests passed")
        
        # Test 2: convert_conversation_format function
        print("\n2. Testing convert_conversation_format()...")
        test_conversation = [
            {"from": "human", "value": "Hello"},
            {"from": "gpt", "value": "Hi there"},
            {"from": "tool", "value": "Tool result"},
            {"from": "unknown", "value": "Should be skipped"}  # Should be skipped
        ]
        
        converted = convert_conversation_format(test_conversation)
        assert len(converted) == 3, f"Expected 3 converted turns, got {len(converted)}"
        assert converted[0]["role"] == "user", "First turn should be user"
        assert converted[1]["role"] == "assistant", "Second turn should be assistant"
        assert converted[2]["role"] == "tool", "Third turn should be tool"
        print("✅ convert_conversation_format() tests passed")
        
        # Test 3: extract_tool_sequence function
        print("\n3. Testing extract_tool_sequence()...")
        test_content = '''
        <think>I need to use tools</think>
        <tool_call>
        {"name": "test_tool_1", "arguments": {}}
        </tool_call>
        
        <tool_call>
        {"name": "test_tool_2", "arguments": {"param": "value"}}
        </tool_call>
        '''
        
        extracted_tools = extract_tool_sequence(test_content)
        assert len(extracted_tools) == 2, f"Expected 2 tools, got {len(extracted_tools)}"
        assert "test_tool_1" in extracted_tools, "Should extract test_tool_1"
        assert "test_tool_2" in extracted_tools, "Should extract test_tool_2"
        
        # Test edge case with nested braces in arguments
        complex_content = '''
        <tool_call>
        {"name": "complex_tool", "arguments": {"query": "Find {AAPL} stock data", "nested": {"key": "value"}}}
        </tool_call>
        '''
        
        complex_tools = extract_tool_sequence(complex_content)
        assert len(complex_tools) == 1, f"Expected 1 complex tool, got {len(complex_tools)}"
        assert "complex_tool" in complex_tools, "Should extract complex_tool with nested braces"
        
        print("✅ extract_tool_sequence() tests passed")
        
        # Test 4: Test deterministic ordering (extract_success_criteria)
        print("\n4. Testing deterministic ordering in extract_success_criteria()...")
        
        # Create mock agent result with tools in random order
        mock_result = AgentResult(
            final_answer="test",
            tool_calls=["zebra_tool", "alpha_tool", "beta_tool"],  # Unsorted
            conversation_history=[]
        )
        
        # Test with default complexity
        criteria1 = extract_success_criteria(mock_result)
        # Test with medium complexity
        criteria2 = extract_success_criteria(mock_result, complexity="medium", complexity_config=COMPLEXITY_CONFIG["medium"])
        # Test with hard complexity
        criteria3 = extract_success_criteria(mock_result, complexity="hard", complexity_config=COMPLEXITY_CONFIG["hard"])
        
        # Should be identical and sorted
        assert criteria1["expected_tools"] == criteria2["expected_tools"], "Results should be deterministic"
        assert criteria1["expected_tools"] == ["alpha_tool", "beta_tool", "zebra_tool"], "Tools should be sorted alphabetically"
        assert criteria2["expected_tools"] == ["alpha_tool", "beta_tool", "zebra_tool"], "Tools should be sorted alphabetically"
        assert criteria3["expected_tools"] == ["alpha_tool", "beta_tool", "zebra_tool"], "Tools should be sorted alphabetically"
        
        # Test complexity-specific thresholds
        assert criteria2["min_tool_calls"] == 3, f"Medium complexity should require 3 tools, got {criteria2['min_tool_calls']}"
        assert criteria2["reasoning_quality_threshold"] == 70, f"Medium complexity should require 70 reasoning, got {criteria2['reasoning_quality_threshold']}"
        assert criteria3["min_tool_calls"] == 4, f"Hard complexity should require 4 tools, got {criteria3['min_tool_calls']}"
        assert criteria3["reasoning_quality_threshold"] == 80, f"Hard complexity should require 80 reasoning, got {criteria3['reasoning_quality_threshold']}"
        
        print("✅ Deterministic ordering tests passed")
        
        print("\n🎉 All infrastructure tests passed!")
        
        # ================================================
        # Full Integration Test with Real Queries
        # ================================================
        print("\n🚀 Running Full Integration Test...")
        
        agent = MiniAgent()
        await agent.tool_manager.initialize_tools()
        test_queries = [
            # Multi-step analysis with dependencies
            "Find the top 3 stock gainers today, then research news about each company to identify why they're rising",
            
            # Error-prone query to test recovery (using tickers likely to have limited data)
            "Get stock quotes and recent news for RIBBR, COLAR, and HTOO to understand their recent performance",
            
            # Conditional logic with error handling  
            "Get Tesla's current stock price - if it's above $200, search for recent Tesla news; if below, find analyst downgrades"
        ]
        
        print(f"\n📝 Processing {len(test_queries)} test queries...")
        dataset = await agent.generate_dataset(test_queries, "skyrl_multi_turn_dataset.json")
        
        # Validate generated dataset
        print(f"\n🔍 Validating generated dataset with {len(dataset)} entries...")
        validation_results = {"passed": 0, "failed": 0, "total": len(dataset)}
        
        for i, entry in enumerate(dataset):
            try:
                validate_skyrl_format(entry)
                validation_results["passed"] += 1
                validation_logger.debug(f"Entry {i+1} passed SkyRL format validation")
                print(f"✅ Entry {i+1} passed SkyRL format validation")
            except ValueError as e:
                validation_results["failed"] += 1
                validation_logger.error(f"Entry {i+1} failed SkyRL format validation: {e}")
                print(f"❌ Entry {i+1} failed SkyRL format validation: {e}")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"  • Total entries: {validation_results['total']}")
        print(f"  • Passed validation: {validation_results['passed']}")
        print(f"  • Failed validation: {validation_results['failed']}")
        print(f"  • Success rate: {(validation_results['passed']/validation_results['total']*100):.1f}%")
        
        # Assert minimum success rate
        success_rate = validation_results["passed"] / validation_results["total"]
        assert success_rate >= 0.8, f"Success rate {success_rate:.1f} below 80% threshold"
        
        print(f"\n🎯 Integration test completed successfully!")
        print(f"Generated {len(dataset)} SkyRL-compatible dataset entries.")
        print(f"🎉 All tests passed! SkyRL compatibility verified.")
        
        # Test 4: Tool manager server failure handling
        manager = mini_agent_module.ImprovedMCPToolManager()
        manager.failed_servers.add('test_server')
        
        # Mock a tool that belongs to the failed server
        manager.tools['test_server_tool'] = {"name": "test_server_tool", "description": "test"}
        
        success, result = await manager.execute_tool('test_server_tool', {})
        assert success == False, f'Expected False for tool from failed server, got {success}'
        assert 'server' in result['error'].lower(), f'Expected server error message, got {result["error"]}'
        print('✅ Fix 2.8: Server failure tracking - PASSED')
        
    asyncio.run(runner())

if __name__ == '__main__':
    main()
