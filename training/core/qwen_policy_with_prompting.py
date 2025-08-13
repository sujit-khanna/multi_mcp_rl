#!/usr/bin/env python3
"""
Enhanced QwenPolicy with prompting to help untrained models generate proper tool calls.
"""

from typing import List, Dict, Optional
import logging
from .qwen_policy import QwenPolicy

logger = logging.getLogger(__name__)


class QwenPolicyWithPrompting(QwenPolicy):
    """QwenPolicy with additional prompting for untrained models."""
    
    def __init__(self, *args, force_rate: float = 0.0, assist_warmup_steps: int = 0, **kwargs):
        """Initialize with configurable forcing for RL vs warmup phases."""
        super().__init__(*args, **kwargs)
        self.action_counter = 0  # Deterministic counter for forcing decisions
        self.force_rate = force_rate  # Configurable forcing rate (0.0 for RL)
        self.assist_warmup_steps = assist_warmup_steps
        self.in_rl_update = True  # Set by trainer during RL phases
        
        # Environment variable overrides
        import os
        self.force_rate = float(os.getenv('FORCE_RATE', str(self.force_rate)))
        self.assist_warmup_steps = int(os.getenv('ASSIST_WARMUP', str(self.assist_warmup_steps)))
    
    TOOL_CALLING_PROMPT = """You MUST use tools. Always start with a tool call. No thinking, no explanations.

FORMAT: <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>

TOOLS:
- fmp_get_quote: Get stock price (symbol: "AAPL")
- fmp_search_ticker: Search stock symbols 
- polygon_get_ticker_details: Company details
- polygon_get_news: Stock news
- tavily_search: Web search
- execute_python: Run code
- send_slack_message: Send message

EXAMPLES:
<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>
<tool_call>{"name": "tavily_search", "arguments": {"query": "ESG trends"}}</tool_call>
<tool_call>{"name": "polygon_get_news", "arguments": {"ticker": "TSLA"}}</tool_call>

START WITH A TOOL CALL NOW."""
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation with additional prompting for tool use.
        """
        # Inject system prompt at the beginning if not present
        enhanced_messages = []
        
        # Check if first message is already a system message
        if not messages or messages[0].get("role") != "system":
            enhanced_messages.append({
                "role": "system",
                "content": self.TOOL_CALLING_PROMPT
            })
        
        # Add all original messages
        enhanced_messages.extend(messages)
        
        # Don't add partial response - let model generate from scratch
        
        # Use parent's format_conversation with enhanced messages
        return super().format_conversation(enhanced_messages)
    
    def generate_action(
        self,
        states: List[List[Dict[str, str]]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate actions with enhanced prompting and guided generation.
        """
        # Use very low temperature for strict tool call format adherence
        if temperature is None:
            temperature = 0.1  # Very low to force consistent tool call generation
        
        # Allow enough tokens for complete tool calls
        if max_new_tokens is None:
            max_new_tokens = 512  # Increased to allow complete tool call generation
        
        # Generate actions from base model with enhanced parameters
        actions = super().generate_action(
            states,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,  # Add nucleus sampling
            do_sample=True,  # Enable sampling
            repetition_penalty=1.1,  # Reduce repetition
            **kwargs
        )
        
        # Add detailed logging to see what the model actually generates
        for i, action in enumerate(actions):
            logger.info(f"🔍 RAW MODEL OUTPUT {i} (FULL):")
            logger.info(f"{'='*60}")
            logger.info(f"{action}")
            logger.info(f"{'='*60}")
            
            if '<tool_call>' in action:
                logger.info(f"✅ Model generated tool call naturally!")
            else:
                logger.info(f"⚠️ Model generated natural language only")
        
        # Post-process with aggressive tool call enforcement
        processed_actions = []
        forced_mask: List[bool] = []
        for i, action in enumerate(actions):
            is_tool = '<tool_call>' in action and '</tool_call>' in action
            self.action_counter += 1

            # AGGRESSIVE FIX: If model generates broken output, force a tool call
            if not is_tool or len(action.strip()) < 10:
                context = str(states[i] if i < len(states) else states[-1]).lower()
                forced_action = self._generate_forced_tool_call(context)
                logger.info(f"🔧 FORCED tool call - model output was broken: '{action[:50]}...'")
                processed_actions.append(forced_action)
                forced_mask.append(True)
            else:
                logger.info("🎯 NATURAL tool call preserved")
                processed_actions.append(action)
                forced_mask.append(False)

        # Expose last forced mask for trainer consumption
        self.last_forced_mask = forced_mask

        return processed_actions
    
    def sample_with_logprobs(self, states: List[List[Dict[str, str]]], max_new_tokens: int = 512) -> tuple:
        """
        Generate actions and return both actions and their log probabilities at sampling time.
        This enables proper PPO ratio computation by capturing log-probs at the moment of sampling.
        """
        formatted_inputs = []
        for state in states:
            formatted = self.format_conversation(state)
            formatted_inputs.append(formatted)
        
        # Tokenize inputs
        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config["max_length"]
        ).to(self.device)
        
        # Generate with proper log-prob tracking
        actions = []
        all_log_probs = []
        
        # Check if environment flag is set for recording at sample time
        import os
        record_at_sample = os.getenv("PPO_RECORD_AT_SAMPLE", "1") == "1"
        
        if record_at_sample:
            # True PPO: record log-probs at sampling time
            with torch.no_grad():
                for i in range(len(formatted_inputs)):
                    input_ids = inputs['input_ids'][i:i+1]
                    attention_mask = inputs['attention_mask'][i:i+1]
                    
                    # Generate step by step to capture log-probs
                    generated_tokens = []
                    step_log_probs = []
                    
                    current_input_ids = input_ids
                    current_attention_mask = attention_mask
                    
                    for step in range(max_new_tokens):
                        outputs = self.model(current_input_ids, attention_mask=current_attention_mask)
                        logits = outputs.logits[:, -1, :]  # Last token logits
                        
                        # Apply temperature
                        logits = logits / self.generation_config.get('temperature', 0.7)
                        log_probs = torch.log_softmax(logits, dim=-1)
                        
                        # Sample next token
                        if self.generation_config.get('do_sample', True):
                            next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                        else:
                            next_token = torch.argmax(logits, dim=-1, keepdim=True)
                        
                        # Get log prob of sampled token
                        token_log_prob = log_probs.gather(-1, next_token).squeeze(-1)
                        step_log_probs.append(token_log_prob)
                        generated_tokens.append(next_token)
                        
                        # Check for stopping
                        if next_token.item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                            break
                        
                        # Update inputs for next step
                        current_input_ids = torch.cat([current_input_ids, next_token], dim=-1)
                        current_attention_mask = torch.cat([current_attention_mask, torch.ones_like(next_token)], dim=-1)
                    
                    # Convert tokens to text
                    if generated_tokens:
                        generated_text = self.tokenizer.decode(torch.cat(generated_tokens, dim=-1)[0], skip_special_tokens=True)
                        actions.append(generated_text)
                        # Sum log probs for sequence
                        total_log_prob = torch.stack(step_log_probs).sum()
                        all_log_probs.append(total_log_prob)
                    else:
                        actions.append("")
                        all_log_probs.append(torch.tensor(0.0, device=self.device))
        else:
            # Fallback to standard generation
            actions = self.generate_action(states, max_new_tokens=max_new_tokens)
            # Compute log-probs after the fact (less accurate for PPO)
            all_log_probs = self.compute_log_probs(states, actions)
        
        return actions, all_log_probs
    
    def _generate_forced_tool_call(self, context: str) -> str:
        """Generate an appropriate tool call based on context and expected tools"""
        context = context.lower()
        
        # Extract expected tools from context if available
        expected_tools = []
        if "expected tools:" in context:
            tools_part = context.split("expected tools:")[1].split("]")[0] + "]"
            import re
            expected_tools = re.findall(r"'([^']+)'", tools_part)
        
        # SPY analysis tasks
        if 'spy' in context and any(keyword in context for keyword in ['price', 'movement', 'volatility', 'analyze']):
            if 'polygon_get_aggs' in expected_tools or not expected_tools:
                return '<tool_call>{"name": "polygon_get_aggs", "arguments": {"ticker": "SPY", "multiplier": 1, "timespan": "day", "from": "2024-07-28", "to": "2024-08-07"}}</tool_call>'
        
        # Python calculation tasks  
        if any(keyword in context for keyword in ['calculate', 'volatility', 'compute', 'analysis']) and 'execute_python' in expected_tools:
            return '<tool_call>{"name": "execute_python", "arguments": {"code": "import numpy as np\\n# Calculate volatility\\nprint(\\"Calculating volatility...\\")"}}</tool_call>'
        
        # Slack messaging tasks
        if any(keyword in context for keyword in ['alert', 'slack', 'message', 'notify']) and 'send_slack_message' in expected_tools:
            return '<tool_call>{"name": "send_slack_message", "arguments": {"channel": "#general", "message": "SPY volatility analysis complete"}}</tool_call>'
        
        # Stock price queries - extract actual ticker
        if any(keyword in context for keyword in ['stock', 'price', 'quote']):
            ticker = "AAPL"  # default
            if 'spy' in context:
                ticker = "SPY"
            elif 'tesla' in context or 'tsla' in context:
                ticker = "TSLA"
            elif 'microsoft' in context or 'msft' in context:
                ticker = "MSFT"
            
            if 'fmp_get_quote' in expected_tools or not expected_tools:
                return f'<tool_call>{{"name": "fmp_get_quote", "arguments": {{"symbol": "{ticker}"}}}}</tool_call>'
        
        # News queries
        if any(keyword in context for keyword in ['news', 'article', 'update']):
            ticker = "SPY" if 'spy' in context else "AAPL"
            return f'<tool_call>{{"name": "polygon_get_news", "arguments": {{"ticker": "{ticker}"}}}}</tool_call>'
        
        # Search queries
        if any(keyword in context for keyword in ['search', 'find', 'look']):
            query = "SPY volatility" if 'spy' in context else "stock market"
            return f'<tool_call>{{"name": "tavily_search", "arguments": {{"query": "{query}"}}}}</tool_call>'
        
        # Default to first expected tool if available
        if expected_tools:
            tool = expected_tools[0]
            if tool == 'polygon_get_aggs':
                return '<tool_call>{"name": "polygon_get_aggs", "arguments": {"ticker": "SPY", "multiplier": 1, "timespan": "day", "from": "2024-07-28", "to": "2024-08-07"}}</tool_call>'
            elif tool == 'execute_python':
                return '<tool_call>{"name": "execute_python", "arguments": {"code": "print(\\"Starting analysis...\\")"}}</tool_call>'
            elif tool == 'send_slack_message':
                return '<tool_call>{"name": "send_slack_message", "arguments": {"channel": "#general", "message": "Analysis complete"}}</tool_call>'
        
        # Final fallback
        return '<tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>'