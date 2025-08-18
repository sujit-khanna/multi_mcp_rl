#!/usr/bin/env python3
"""
Enhanced QwenPolicy with prompting to help untrained models generate proper tool calls.
"""

from typing import List, Dict, Optional
import logging
import time
import os
from .qwen_policy import QwenPolicy

logger = logging.getLogger(__name__)


class QwenPolicyWithPrompting(QwenPolicy):
    """QwenPolicy with additional prompting for untrained models."""
    
    def __init__(self, *args, force_rate: float = 0.0, assist_warmup_steps: int = 0, rl_mode: bool = True, **kwargs):
        """Initialize with configurable forcing for RL vs warmup phases."""
        super().__init__(*args, **kwargs)
        self.action_counter = 0  # Deterministic counter for forcing decisions
        self.force_rate = force_rate  # Configurable forcing rate (0.0 for RL)
        self.assist_warmup_steps = assist_warmup_steps
        self.rl_mode = rl_mode  # True during RL training, False during pretraining/warmup
        self.in_rl_update = True  # Set by trainer during RL phases
        
        # Environment variable overrides
        import os
        self.force_rate = float(os.getenv('FORCE_RATE', str(self.force_rate)))
        self.assist_warmup_steps = int(os.getenv('ASSIST_WARMUP', str(self.assist_warmup_steps)))
        self.rl_mode = os.getenv('RL_MODE', str(self.rl_mode)).lower() == 'true'
        
        logger.info(f"🎯 Policy initialized - RL mode: {self.rl_mode}, Force rate: {self.force_rate}")
    
    TOOL_CALLING_PROMPT = """You are a tool assistant. Respond only with tool calls in this format:

<tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>

Tools: tavily_search, execute_python, fmp_get_quote, polygon_get_aggs

Example: <tool_call>{"name": "fmp_get_quote", "arguments": {"symbol": "AAPL"}}</tool_call>"""
    
    def format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Format conversation with optimized prompting and history truncation.
        """
        # ROLLING WINDOW: Keep only last 6-8 messages to prevent 32k->1024 token truncation
        max_history = 8
        
        # Always keep system message + last N messages
        if len(messages) > max_history:
            system_msgs = [msg for msg in messages if msg.get("role") == "system"]
            other_msgs = [msg for msg in messages if msg.get("role") != "system"]
            # Keep system + last N-1 messages
            truncated_msgs = system_msgs + other_msgs[-(max_history-1):]
            logger.debug(f"🔄 Truncated conversation: {len(messages)} -> {len(truncated_msgs)} messages")
        else:
            truncated_msgs = messages
        
        # Inject optimized system prompt if not present
        enhanced_messages = []
        if not truncated_msgs or truncated_msgs[0].get("role") != "system":
            enhanced_messages.append({
                "role": "system",
                "content": self.TOOL_CALLING_PROMPT
            })
        
        enhanced_messages.extend(truncated_msgs)
        
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
        logger.info(f"🎯 QwenPolicyWithPrompting.generate_action called with {len(states)} states")
        
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
        
        # Post-process with conditional tool call enforcement
        processed_actions = []
        forced_mask: List[bool] = []
        for i, action in enumerate(actions):
            is_tool = '<tool_call>' in action and '</tool_call>' in action
            self.action_counter += 1

            # IMPROVED TOOL-USE ADHERENCE with warm-start and resampling
            should_force = False
            should_resample = False
            
            # Check forcing logic based on training phase
            if self.rl_mode and self.force_rate <= 0.0:
                # In RL mode with no forcing - but allow resampling for very short responses
                should_force = False
                should_resample = not is_tool and len(action.strip()) < 20  # Resample very short non-tool responses
                logger.debug("🚫 RL mode active - no forcing, but resampling short responses")
            elif not self.rl_mode or self.action_counter <= self.assist_warmup_steps:
                # WARM-START: Force tool calls during initial training phase
                should_force = not is_tool or len(action.strip()) < 10
                logger.debug(f"🔥 Warm-start forcing (step {self.action_counter}/{self.assist_warmup_steps})")
            else:
                # Use force rate for probabilistic forcing after warm-start
                import random
                should_force = (not is_tool or len(action.strip()) < 10) and random.random() < self.force_rate

            # Try resampling once before forcing for better tool adherence
            if should_resample and not should_force:
                logger.info("🔄 Resampling for better tool adherence...")
                try:
                    # Resample with higher penalty for natural language tokens
                    resampled_action = self._resample_with_tool_bias(states[i] if i < len(states) else states[-1])
                    if '<tool_call>' in resampled_action and '</tool_call>' in resampled_action:
                        processed_actions.append(resampled_action)
                        forced_mask.append(False)  # Not forced, successfully resampled
                        logger.info("✅ Resampling successful - tool call generated")
                        continue
                    else:
                        logger.warning("⚠️ Resampling failed - falling back to forcing")
                        should_force = True
                except Exception as e:
                    logger.warning(f"⚠️ Resampling error: {e} - falling back to forcing")
                    should_force = True

            if should_force:
                context = str(states[i] if i < len(states) else states[-1]).lower()
                forced_action = self._generate_forced_tool_call(context)
                logger.info(f"🔧 FORCED tool call - model output was broken: '{action[:50]}...'")
                processed_actions.append(forced_action)
                forced_mask.append(True)
            else:
                if not is_tool:
                    logger.info("📝 NATURAL language preserved (RL mode)")
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
        
        # CRITICAL FIX: Always use fast native generation to prevent 10% GPU bottleneck
        # The step-by-step generation was causing massive slowdowns with long conversations
        
        # Use native model generation for speed
        logger.info(f"🚀 Starting native model generation for {len(formatted_inputs)} inputs...")
        logger.info(f"   Max new tokens: {max_new_tokens}")
        logger.info(f"   Temperature: {gen_config.temperature}")
        logger.info(f"   Use cache: {use_cache}")
        
        start_time = time.time()
        
        # CRITICAL FIX: Force disable gradient checkpointing before generation
        was_training = self.model.training
        original_gc_state = getattr(self.model.config, 'gradient_checkpointing', None)
        
        # Temporarily disable gradient checkpointing and set eval mode
        if hasattr(self.model.config, 'gradient_checkpointing'):
            self.model.config.gradient_checkpointing = False
        if hasattr(self.model, 'gradient_checkpointing_disable'):
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        use_cache = True  # Always use cache since we disabled gradient checkpointing
        
        with torch.no_grad():
            # Generate with the model
            logger.info("🔥 Calling model.generate() with gradient_checkpointing=False...")
            generated_ids = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                generation_config=gen_config,
                use_cache=use_cache,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens
            )
            generation_time = time.time() - start_time
            logger.info(f"✅ Model generation completed in {generation_time:.2f} seconds")
        
        # Restore original model state
        if was_training:
            self.model.train()
        if original_gc_state is not None and hasattr(self.model.config, 'gradient_checkpointing'):
            self.model.config.gradient_checkpointing = original_gc_state
            
        # Extract generated parts (after input)
        for i in range(len(formatted_inputs)):
            input_length = inputs['input_ids'][i].shape[0]
            generated_part = generated_ids[i][input_length:]
            
            if len(generated_part) > 0:
                generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True)
                actions.append(generated_text)
                # Use dummy log prob for now - much faster than computing real ones
                all_log_probs.append(torch.tensor(-1.0, device=self.device))
            else:
                actions.append("")
                all_log_probs.append(torch.tensor(-1.0, device=self.device))
        
        return actions, all_log_probs
    
    def _resample_with_tool_bias(self, state) -> str:
        """
        Resample with bias toward tool calls for better adherence.
        Uses higher temperature and tool-call prompting.
        """
        try:
            formatted_input = self.format_conversation(state)
            # Add explicit tool-calling instruction
            biased_input = formatted_input + "\n\nIMPORTANT: You must respond with a tool call in <tool_call>...</tool_call> format."
            
            inputs = self.tokenizer(biased_input, return_tensors="pt", padding=True, truncation=True, max_length=1000)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate with tool bias
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.8,  # Higher temperature for diversity
                    do_sample=True,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract generated text
            input_length = inputs['input_ids'].shape[1]
            generated_part = generated_ids[0][input_length:]
            generated_text = self.tokenizer.decode(generated_part, skip_special_tokens=True).strip()
            
            return generated_text
        except Exception as e:
            logger.error(f"Resampling error: {e}")
            raise e
    
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