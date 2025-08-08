#!/usr/bin/env python3
"""
Data Utilities for SkyRL Tool Agent
====================================

Converts tool-use conversation datasets to SkyRL-compatible format
for GRPO training. Handles multi-turn conversations with tool calls
and creates appropriate reward specifications.

Author: SkyRL Tool Agent Team
Date: 2025-07-22
"""

import json
import re
import os
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import datasets
from datasets import Dataset
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_complexity(user_query: str) -> str:
    """Extract complexity level from user query"""
    complexity_match = re.search(r'\[complexity:(\w+)\]', user_query)
    if complexity_match:
        return complexity_match.group(1)
    return "medium"  # default


def extract_tool_sequence(conversation: List[Dict[str, str]]) -> List[str]:
    """Extract the sequence of tools used from conversation"""
    tool_sequence = []
    
    for turn in conversation:
        # Skip system and human messages
        if turn.get("from") in ["system", "human"]:
            continue
            
        # For assistant messages, look for function calls
        if turn.get("from") == "gpt" or turn.get("role") == "assistant":
            content = turn.get("value", turn.get("content", ""))
            
            # Look for JSON function calls
            if "function_call" in content:
                try:
                    # Extract function call JSON
                    start = content.find('{"function_call"')
                    if start != -1:
                        end = content.find('}}}', start) + 3
                        if end > start:
                            func_json = json.loads(content[start:end])
                            if "function_call" in func_json:
                                tool_name = func_json["function_call"].get("name")
                                if tool_name:
                                    tool_sequence.append(tool_name)
                except (json.JSONDecodeError, KeyError):
                    pass
                    
        # For tool responses
        elif turn.get("from") == "tool":
            # Tool responses follow function calls, already captured above
            pass
    
    return tool_sequence


def extract_final_state(conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract the final state information from conversation"""
    final_state = {
        "task_completed": False,
        "tools_executed": 0,
        "errors_encountered": 0,
        "final_response_type": None
    }
    
    tool_count = 0
    error_count = 0
    
    for turn in conversation:
        content = turn.get("value", turn.get("content", ""))
        
        # Count tool executions
        if turn.get("from") == "tool" or "Function result:" in content:
            tool_count += 1
            
            # Check for errors in tool results
            if "error" in content.lower() or "failed" in content.lower():
                error_count += 1
        
        # Check if this is the final assistant message
        if turn.get("from") in ["gpt", "assistant"]:
            # If this message doesn't contain function_call, it's likely a final answer
            if "function_call" not in content:
                final_state["task_completed"] = True
                final_state["final_response_type"] = "answer"
    
    final_state["tools_executed"] = tool_count
    final_state["errors_encountered"] = error_count
    
    # Check for agent crashes or failures
    if any("AGENT_CRASHED" in turn.get("value", turn.get("content", "")) 
           for turn in conversation):
        final_state["task_completed"] = False
        final_state["final_response_type"] = "crash"
    
    return final_state


def extract_success_criteria(example: Dict[str, Any]) -> Dict[str, Any]:
    """Extract success criteria from the example"""
    criteria = {
        "must_use_tools": [],
        "expected_outcome": None,
        "validation_rules": []
    }
    
    # Extract from user query
    user_query = example.get("user_query", "")
    
    # Common patterns for required tools
    if "stock price" in user_query.lower():
        criteria["must_use_tools"].append("fmp_get_quote")
    if "send" in user_query.lower() and "slack" in user_query.lower():
        criteria["must_use_tools"].append("send_slack_message")
    if "search" in user_query.lower():
        criteria["must_use_tools"].append("tavily_search")
    if "python" in user_query.lower() or "plot" in user_query.lower():
        criteria["must_use_tools"].append("execute_python")
    
    # Extract expected tools from the actual conversation
    actual_tools = extract_tool_sequence(example.get("conversation", []))
    if actual_tools:
        criteria["expected_tools"] = list(set(actual_tools))
    
    return criteria


def convert_conversation_format(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Convert conversation from old format to new format"""
    converted = []
    
    for turn in conversation:
        # Handle different formats
        from_field = turn.get("from")
        role_field = turn.get("role")
        
        # Determine role
        if from_field == "gpt" or role_field == "assistant":
            role = "assistant"
        elif from_field == "human" or role_field == "user":
            role = "user"
        elif from_field == "system" or role_field == "system":
            role = "system"
        elif from_field == "tool":
            role = "tool"
        else:
            role = "unknown"
        
        # Get content
        content = turn.get("value", turn.get("content", ""))
        
        converted.append({
            "role": role,
            "content": content
        })
    
    return converted


def process_single_example(example: Dict[str, Any], idx: int, split: str) -> Dict[str, Any]:
    """Process a single example to SkyRL format"""
    
    # Extract user query
    user_query = example.get("user_query", "")
    
    # Extract tool sequence and final state
    conversation = example.get("conversation", [])
    tool_sequence = extract_tool_sequence(conversation)
    final_state = extract_final_state(conversation)
    success_criteria = extract_success_criteria(example)
    
    # Create ground truth
    ground_truth = {
        "tool_sequence": tool_sequence,
        "final_state": final_state,
        "success_criteria": success_criteria,
        "conversation_length": len(conversation),
        "expected_tools": success_criteria.get("expected_tools", tool_sequence)
    }
    
    # Extract complexity
    complexity = extract_complexity(user_query)
    
    # Build SkyRL format
    skyrl_example = {
        "data_source": "multi_tool_dataset",
        "prompt": [{"role": "user", "content": user_query}],
        "env_class": "multi_tool_environment",
        "reward_spec": {
            "method": "rule",
            "ground_truth": ground_truth
        },
        "extra_info": {
            "task_id": example.get("task_id", f"{split}_{idx}"),
            "complexity": complexity,
            "category": example.get("category", "general"),
            "split": split,
            "index": idx,
            "original_conversation": convert_conversation_format(conversation)
        }
    }
    
    return skyrl_example


def convert_dataset_to_skyrl(
    input_path: str,
    output_dir: str,
    split: str = "train",
    sample_size: Optional[int] = None
) -> None:
    """Convert a dataset file to SkyRL format"""
    
    logger.info(f"Loading dataset from {input_path}")
    
    # Load the dataset
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle different dataset structures
    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(f"Unknown dataset structure in {input_path}")
    
    logger.info(f"Found {len(examples)} examples")
    
    # Sample if requested
    if sample_size and sample_size < len(examples):
        import random
        random.seed(42)
        examples = random.sample(examples, sample_size)
        logger.info(f"Sampled {sample_size} examples")
    
    # Process examples
    processed_examples = []
    failed_count = 0
    
    for idx, example in enumerate(examples):
        try:
            processed = process_single_example(example, idx, split)
            processed_examples.append(processed)
        except Exception as e:
            logger.warning(f"Failed to process example {idx}: {e}")
            failed_count += 1
    
    logger.info(f"Successfully processed {len(processed_examples)} examples, {failed_count} failed")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as both JSON and Parquet
    output_json = os.path.join(output_dir, f"{split}.json")
    output_parquet = os.path.join(output_dir, f"{split}.parquet")
    
    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(processed_examples, f, indent=2)
    logger.info(f"Saved JSON to {output_json}")
    
    # Convert to Dataset and save as Parquet
    # Convert complex nested structures to JSON strings for Parquet compatibility
    parquet_examples = []
    for example in processed_examples:
        parquet_example = example.copy()
        # Convert lists and dicts to JSON strings
        parquet_example["prompt"] = json.dumps(example["prompt"])
        parquet_example["reward_spec"] = json.dumps(example["reward_spec"])
        parquet_example["extra_info"] = json.dumps(example["extra_info"])
        parquet_examples.append(parquet_example)
    
    dataset = Dataset.from_list(parquet_examples)
    dataset.to_parquet(output_parquet)
    logger.info(f"Saved Parquet to {output_parquet}")
    
    # Print sample
    if processed_examples:
        logger.info("\nSample converted example:")
        print(json.dumps(processed_examples[0], indent=2))


def validate_skyrl_format(data_path: str) -> bool:
    """Validate that the converted data matches SkyRL format requirements"""
    
    logger.info(f"Validating SkyRL format for {data_path}")
    
    # Load data
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
    elif data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
        data = df.to_dict('records')
        # Parse JSON strings back to objects for validation
        for example in data:
            if isinstance(example.get("prompt"), str):
                example["prompt"] = json.loads(example["prompt"])
            if isinstance(example.get("reward_spec"), str):
                example["reward_spec"] = json.loads(example["reward_spec"])
            if isinstance(example.get("extra_info"), str):
                example["extra_info"] = json.loads(example["extra_info"])
    else:
        raise ValueError(f"Unknown file format: {data_path}")
    
    # Check required fields
    required_fields = ["data_source", "prompt", "env_class", "reward_spec"]
    valid_count = 0
    invalid_count = 0
    
    for idx, example in enumerate(data):
        try:
            # Check top-level fields
            for field in required_fields:
                assert field in example, f"Missing field: {field}"
            
            # Check prompt structure
            assert isinstance(example["prompt"], list), "Prompt must be a list"
            assert len(example["prompt"]) > 0, "Prompt cannot be empty"
            assert "role" in example["prompt"][0], "Prompt missing role"
            assert "content" in example["prompt"][0], "Prompt missing content"
            
            # Check reward spec
            assert "method" in example["reward_spec"], "Reward spec missing method"
            assert "ground_truth" in example["reward_spec"], "Reward spec missing ground_truth"
            
            valid_count += 1
            
        except AssertionError as e:
            logger.warning(f"Example {idx} failed validation: {e}")
            invalid_count += 1
    
    logger.info(f"Validation complete: {valid_count} valid, {invalid_count} invalid")
    
    return invalid_count == 0


def main():
    """Main conversion script"""
    parser = argparse.ArgumentParser(description="Convert tool-use datasets to SkyRL format")
    parser.add_argument("--input", type=str, required=True, help="Input dataset path")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument("--sample", type=int, help="Sample size (for testing)")
    parser.add_argument("--validate", action="store_true", help="Validate output format")
    
    args = parser.parse_args()
    
    # Convert dataset
    convert_dataset_to_skyrl(
        input_path=args.input,
        output_dir=args.output_dir,
        split=args.split,
        sample_size=args.sample
    )
    
    # Validate if requested
    if args.validate:
        output_parquet = os.path.join(args.output_dir, f"{args.split}.parquet")
        if validate_skyrl_format(output_parquet):
            logger.info("✅ Dataset is valid SkyRL format")
        else:
            logger.error("❌ Dataset validation failed")


if __name__ == "__main__":
    main()