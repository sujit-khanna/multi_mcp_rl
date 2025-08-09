#!/usr/bin/env python3
"""
Tool Validator - Preflight check for tool availability vs expected tools
=========================================================================

This module provides utilities to validate that all expected tools are available
in the MCP tool manager before training starts, helping prevent tool call failures
during training that degrade rewards.

Author: SkyRL Tool Agent Team
Date: 2024-08-09
"""

import asyncio
import logging
import json
from typing import Dict, List, Set, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolValidator:
    """Validates tool availability against expected tools"""
    
    def __init__(self):
        self.expected_tools = set()
        self.available_tools = set()
        self.missing_tools = set()
        self.unexpected_tools = set()
    
    def load_expected_tools_from_data(self, data_path: str) -> Set[str]:
        """Extract expected tools from training data"""
        expected_tools = set()
        
        try:
            if Path(data_path).exists():
                with open(data_path, 'r') as f:
                    data = json.load(f)
                
                # Extract tool names from training data
                if isinstance(data, list):
                    for item in data:
                        # Look for tool calls in conversations
                        if 'messages' in item:
                            for message in item['messages']:
                                if message.get('role') == 'assistant' and 'tool_calls' in message:
                                    for tool_call in message['tool_calls']:
                                        if 'function' in tool_call:
                                            expected_tools.add(tool_call['function']['name'])
                        
                        # Look for tools in task definitions
                        if 'tools' in item:
                            for tool in item['tools']:
                                if isinstance(tool, dict) and 'name' in tool:
                                    expected_tools.add(tool['name'])
                                elif isinstance(tool, str):
                                    expected_tools.add(tool)
        
        except Exception as e:
            logger.warning(f"Could not extract expected tools from {data_path}: {e}")
        
        return expected_tools
    
    def load_expected_tools_from_config(self, config_path: str) -> Set[str]:
        """Load expected tools from configuration file"""
        expected_tools = set()
        
        try:
            if Path(config_path).exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Look for tool lists in config
                if 'expected_tools' in config:
                    expected_tools.update(config['expected_tools'])
                
                if 'tools' in config:
                    if isinstance(config['tools'], list):
                        expected_tools.update(config['tools'])
                    elif isinstance(config['tools'], dict):
                        expected_tools.update(config['tools'].keys())
        
        except Exception as e:
            logger.warning(f"Could not load expected tools from config {config_path}: {e}")
        
        return expected_tools
    
    def add_common_expected_tools(self) -> Set[str]:
        """Add commonly used tools based on the error messages mentioned"""
        return {
            'send_slack_message',
            'fmp_get_change_percent', 
            'fmp_get_previous_close',
            'polygon_get_aggs',
            'execute_python',  # Correct name from python server
            'tavily_search'
        }
    
    async def get_available_tools(self, tool_manager) -> Set[str]:
        """Get available tools from tool manager"""
        available_tools = set()
        
        try:
            # Initialize tool manager if needed
            if hasattr(tool_manager, 'initialize'):
                await tool_manager.initialize()
            
            # Get available tools
            if hasattr(tool_manager, 'get_available_tools'):
                tools = tool_manager.get_available_tools()
                for tool in tools:
                    if isinstance(tool, dict) and 'name' in tool:
                        available_tools.add(tool['name'])
                    elif hasattr(tool, 'name'):
                        available_tools.add(tool.name)
            
            elif hasattr(tool_manager, 'available_tools'):
                # Direct access to available_tools dict
                available_tools.update(tool_manager.available_tools.keys())
        
        except Exception as e:
            logger.error(f"Could not get available tools: {e}")
        
        return available_tools
    
    async def validate_tools(self, tool_manager, data_paths: List[str] = None, 
                           config_paths: List[str] = None) -> Dict[str, any]:
        """Perform comprehensive tool validation"""
        
        # Collect expected tools from multiple sources
        self.expected_tools = set()
        
        # Add tools from data files
        if data_paths:
            for data_path in data_paths:
                self.expected_tools.update(self.load_expected_tools_from_data(data_path))
        
        # Add tools from config files  
        if config_paths:
            for config_path in config_paths:
                self.expected_tools.update(self.load_expected_tools_from_config(config_path))
        
        # Add common tools
        self.expected_tools.update(self.add_common_expected_tools())
        
        # Get available tools
        self.available_tools = await self.get_available_tools(tool_manager)
        
        # Calculate differences
        self.missing_tools = self.expected_tools - self.available_tools
        self.unexpected_tools = self.available_tools - self.expected_tools
        
        # Prepare validation report
        validation_report = {
            'status': 'PASS' if len(self.missing_tools) == 0 else 'WARN',
            'expected_tools_count': len(self.expected_tools),
            'available_tools_count': len(self.available_tools),
            'missing_tools_count': len(self.missing_tools),
            'expected_tools': sorted(list(self.expected_tools)),
            'available_tools': sorted(list(self.available_tools)),
            'missing_tools': sorted(list(self.missing_tools)),
            'unexpected_tools': sorted(list(self.unexpected_tools)),
            'coverage_percentage': (len(self.available_tools & self.expected_tools) / 
                                  len(self.expected_tools) * 100) if self.expected_tools else 100
        }
        
        return validation_report
    
    def log_validation_results(self, validation_report: Dict[str, any]):
        """Log validation results in a readable format"""
        
        logger.info("=" * 60)
        logger.info("TOOL VALIDATION PREFLIGHT CHECK")
        logger.info("=" * 60)
        
        status = validation_report['status']
        if status == 'PASS':
            logger.info("✅ VALIDATION PASSED - All expected tools are available")
        else:
            logger.warning("⚠️  VALIDATION WARNING - Some expected tools are missing")
        
        logger.info(f"Expected tools: {validation_report['expected_tools_count']}")
        logger.info(f"Available tools: {validation_report['available_tools_count']}")
        logger.info(f"Coverage: {validation_report['coverage_percentage']:.1f}%")
        
        if validation_report['missing_tools']:
            logger.warning(f"❌ Missing tools ({len(validation_report['missing_tools'])}):")
            for tool in validation_report['missing_tools']:
                logger.warning(f"   - {tool}")
            logger.warning("These tools may cause failures during training!")
        
        if validation_report['unexpected_tools']:
            logger.info(f"➕ Additional available tools ({len(validation_report['unexpected_tools'])}):")
            for tool in sorted(validation_report['unexpected_tools'])[:10]:  # Show first 10
                logger.info(f"   - {tool}")
            if len(validation_report['unexpected_tools']) > 10:
                logger.info(f"   ... and {len(validation_report['unexpected_tools']) - 10} more")
        
        logger.info("=" * 60)


async def validate_tools_before_training(tool_manager, data_paths: List[str] = None,
                                       config_paths: List[str] = None) -> Dict[str, any]:
    """
    Convenience function to validate tools before training starts
    
    Args:
        tool_manager: The MCP tool manager instance
        data_paths: List of paths to training data files
        config_paths: List of paths to configuration files
        
    Returns:
        Dict containing validation results
    """
    validator = ToolValidator()
    validation_report = await validator.validate_tools(tool_manager, data_paths, config_paths)
    validator.log_validation_results(validation_report)
    
    return validation_report


if __name__ == "__main__":
    # Example usage for testing
    async def test_validator():
        import sys
        from pathlib import Path
        
        # Add environments path
        env_path = str(Path(__file__).parent.parent.parent / "environments")
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
        
        from simple_shared_manager import get_simple_shared_tool_manager
        
        tool_manager = get_simple_shared_tool_manager()
        
        data_paths = [
            "data/processed/train.json",
            "data/inputs/train.json"
        ]
        
        config_paths = [
            "training/configs/training_config_qwen3_0.6b.yaml"
        ]
        
        report = await validate_tools_before_training(tool_manager, data_paths, config_paths)
        print(f"Validation status: {report['status']}")
    
    asyncio.run(test_validator())