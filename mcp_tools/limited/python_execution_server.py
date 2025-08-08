#!/usr/bin/env python3
"""
Python Execution MCP Server - Augments agent with code execution capabilities
Allows agents to write and execute Python code for intermediate data processing tasks,
ingesting outputs from other MCP servers for analysis and transformation.

Security: Uses restricted execution environment with safe built-ins only.
"""

import logging
import os
import sys
import json
import pandas as pd
import numpy as np
import asyncio
import traceback
from typing import Any, Dict, List
from datetime import datetime, timedelta
import io
import contextlib
import ast
import operator
import base64
import math
import statistics

from mcp.server import Server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
load_dotenv('../../.env', override=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('python_execution_server.log')]
)
logger = logging.getLogger('python-execution-server')

# Try to import matplotlib - make it optional
try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')  # Use non-interactive backend
    MATPLOTLIB_AVAILABLE = True
    logger.info("matplotlib imported successfully")
except ImportError:
    matplotlib = None
    plt = None
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available - plotting features will be limited")

server = Server("python-execution-server")
server.version = "1.0.0"

# Safe execution environment globals
SAFE_GLOBALS = {
    # Built-in functions (safe subset) - need to include __import__ for pandas functionality
    '__builtins__': {
        'abs': abs,
        'all': all,
        'any': any,
        'bool': bool,
        'dict': dict,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'int': int,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'range': range,
        'round': round,
        'set': set,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'zip': zip,
        'print': print,
        '__import__': __import__,  # Needed for pandas imports
        '__name__': '__main__',
        '__doc__': None,
    },
    # Data manipulation libraries
    'pd': pd,
    'np': np,
    'json': json,
    'datetime': datetime,
    'timedelta': timedelta,
    # Math operators
    'operator': operator,
    # AST for safe evaluation
    'ast': ast,
    # Visualization libraries (if available)
    'plt': plt if MATPLOTLIB_AVAILABLE else None,
    'matplotlib': matplotlib if MATPLOTLIB_AVAILABLE else None,
    # Encoding/IO libraries
    'io': io,
    'base64': base64,
    # Math libraries
    'math': math,
    'statistics': statistics,
}

# Execution context storage for multi-step operations
EXECUTION_CONTEXT = {}

def validate_code_safety(code: str) -> tuple[bool, str]:
    """
    Validate that code is safe to execute by checking for dangerous patterns.
    Returns (is_safe, error_message)
    """
    dangerous_patterns = [
        'exec',
        'eval', 
        'raw_input',
        'compile',
        'reload',
        'exit',
        'quit',
        'subprocess',
        'socket',
        'urllib',
        'requests',
        'http',
    ]
    
    code_lower = code.lower()
    for pattern in dangerous_patterns:
        if pattern in code_lower:
            return False, f"Potentially unsafe code detected: '{pattern}' is not allowed"
    
    # Check for import statements - allow common data science libraries
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    allowed_imports = ['json', 'math', 'statistics', 'collections', 'itertools', 'pandas', 'numpy', 'io', 'base64']
                    if MATPLOTLIB_AVAILABLE:
                        allowed_imports.extend(['matplotlib'])
                    if alias.name not in allowed_imports and not alias.name.startswith('matplotlib.'):
                        return False, f"Import '{alias.name}' is not allowed"
            elif isinstance(node, ast.ImportFrom):
                allowed_modules = ['json', 'math', 'statistics', 'collections', 'itertools', 'datetime', 'pandas', 'numpy', 'io', 'base64']
                if MATPLOTLIB_AVAILABLE:
                    allowed_modules.extend(['matplotlib', 'matplotlib.pyplot'])
                if node.module not in allowed_modules:
                    return False, f"Import from '{node.module}' is not allowed"
    except SyntaxError as e:
        return False, f"Syntax error in code: {e}"
    
    return True, ""

def safe_execute_code(code: str, context: Dict[str, Any] = None) -> tuple[bool, str, Any]:
    """
    Safely execute Python code with restricted globals and capture output.
    Returns (success, output_text, result_value)
    """
    if context is None:
        context = {}
    
    # Validate code safety
    is_safe, safety_error = validate_code_safety(code)
    if not is_safe:
        return False, f"Security Error: {safety_error}", None
    
    # Prepare execution environment
    execution_globals = SAFE_GLOBALS.copy()
    execution_globals.update(context)
    
    # Capture stdout
    stdout_capture = io.StringIO()
    result_value = None
    
    try:
        with contextlib.redirect_stdout(stdout_capture):
            # Try to evaluate as expression first
            try:
                result_value = eval(code, execution_globals)
                if result_value is not None:
                    print(f"Result: {result_value}")
            except SyntaxError:
                # If not an expression, execute as statements
                exec(code, execution_globals)
                # Try to find result in common variable names
                for var_name in ['result', 'output', 'data', 'df', 'analysis']:
                    if var_name in execution_globals and var_name not in SAFE_GLOBALS:
                        result_value = execution_globals[var_name]
                        break
        
        output_text = stdout_capture.getvalue()
        return True, output_text, result_value
        
    except Exception as e:
        error_msg = f"Execution Error: {type(e).__name__}: {str(e)}"
        traceback_info = traceback.format_exc()
        return False, f"{error_msg}\n\nTraceback:\n{traceback_info}", None

def format_execution_result(success: bool, output: str, result: Any) -> str:
    """Format execution results for display"""
    if not success:
        return f"❌ **Execution Failed**\n\n```\n{output}\n```"
    
    formatted_output = "✅ **Code Executed Successfully**\n\n"
    
    if output.strip():
        formatted_output += f"**Output:**\n```\n{output.strip()}\n```\n\n"
    
    if result is not None:
        if isinstance(result, pd.DataFrame):
            formatted_output += f"**Result (DataFrame):**\n```\n{result.to_string()}\n```\n"
        elif isinstance(result, (dict, list)):
            formatted_output += f"**Result (JSON):**\n```json\n{json.dumps(result, indent=2, default=str)}\n```\n"
        elif hasattr(result, '__str__') and len(str(result)) < 1000:
            formatted_output += f"**Result:**\n```\n{result}\n```\n"
        else:
            formatted_output += f"**Result Type:** {type(result).__name__}\n"
    
    return formatted_output

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List Python execution tools"""
    logger.info("Listing Python execution tools...")
    
    tools = [
        Tool(
            name="execute_python",
            description="Execute Python code safely for data processing and analysis. Supports pandas, numpy, json, and datetime.",
            inputSchema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"},
                    "description": {"type": "string", "description": "Optional description of what the code does"}
                },
                "required": ["code"]
            }
        ),
        Tool(
            name="process_mcp_data",
            description="Process data from other MCP servers using Python. Expects JSON data input.",
            inputSchema={
                "type": "object", 
                "properties": {
                    "data": {"type": "string", "description": "JSON data from other MCP servers"},
                    "processing_code": {"type": "string", "description": "Python code to process the data"},
                    "description": {"type": "string", "description": "Description of the processing task"}
                },
                "required": ["data", "processing_code"]
            }
        ),
        Tool(
            name="analyze_financial_data",
            description="Analyze financial data using pandas and numpy with built-in financial analysis functions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "data": {"type": "string", "description": "Financial data in JSON format"},
                    "analysis_type": {"type": "string", "description": "Type of analysis: 'summary', 'trend', 'correlation', 'custom'"},
                    "custom_code": {"type": "string", "description": "Custom analysis code (required if analysis_type is 'custom')"}
                },
                "required": ["data", "analysis_type"]
            }
        ),
        Tool(
            name="create_data_pipeline",
            description="Create a multi-step data processing pipeline that can chain operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "pipeline_steps": {"type": "string", "description": "JSON array of pipeline steps with code"},
                    "input_data": {"type": "string", "description": "Initial input data"},
                    "description": {"type": "string", "description": "Description of the pipeline"}
                },
                "required": ["pipeline_steps", "input_data"]
            }
        ),
        Tool(
            name="get_execution_context",
            description="Get stored variables from previous executions for multi-step operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_key": {"type": "string", "description": "Key to retrieve from execution context"}
                }
            }
        ),
        Tool(
            name="store_execution_result",
            description="Store execution result in context for later use in multi-step operations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "context_key": {"type": "string", "description": "Key to store the result under"},
                    "data": {"type": "string", "description": "Data to store (JSON format)"},
                    "description": {"type": "string", "description": "Description of the stored data"}
                },
                "required": ["context_key", "data"]
            }
        )
    ]
    
    logger.info(f"Python execution tools loaded: {len(tools)}")
    return tools

@server.call_tool()
async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle Python execution tool calls"""
    try:
        logger.info(f"Calling tool: {name} with args: {args}")

        if name == "execute_python":
            code = args["code"]
            description = args.get("description", "")
            
            logger.info(f"Executing Python code: {description}")
            
            success, output, result = safe_execute_code(code)
            formatted_result = format_execution_result(success, output, result)
            
            return [TextContent(type="text", text=formatted_result)]

        elif name == "process_mcp_data":
            data_json = args["data"]
            processing_code = args["processing_code"]
            description = args.get("description", "")
            
            logger.info(f"Processing MCP data: {description}")
            
            try:
                # Parse input data
                data = json.loads(data_json)
                
                # Create context with the data (using mcp_data to avoid input() confusion)
                context = {"mcp_data": data, "data": data}
                
                # Replace input_data references with mcp_data in the code
                processing_code = processing_code.replace("input_data", "mcp_data")
                
                # Execute processing code
                success, output, result = safe_execute_code(processing_code, context)
                formatted_result = format_execution_result(success, output, result)
                
                return [TextContent(type="text", text=formatted_result)]
                
            except json.JSONDecodeError as e:
                return [TextContent(type="text", text=f"❌ **JSON Parse Error**: {e}")]

        elif name == "analyze_financial_data":
            data_json = args["data"]
            analysis_type = args["analysis_type"]
            custom_code = args.get("custom_code", "")
            
            logger.info(f"Analyzing financial data: {analysis_type}")
            
            try:
                # Parse financial data
                data = json.loads(data_json)
                
                # Create DataFrame if data is list of records
                if isinstance(data, list) and len(data) > 0:
                    df = pd.DataFrame(data)
                    context = {"df": df, "data": data, "financial_data": df}
                else:
                    context = {"data": data}
                
                # Generate analysis code based on type
                if analysis_type == "summary":
                    code = """
# Financial Data Summary Analysis
result = {}
try:
    result['shape'] = df.shape
    result['columns'] = list(df.columns)
    result['numeric_summary'] = df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    result['missing_values'] = df.isnull().sum().to_dict()
    print("Financial Data Summary:")
    print(f"Shape: {result['shape']}")
    print(f"Columns: {result['columns']}")
except NameError:
    result['data_type'] = str(type(data))
    result['data_preview'] = str(data)[:500]
    print("Data Summary:")
    print(f"Type: {result['data_type']}")
"""
                elif analysis_type == "trend":
                    code = """
# Trend Analysis
result = {}
try:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        result['trends'] = {}
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 1:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                result['trends'][col] = {
                    'slope': float(trend),
                    'direction': 'increasing' if trend > 0 else 'decreasing',
                    'mean': float(values.mean()),
                    'std': float(values.std())
                }
    print("Trend Analysis Results:")
    for col, trend in result.get('trends', {}).items():
        print(f"{col}: {trend['direction']} (slope: {trend['slope']:.4f})")
except NameError:
    print("Cannot perform trend analysis on non-DataFrame data")
    result = {"error": "Data must be in DataFrame format for trend analysis"}
"""
                elif analysis_type == "correlation":
                    code = """
# Correlation Analysis
result = {}
try:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        result['correlation_matrix'] = corr_matrix.to_dict()
        # Find highest correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if not pd.isna(corr_val):
                    corr_pairs.append({
                        'pair': f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}",
                        'correlation': float(corr_val)
                    })
        result['top_correlations'] = sorted(corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]
    print("Correlation Analysis:")
    if 'top_correlations' in result:
        for pair in result['top_correlations']:
            print(f"{pair['pair']}: {pair['correlation']:.3f}")
except NameError:
    print("Cannot perform correlation analysis on non-DataFrame data")
    result = {"error": "Data must be in DataFrame format for correlation analysis"}
"""
                elif analysis_type == "custom":
                    if not custom_code:
                        return [TextContent(type="text", text="❌ **Error**: custom_code is required for custom analysis")]
                    code = custom_code
                else:
                    return [TextContent(type="text", text=f"❌ **Error**: Unknown analysis type '{analysis_type}'")]
                
                # Execute analysis
                success, output, result = safe_execute_code(code, context)
                formatted_result = format_execution_result(success, output, result)
                
                return [TextContent(type="text", text=formatted_result)]
                
            except json.JSONDecodeError as e:
                return [TextContent(type="text", text=f"❌ **JSON Parse Error**: {e}")]

        elif name == "create_data_pipeline":
            pipeline_steps_json = args["pipeline_steps"]
            input_data_json = args["input_data"]
            description = args.get("description", "")
            
            logger.info(f"Creating data pipeline: {description}")
            
            try:
                # Parse pipeline steps and input data
                pipeline_steps = json.loads(pipeline_steps_json)
                input_data = json.loads(input_data_json)
                
                # Initialize pipeline context
                context = {"data": input_data, "pipeline_results": []}
                
                # Execute each step
                pipeline_output = ["🔄 **Data Pipeline Execution**\n"]
                
                for i, step in enumerate(pipeline_steps, 1):
                    step_code = step.get("code", "")
                    step_desc = step.get("description", f"Step {i}")
                    
                    pipeline_output.append(f"**Step {i}: {step_desc}**")
                    
                    success, output, result = safe_execute_code(step_code, context)
                    
                    if success:
                        pipeline_output.append(f"✅ Success")
                        if output.strip():
                            pipeline_output.append(f"```\n{output.strip()}\n```")
                        
                        # Update context with result
                        if result is not None:
                            context["data"] = result
                            context["pipeline_results"].append(result)
                    else:
                        pipeline_output.append(f"❌ Failed: {output}")
                        break
                    
                    pipeline_output.append("")
                
                # Final result
                final_result = context.get("data", input_data)
                if isinstance(final_result, pd.DataFrame):
                    pipeline_output.append(f"**Final Result (DataFrame):**\n```\n{final_result.to_string()}\n```")
                elif isinstance(final_result, (dict, list)):
                    pipeline_output.append(f"**Final Result:**\n```json\n{json.dumps(final_result, indent=2, default=str)}\n```")
                
                return [TextContent(type="text", text="\n".join(pipeline_output))]
                
            except json.JSONDecodeError as e:
                return [TextContent(type="text", text=f"❌ **JSON Parse Error**: {e}")]

        elif name == "get_execution_context":
            context_key = args.get("context_key", "")
            
            if not context_key:
                # Return all context keys
                keys = list(EXECUTION_CONTEXT.keys())
                return [TextContent(type="text", text=f"**Available Context Keys:**\n{json.dumps(keys, indent=2)}")]
            
            if context_key in EXECUTION_CONTEXT:
                data = EXECUTION_CONTEXT[context_key]
                return [TextContent(type="text", text=f"**Context Data for '{context_key}':**\n```json\n{json.dumps(data, indent=2, default=str)}\n```")]
            else:
                return [TextContent(type="text", text=f"❌ **Context key '{context_key}' not found**")]

        elif name == "store_execution_result":
            context_key = args["context_key"]
            data_json = args["data"]
            description = args.get("description", "")
            
            try:
                data = json.loads(data_json)
                EXECUTION_CONTEXT[context_key] = data
                
                return [TextContent(type="text", text=f"✅ **Stored data under key '{context_key}'**\nDescription: {description}")]
                
            except json.JSONDecodeError as e:
                return [TextContent(type="text", text=f"❌ **JSON Parse Error**: {e}")]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' not found or not available")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

async def run():
    """Run the Python execution MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            logger.info("Starting Python Execution MCP Server...")
            await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Python Execution MCP Server...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")