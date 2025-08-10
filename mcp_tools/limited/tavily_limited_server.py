#!/usr/bin/env python3
"""
Tavily MCP Server - Limited to verified working tools only
Contains only Tavily web search tools that have been tested and confirmed to work.
"""

import logging
import os
import asyncio
from typing import Any, Dict, List

from mcp.server import Server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
import pathlib
# Get the project root (two levels up from this file)
project_root = pathlib.Path(__file__).parent.parent.parent
env_path = project_root / '.env'
load_dotenv(env_path, override=True)

try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('tavily_limited_server.log')]
)
logger = logging.getLogger('tavily-limited-server')

server = Server("tavily-limited-server")
server.version = "1.0.0"

# Environment variables
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Initialize client
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TavilyClient and TAVILY_API_KEY else None

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List Tavily tools"""
    logger.info("Listing Tavily tools...")
    tools = []

    if tavily_client:
        logger.info("Adding Tavily tools")
        tools.extend([
            Tool(
                name="tavily_search",
                description="Perform real-time web search using Tavily's advanced search capabilities.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query to execute"},
                        "max_results": {"type": "integer", "description": "Maximum number of results to return", "default": 5},
                        "search_depth": {"type": "string", "description": "Search depth: 'basic' or 'advanced'", "enum": ["basic", "advanced"], "default": "basic"}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="tavily_qna_search",
                description="Get direct answers to questions using Tavily's QnA search.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question to get an answer for"}
                    },
                    "required": ["query"]
                }
            ),
        ])
    else:
        logger.warning("Tavily client not available - check TAVILY_API_KEY")

    logger.info(f"Tavily tools loaded: {len(tools)}")
    return tools

@server.call_tool()
async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle Tavily tool calls"""
    try:
        logger.info(f"Calling tool: {name} with args: {args}")

        if name == "tavily_search" and tavily_client:
            query = args["query"]
            max_results = args.get("max_results", 5)
            search_depth = args.get("search_depth", "basic")
            
            def _tavily_search():
                response = tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth=search_depth
                )
                
                results = []
                for result in response.get("results", []):
                    result_text = f"**{result.get('title', 'No title')}**\n"
                    result_text += f"URL: {result.get('url', 'No URL')}\n"
                    result_text += f"Content: {result.get('content', 'No content')}\n"
                    if result.get('score'):
                        result_text += f"Relevance Score: {result.get('score')}\n"
                    result_text += "---\n"
                    results.append(result_text)
                
                return "\n".join(results) if results else "No results found."
            
            result = await asyncio.to_thread(_tavily_search)
            return [TextContent(type="text", text=result)]

        elif name == "tavily_qna_search" and tavily_client:
            query = args["query"]
            
            def _tavily_qna_search():
                response = tavily_client.qna_search(query=query)
                
                # Tavily QnA returns a string directly, not a dictionary
                if isinstance(response, str):
                    result_text = f"**Answer:** {response}\n"
                else:
                    # Fallback for dictionary response (if API changes)
                    answer = response.get("answer", "No answer found.") if hasattr(response, 'get') else str(response)
                    sources = response.get("sources", []) if hasattr(response, 'get') else []
                    
                    result_text = f"**Answer:** {answer}\n\n"
                    if sources:
                        result_text += "**Sources:**\n"
                        for source in sources:
                            result_text += f"- {source}\n"
                
                return result_text
            
            result = await asyncio.to_thread(_tavily_qna_search)
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' not found or not available")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

async def run():
    """Run the Tavily MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            logger.info("Starting Tavily Limited MCP Server...")
            await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Tavily MCP Server...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")