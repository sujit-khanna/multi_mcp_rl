#!/usr/bin/env python3
"""
Slack MCP Server - Limited to verified working tools only
Contains only Slack integration tools that have been tested and confirmed to work.
"""

import logging
import os
import asyncio
from typing import Any, Dict, List
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
load_dotenv('../../.env', override=True)

try:
    from slack_sdk.web.async_client import AsyncWebClient
    from slack_sdk.errors import SlackApiError
except ImportError:
    AsyncWebClient = None
    SlackApiError = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('slack_limited_server.log')]
)
logger = logging.getLogger('slack-limited-server')

server = Server("slack-limited-server")
server.version = "1.0.0"

# Environment variables
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

# Initialize client
slack_client = AsyncWebClient(token=SLACK_BOT_TOKEN) if AsyncWebClient and SLACK_BOT_TOKEN else None

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List Slack tools"""
    logger.info("Listing Slack tools...")
    tools = []

    if slack_client:
        logger.info("Adding Slack tools")
        tools.extend([
            Tool(
                name="list_slack_channels",
                description="Lists all public channels in the Slack workspace.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="send_slack_message",
                description="Send a message to a specified Slack channel.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "channel": {"type": "string", "description": "Channel name (without #) or Channel ID."},
                        "message": {"type": "string", "description": "Message text."},
                    },
                    "required": ["channel", "message"],
                },
            ),
            Tool(
                name="get_slack_channel_history",
                description="Retrieve recent messages from a specific channel.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Channel ID to get history from"},
                        "limit": {"type": "integer", "description": "Number of messages to retrieve", "default": 10},
                    },
                    "required": ["channel_id"],
                },
            ),
        ])
    else:
        logger.warning("Slack client not available - check SLACK_BOT_TOKEN")

    logger.info(f"Slack tools loaded: {len(tools)}")
    return tools

@server.call_tool()
async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle Slack tool calls"""
    try:
        logger.info(f"Calling tool: {name} with args: {args}")

        if name == "list_slack_channels" and slack_client:
            try:
                result = await slack_client.conversations_list(limit=200, exclude_archived=True)
                channels = result.get("channels", [])
                
                if not channels:
                    return [TextContent(type="text", text="No channels found or bot doesn't have permission.")]
                
                channel_info = []
                for c in channels:
                    if c.get('is_channel'):
                        channel_info.append(f"#{c['name']} (ID: {c['id']}) - {c.get('purpose', {}).get('value', 'No description')}")
                
                result_text = f"**Available Public Channels ({len(channel_info)} found):**\n\n" + "\n".join(channel_info)
                return [TextContent(type="text", text=result_text)]
                
            except SlackApiError as e:
                error_code = e.response.get("error", "unknown_error")
                return [TextContent(type="text", text=f"❌ Slack API error: {error_code}")]

        elif name == "send_slack_message" and slack_client:
            channel = args["channel"]
            message = args["message"]
            
            if not channel.startswith("C") and not channel.startswith("#"):
                channel = f"#{channel}"
            
            try:
                result = await slack_client.chat_postMessage(channel=channel, text=message)
                if result.get("ok"):
                    return [TextContent(type="text", text=f"✅ Successfully sent message to {channel}")]
                else:
                    error_msg = result.get("error", "Unknown error")
                    return [TextContent(type="text", text=f"❌ Failed to send message: {error_msg}")]
                    
            except SlackApiError as e:
                error_code = e.response.get("error", "unknown_error")
                return [TextContent(type="text", text=f"❌ Slack API error: {error_code}")]

        elif name == "get_slack_channel_history" and slack_client:
            channel_id = args["channel_id"]
            limit = args.get("limit", 10)
            
            try:
                response = await slack_client.conversations_history(channel=channel_id, limit=limit)
                
                if response.get("ok"):
                    messages = response.get("messages", [])
                    if not messages:
                        return [TextContent(type="text", text="No messages found in this channel.")]
                    
                    history_lines = []
                    for msg in reversed(messages):
                        timestamp = datetime.fromtimestamp(float(msg['ts'])).strftime('%Y-%m-%d %H:%M:%S')
                        user = msg.get('user', 'Bot')
                        text = msg.get('text', '')
                        history_lines.append(f"[{timestamp}] @{user}: {text}")
                    
                    result_text = f"**Channel History (last {len(messages)} messages):**\n\n" + "\n".join(history_lines)
                    return [TextContent(type="text", text=result_text)]
                else:
                    error_msg = response.get("error", "Unknown error")
                    return [TextContent(type="text", text=f"❌ Failed to get channel history: {error_msg}")]
                    
            except SlackApiError as e:
                error_code = e.response.get("error", "unknown_error")
                return [TextContent(type="text", text=f"❌ Slack API error: {error_code}")]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' not found or not available")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

async def run():
    """Run the Slack MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            logger.info("Starting Slack Limited MCP Server...")
            await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Slack MCP Server...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")