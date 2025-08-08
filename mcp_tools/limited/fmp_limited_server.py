#!/usr/bin/env python3
"""
FMP MCP Server - Limited to verified working tools only
Contains only Financial Modeling Prep tools that have been tested and confirmed to work.
"""

import logging
import os
import time
import aiohttp
import asyncio
from typing import Any, Dict, List

from mcp.server import Server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
load_dotenv('../../.env', override=True)

# Caching
CACHE: Dict[str, tuple[Any, float]] = {}
CACHE_TTL_SECONDS = 300

def set_cache(key: str, value: Any):
    """Set cache with TTL"""
    CACHE[key] = (value, time.time())

def get_cache(key: str) -> Any | None:
    """Get cache if not expired"""
    if key in CACHE:
        value, timestamp = CACHE[key]
        if time.time() - timestamp < CACHE_TTL_SECONDS:
            return value
        else:
            del CACHE[key]
    return None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('fmp_limited_server.log')]
)
logger = logging.getLogger('fmp-limited-server')

server = Server("fmp-limited-server")
server.version = "1.0.0"

# Environment variables
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")
FMP_BASE_URL = "https://financialmodelingprep.com/api"

_aiohttp_session: aiohttp.ClientSession | None = None

async def get_http_session() -> aiohttp.ClientSession:
    global _aiohttp_session
    if _aiohttp_session is None or _aiohttp_session.closed:
        _aiohttp_session = aiohttp.ClientSession()
    return _aiohttp_session

async def fmp_request(endpoint: str, params: dict = None) -> dict:
    """Make a request to FMP API with caching"""
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY is not configured")
    
    if params is None:
        params = {}
    params['apikey'] = FMP_API_KEY
    
    cache_key = f"fmp_{endpoint}_{str(sorted(params.items()))}"
    cached_result = get_cache(cache_key)
    if cached_result:
        return cached_result
    
    session = await get_http_session()
    url = f"{FMP_BASE_URL}{endpoint}"
    
    async with session.get(url, params=params) as response:
        if response.status == 200:
            data = await response.json()
            set_cache(cache_key, data)
            return data
        else:
            error_text = await response.text()
            logger.error(f"FMP API error: {response.status} - {error_text}")
            raise Exception(f"FMP API error: {response.status} - {error_text}")

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List FMP tools"""
    logger.info("Listing FMP financial tools...")
    tools = []

    if FMP_API_KEY:
        logger.info("Adding FMP financial tools")
        tools.extend([
            Tool(
                name="fmp_get_quote",
                description="Get real-time stock quote including price, volume, and key metrics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL')"}
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="fmp_get_company_profile",
                description="Get comprehensive company profile including business description, sector, and industry",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="fmp_get_income_statement",
                description="Get income statement data for a company",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker symbol"},
                        "period": {"type": "string", "description": "annual or quarterly", "enum": ["annual", "quarterly"], "default": "annual"},
                        "limit": {"type": "integer", "description": "Number of periods to retrieve", "default": 2}
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="fmp_get_market_gainers",
                description="Get top market gainers for the day",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of gainers to return", "default": 10}
                    }
                }
            ),
            Tool(
                name="fmp_get_market_losers",
                description="Get top market losers for the day",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "Number of losers to return", "default": 10}
                    }
                }
            ),
            Tool(
                name="fmp_search_ticker",
                description="Search for stock tickers by company name or symbol",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query (company name or ticker)"},
                        "limit": {"type": "integer", "description": "Maximum number of results", "default": 10}
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="fmp_get_stock_news",
                description="Get latest news for a specific stock or general market news",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tickers": {"type": "string", "description": "Comma-separated ticker symbols (optional)"},
                        "limit": {"type": "integer", "description": "Number of articles", "default": 10}
                    }
                }
            ),
        ])
    else:
        logger.warning("FMP client not available - check FMP_API_KEY")

    logger.info(f"FMP tools loaded: {len(tools)}")
    return tools

@server.call_tool()
async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle FMP tool calls"""
    try:
        logger.info(f"Calling tool: {name} with args: {args}")

        if name == "fmp_get_quote":
            data = await fmp_request(f"/v3/quote/{args['symbol']}")
            if data:
                quote = data[0] if isinstance(data, list) else data
                result = f"**{quote.get('symbol', 'N/A')} - {quote.get('name', 'N/A')}**\n"
                result += f"Price: ${quote.get('price', 'N/A')}\n"
                result += f"Change: ${quote.get('change', 'N/A')} ({quote.get('changesPercentage', 'N/A')}%)\n"
                result += f"Day Range: ${quote.get('dayLow', 'N/A')} - ${quote.get('dayHigh', 'N/A')}\n"
                result += f"Volume: {quote.get('volume', 'N/A'):,}\n"
                result += f"Market Cap: ${quote.get('marketCap', 'N/A'):,}\n"
                result += f"P/E Ratio: {quote.get('pe', 'N/A')}\n"
                return [TextContent(type="text", text=result)]

        elif name == "fmp_get_company_profile":
            data = await fmp_request(f"/v3/profile/{args['symbol']}")
            if data:
                profile = data[0] if isinstance(data, list) else data
                result = f"**{profile.get('companyName', 'N/A')} ({profile.get('symbol', 'N/A')})**\n"
                result += f"Industry: {profile.get('industry', 'N/A')}\n"
                result += f"Sector: {profile.get('sector', 'N/A')}\n"
                result += f"Country: {profile.get('country', 'N/A')}\n"
                result += f"Website: {profile.get('website', 'N/A')}\n"
                result += f"CEO: {profile.get('ceo', 'N/A')}\n"
                
                employees = profile.get('fullTimeEmployees')
                if employees and isinstance(employees, (int, float)):
                    result += f"Employees: {int(employees):,}\n"
                
                mkt_cap = profile.get('mktCap')
                if mkt_cap and isinstance(mkt_cap, (int, float)):
                    result += f"Market Cap: ${int(mkt_cap):,}\n"
                
                description = profile.get('description', 'N/A')
                if description and description != 'N/A':
                    result += f"Description: {description[:300]}..."
                
                return [TextContent(type="text", text=result)]

        elif name == "fmp_get_income_statement":
            data = await fmp_request(f"/v3/income-statement/{args['symbol']}", 
                                   {"period": args.get('period', 'annual'), "limit": args.get('limit', 2)})
            if data:
                result = f"**Income Statement for {args['symbol']}**\n\n"
                for statement in data:
                    result += f"Period: {statement.get('calendarYear', 'N/A')} - {statement.get('period', 'N/A')}\n"
                    result += f"Revenue: ${statement.get('revenue', 'N/A'):,}\n"
                    result += f"Gross Profit: ${statement.get('grossProfit', 'N/A'):,}\n"
                    result += f"Operating Income: ${statement.get('operatingIncome', 'N/A'):,}\n"
                    result += f"Net Income: ${statement.get('netIncome', 'N/A'):,}\n"
                    result += f"EPS: ${statement.get('eps', 'N/A')}\n---\n"
                return [TextContent(type="text", text=result)]

        elif name == "fmp_get_market_gainers":
            data = await fmp_request("/v3/stock_market/gainers", {"limit": args.get('limit', 10)})
            if data:
                result = "**Top Market Gainers**\n\n"
                for stock in data:
                    result += f"{stock.get('symbol', 'N/A')}: ${stock.get('price', 'N/A')} "
                    result += f"(+{stock.get('change', 'N/A')} / +{stock.get('changesPercentage', 'N/A')}%)\n"
                return [TextContent(type="text", text=result)]

        elif name == "fmp_get_market_losers":
            data = await fmp_request("/v3/stock_market/losers", {"limit": args.get('limit', 10)})
            if data:
                result = "**Top Market Losers**\n\n"
                for stock in data:
                    result += f"{stock.get('symbol', 'N/A')}: ${stock.get('price', 'N/A')} "
                    result += f"({stock.get('change', 'N/A')} / {stock.get('changesPercentage', 'N/A')}%)\n"
                return [TextContent(type="text", text=result)]

        elif name == "fmp_search_ticker":
            data = await fmp_request(f"/v3/search", {"query": args['query'], "limit": args.get('limit', 10)})
            if data:
                result = f"**Search Results for '{args['query']}'**\n\n"
                for item in data:
                    result += f"Symbol: {item.get('symbol', 'N/A')}\n"
                    result += f"Name: {item.get('name', 'N/A')}\n"
                    result += f"Exchange: {item.get('stockExchange', 'N/A')}\n---\n"
                return [TextContent(type="text", text=result)]

        elif name == "fmp_get_stock_news":
            params = {"limit": args.get('limit', 10)}
            if args.get('tickers'):
                params['tickers'] = args['tickers']
            data = await fmp_request("/v3/stock_news", params)
            if data:
                result = "**Stock News**\n\n"
                for article in data:
                    result += f"Title: {article.get('title', 'N/A')}\n"
                    result += f"URL: {article.get('url', 'N/A')}\n"
                    result += f"Published: {article.get('publishedDate', 'N/A')}\n"
                    result += f"Site: {article.get('site', 'N/A')}\n---\n"
                return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' not found or not available")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

async def run():
    """Run the FMP MCP server"""
    from mcp.server.stdio import stdio_server
    global _aiohttp_session
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            logger.info("Starting FMP Limited MCP Server...")
            await get_http_session()
            await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down FMP MCP Server...")
            if _aiohttp_session and not _aiohttp_session.closed:
                await _aiohttp_session.close()

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")