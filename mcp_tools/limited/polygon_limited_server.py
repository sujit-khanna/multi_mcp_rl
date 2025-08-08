#!/usr/bin/env python3
"""
Polygon MCP Server - Limited to verified working tools only
Contains only Polygon.io tools that have been tested and confirmed to work with Stocks Starter plan.
"""

import logging
import os
import asyncio
import requests
import pandas as pd
from typing import Any, Dict, List
from datetime import datetime

from mcp.server import Server
from mcp.types import Tool, TextContent

from dotenv import load_dotenv
load_dotenv('../../.env', override=True)

try:
    from polygon import RESTClient as PolygonClient
    from polygon.rest.models import Agg
except ImportError:
    PolygonClient = None
    Agg = None

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('polygon_limited_server.log')]
)
logger = logging.getLogger('polygon-limited-server')

server = Server("polygon-limited-server")
server.version = "1.0.0"

# Environment variables
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# Initialize client
polygon_client = PolygonClient(api_key=POLYGON_API_KEY) if PolygonClient and POLYGON_API_KEY else None

def fetch_price_data(ticker, start_date, end_date, frequency="daily"):
    """
    Fetch price data for a ticker using Polygon API.
    
    Parameters:
    -----------
    ticker : str
        The ticker symbol
    start_date : datetime or str
        Start date for data retrieval
    end_date : datetime or str  
        End date for data retrieval
    frequency : str
        Data frequency ("hourly", "minute", or "daily")
        
    Returns:
    --------
    pandas DataFrame with OHLC data
    """
    if not polygon_client:
        return pd.DataFrame()
    
    # Ensure dates are in the right format
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Format dates for Polygon API
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    aggs = []
    try:
        # Set timespan based on frequency
        if frequency.lower() == "daily":
            timespan = "day"
            multiplier = 1
        elif frequency == "minute":
            timespan = "minute"
            multiplier = 15
        else:  # hourly
            timespan = "hour"
            multiplier = 1
            
        # Fetch data
        data = polygon_client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_str,
            to=end_str,
            limit=50000
        )
        aggs.extend(data)
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame([
        {
            "datetime": datetime.fromtimestamp(agg.timestamp / 1000),
            "Open": agg.open,
            "High": agg.high,
            "Low": agg.low,
            "Close": agg.close,
            "Volume": agg.volume,
        }
        for agg in aggs if isinstance(agg, Agg)
    ])

    if df.empty:
        return df

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    
    # Filter for market hours only if hourly data is requested
    if frequency.lower() == "hourly":
        df = df.between_time('09:30', '16:00')
    
    return df

def get_polygon_news(ticker, limit=10):
    """
    Fetch and process news articles for a given ticker from Polygon API
    
    Args:
        ticker (str): Stock ticker symbol
        limit (int): Maximum number of news articles to fetch
    
    Returns:
        dict: Dictionary with publish dates as keys and article text as values
    """
    articles_by_date = {}
    
    if not POLYGON_API_KEY:
        return articles_by_date
    
    # Fetch news data
    base_url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": limit,
        "apiKey": POLYGON_API_KEY
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Polygon news API error for {ticker}: {e}")
        return articles_by_date
    
    # Check if we have results
    if not news_data or "results" not in news_data:
        return articles_by_date
    
    for article in news_data["results"]:
        # Get publish date
        published_utc = article.get("published_utc", "")
        if not published_utc:
            continue
            
        dt = datetime.strptime(published_utc, "%Y-%m-%dT%H:%M:%SZ")
        publish_date = dt.strftime("%Y-%m-%d")
        
        # Build article text
        article_text = []
        article_text.append(f"Title: {article.get('title', 'No Title')}")
        article_text.append("")
        article_text.append(f"Description: {article.get('description', 'No Description')}")
        article_text.append("")
        
        # Add insights if available
        insights = article.get("insights", [])
        if insights:
            article_text.append("Insights:")
            for insight in insights:
                article_text.append(f"* {insight['ticker']} ({insight['sentiment']})")
                article_text.append(f"  * {insight['sentiment_reasoning']}")
            article_text.append("")
        
        # Store in dictionary with publish date as key
        articles_by_date[publish_date] = "\n".join(article_text)
    
    return articles_by_date

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List Polygon tools"""
    logger.info("Listing Polygon tools...")
    tools = []

    if polygon_client:
        logger.info("Adding Polygon tools (Stocks Starter plan)")
        tools.extend([
            Tool(
                name="polygon_get_aggs",
                description="Get stock OHLC aggregates data with customizable frequency (daily, hourly, minute).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., 'AAPL')"},
                        "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                        "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                        "frequency": {"type": "string", "description": "Data frequency", "enum": ["daily", "hourly", "minute"], "default": "daily"}
                    },
                    "required": ["ticker", "start_date", "end_date"]
                }
            ),
            Tool(
                name="polygon_get_news",
                description="Get recent news articles for a stock ticker with sentiment analysis.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"},
                        "limit": {"type": "integer", "description": "Number of articles to retrieve", "default": 10}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="polygon_get_ticker_details",
                description="Get detailed information about a ticker including company data.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            ),
            Tool(
                name="polygon_get_market_status",
                description="Get current market status and trading hours.",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="polygon_get_previous_close",
                description="Get previous close data for a ticker.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            ),
        ])
    else:
        logger.warning("Polygon client not available - check POLYGON_API_KEY")

    logger.info(f"Polygon tools loaded: {len(tools)}")
    return tools

@server.call_tool()
async def call_tool(name: str, args: Dict[str, Any]) -> List[TextContent]:
    """Handle Polygon tool calls"""
    try:
        logger.info(f"Calling tool: {name} with args: {args}")

        if name == "polygon_get_aggs" and polygon_client:
            ticker = args["ticker"]
            start_date = args["start_date"]
            end_date = args["end_date"]
            frequency = args.get("frequency", "daily")
            
            def _polygon_get_aggs():
                df = fetch_price_data(ticker, start_date, end_date, frequency)
                
                if df.empty:
                    return f"No price data found for {ticker}"
                
                result = f"**{ticker} Price Data ({frequency}):**\n"
                result += f"Period: {start_date} to {end_date}\n"
                result += f"Records: {len(df)}\n\n"
                
                # Show summary stats
                result += f"**Summary:**\n"
                result += f"High: ${df['High'].max():.2f}\n"
                result += f"Low: ${df['Low'].min():.2f}\n"
                result += f"Avg Volume: {df['Volume'].mean():,.0f}\n\n"
                
                # Show last few records
                result += f"**Recent Data:**\n"
                for idx, row in df.tail(5).iterrows():
                    result += f"{idx.strftime('%Y-%m-%d %H:%M')}: O=${row['Open']:.2f} H=${row['High']:.2f} L=${row['Low']:.2f} C=${row['Close']:.2f} V={row['Volume']:,.0f}\n"
                
                return result
            
            result = await asyncio.to_thread(_polygon_get_aggs)
            return [TextContent(type="text", text=result)]

        elif name == "polygon_get_news" and polygon_client:
            ticker = args["ticker"]
            limit = args.get("limit", 10)
            
            def _polygon_get_news():
                articles = get_polygon_news(ticker, limit)
                
                if not articles:
                    return f"No news found for {ticker}"
                
                result = f"**Polygon News for {ticker}:**\n\n"
                for date, article in articles.items():
                    result += f"**{date}:**\n{article}\n---\n"
                
                return result
            
            result = await asyncio.to_thread(_polygon_get_news)
            return [TextContent(type="text", text=result)]

        elif name == "polygon_get_ticker_details" and polygon_client:
            ticker = args["ticker"]
            
            def _polygon_get_ticker_details():
                try:
                    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
                    params = {"apikey": POLYGON_API_KEY}
                    
                    response = requests.get(url, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    if "results" not in data:
                        return f"No details found for {ticker}"
                    
                    details = data["results"]
                    result = f"**{ticker} Ticker Details:**\n"
                    result += f"Name: {details.get('name', 'N/A')}\n"
                    result += f"Market: {details.get('market', 'N/A')}\n"
                    result += f"Primary Exchange: {details.get('primary_exchange', 'N/A')}\n"
                    result += f"Type: {details.get('type', 'N/A')}\n"
                    result += f"Active: {details.get('active', 'N/A')}\n"
                    result += f"Currency: {details.get('currency_name', 'N/A')}\n"
                    
                    if 'description' in details:
                        result += f"Description: {details['description'][:300]}...\n"
                    
                    return result
                    
                except Exception as e:
                    return f"Error getting ticker details: {e}"
            
            result = await asyncio.to_thread(_polygon_get_ticker_details)
            return [TextContent(type="text", text=result)]

        elif name == "polygon_get_market_status" and polygon_client:
            def _polygon_get_market_status():
                try:
                    url = "https://api.polygon.io/v1/marketstatus/now"
                    params = {"apikey": POLYGON_API_KEY}
                    
                    response = requests.get(url, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    result = "**Market Status:**\n"
                    result += f"Market: {data.get('market', 'N/A')}\n"
                    result += f"Server Time: {data.get('serverTime', 'N/A')}\n"
                    
                    if 'exchanges' in data:
                        result += "\n**Exchanges:**\n"
                        for exchange, status in data['exchanges'].items():
                            result += f"{exchange}: {status}\n"
                    
                    if 'currencies' in data:
                        result += "\n**Currencies:**\n"
                        for currency, status in data['currencies'].items():
                            result += f"{currency}: {status}\n"
                    
                    return result
                    
                except Exception as e:
                    return f"Error getting market status: {e}"
            
            result = await asyncio.to_thread(_polygon_get_market_status)
            return [TextContent(type="text", text=result)]

        elif name == "polygon_get_previous_close" and polygon_client:
            ticker = args["ticker"]
            
            def _polygon_get_previous_close():
                try:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev"
                    params = {"apikey": POLYGON_API_KEY}
                    
                    response = requests.get(url, params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                    
                    if "results" not in data or not data["results"]:
                        return f"No previous close data for {ticker}"
                    
                    prev = data["results"][0]
                    result = f"**{ticker} Previous Close:**\n"
                    result += f"Date: {datetime.fromtimestamp(prev['t']/1000).strftime('%Y-%m-%d')}\n"
                    result += f"Open: ${prev.get('o', 'N/A')}\n"
                    result += f"High: ${prev.get('h', 'N/A')}\n"
                    result += f"Low: ${prev.get('l', 'N/A')}\n"
                    result += f"Close: ${prev.get('c', 'N/A')}\n"
                    result += f"Volume: {prev.get('v', 'N/A'):,}\n"
                    
                    return result
                    
                except Exception as e:
                    return f"Error getting previous close: {e}"
            
            result = await asyncio.to_thread(_polygon_get_previous_close)
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Tool '{name}' not found or not available")]

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool {name}: {str(e)}")]

async def run():
    """Run the Polygon MCP server"""
    from mcp.server.stdio import stdio_server
    
    async with stdio_server() as (read_stream, write_stream):
        try:
            logger.info("Starting Polygon Limited MCP Server...")
            await server.run(read_stream, write_stream, server.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}", exc_info=True)
        finally:
            logger.info("Shutting down Polygon MCP Server...")

if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")