import os
import requests
import pandas as pd
import time
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Set your API key
API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY", "your_api_key_here")

class Price(BaseModel):
    time: str
    open: float
    close: float
    high: float
    low: float
    volume: int

class PriceResponse(BaseModel):
    prices: List[Price]

def get_price_data(ticker: str, start_date: str, end_date: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Fetch price data based on specified timeframe"""
    headers = {"X-API-KEY": API_KEY}
    
    # Map timeframe to API parameters
    interval_map = {
        "5min": ("minute", 5),
        "15min": ("minute", 15),
        "30min": ("minute", 30),
        "60min": ("hour", 1),
        "daily": ("day", 1)
    }
    
    if timeframe not in interval_map:
        logger.error(f"Unsupported timeframe: {timeframe}")
        return None
        
    interval, interval_multiplier = interval_map[timeframe]
    
    url = (
        f"https://api.financialdatasets.ai/prices/?ticker={ticker}"
        f"&interval={interval}&interval_multiplier={interval_multiplier}"
        f"&start_date={start_date}&end_date={end_date}"
    )
    
    try:
        logger.debug(f"Requesting data: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        parsed = PriceResponse(**data)
        
        df = pd.DataFrame([p.model_dump() for p in parsed.prices])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('US/Eastern')
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            logger.debug(f"Retrieved {len(df)} records")
            return df
        return None
        
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return None

def get_recent_data(ticker: str, num_candles: int, timeframe: str) -> Optional[List[List[float]]]:
    """Get recent OHLCV data for pattern detection"""
    tz = pytz.timezone('US/Eastern')
    end_date = datetime.now(tz)
    
    # Calculate start date based on timeframe to get enough data
    if timeframe == "5min":
        days_back = max(5, num_candles * 5 / (60 * 6.5))  # 6.5 trading hours per day
        start_date = end_date - timedelta(days=days_back)
    elif timeframe == "15min":
        days_back = max(5, num_candles * 15 / (60 * 6.5))
        start_date = end_date - timedelta(days=days_back)
    elif timeframe == "30min":
        days_back = max(5, num_candles * 30 / (60 * 6.5))
        start_date = end_date - timedelta(days=days_back)
    elif timeframe == "60min":
        days_back = max(5, num_candles / 6.5)  # ~6.5 candles per day
        start_date = end_date - timedelta(days=days_back)
    else:  # daily
        days_back = num_candles + 10  # Add buffer for weekends/holidays
        start_date = end_date - timedelta(days=days_back)
    
    df = get_price_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), timeframe)
    
    if df is None or df.empty:
        logger.error("No data retrieved from API")
        return None
    
    # Get the latest candles
    last_candles = df.tail(num_candles)
    
    # Convert to required format: [[open, high, low, close, volume], ...]
    candles = []
    for _, row in last_candles.iterrows():
        candles.append([
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            int(row['volume'])
        ])
    
    logger.debug(f"Converted {len(candles)} candles for {ticker}")
    return candles

def get_full_data(ticker: str, start_date: str, end_date: str, timeframe: str = "5min") -> Optional[pd.DataFrame]:
    """
    Fetch full data for a date range with chunking for large requests
    """
    # Convert to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate monthly chunks for large requests
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=30), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    
    all_data = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        retry = 0
        while retry < 3:  # Max 3 retries
            df_chunk = get_price_data(ticker, chunk_start, chunk_end, timeframe)
            if df_chunk is not None:
                all_data.append(df_chunk)
                break
            retry += 1
            time.sleep(2 ** retry)  # Exponential backoff
        else:
            logger.warning(f"Failed to get data for {chunk_start} to {chunk_end} after 3 attempts")
        
        # Add delay between chunks to avoid rate limiting
        if i < len(chunks) - 1:
            time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data).sort_index()
