import os
import requests
import pandas as pd
import time
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pytz

# Load environment variables
load_dotenv()

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

def get_intraday_5min_data_chunk(ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    headers = {"X-API-KEY": API_KEY}
    url = (
        f"https://api.financialdatasets.ai/prices/?ticker={ticker}"
        f"&interval=minute&interval_multiplier=5"
        f"&start_date={start_date}&end_date={end_date}"
    )
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code != 200:
            return None
        
        data = response.json()
        parsed = PriceResponse(**data)
        
        df = pd.DataFrame([p.dict() for p in parsed.prices])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('US/Eastern')
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            return df
        return None
        
    except Exception:
        return None

def get_intraday_5min_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    chunks = []
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=30), end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)
    
    all_data = []
    for i, (chunk_start, chunk_end) in enumerate(chunks):
        retry = 0
        while retry < 3:
            df_chunk = get_intraday_5min_data_chunk(ticker, chunk_start, chunk_end)
            if df_chunk is not None:
                all_data.append(df_chunk)
                break
            retry += 1
            time.sleep(2 ** retry)
        if i < len(chunks) - 1:
            time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data).sort_index()

def main():
    try:
        ticker = "AAPL"
        tz = pytz.timezone('US/Eastern')
        now = datetime.now(tz)
        end_date = now.strftime("%Y-%m-%d")
        start_date = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        
        df = get_intraday_5min_data(ticker, start_date, end_date)
        if df.empty:
            return
        
        # Get last 20 candles
        last_20 = df.tail(20)
        candles = []
        for _, row in last_20.iterrows():
            # Convert to native Python types
            candles.append([
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ])
        
        # Format output
        formatted = []
        for i in range(0, 20, 4):
            line = ", ".join([str(c) for c in candles[i:i+4]])
            formatted.append(line)
        output = ",\n        ".join(formatted)
        
        print(output)
        
    except Exception:
        pass

if __name__ == "__main__":
    main()