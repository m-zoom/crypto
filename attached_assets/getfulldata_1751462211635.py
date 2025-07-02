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
    """
    Fetch 5-minute intraday data for a specific date range (single chunk)
    Returns None on error to allow continuation
    """
    headers = {"X-API-KEY": API_KEY}
    url = (
        f"https://api.financialdatasets.ai/prices/?ticker={ticker}"
        f"&interval=minute&interval_multiplier=5"
        f"&start_date={start_date}&end_date={end_date}"
    )
    
    try:
        print(f"Requesting {start_date} to {end_date}...")
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"⚠️ Error for {start_date}-{end_date}: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        parsed = PriceResponse(**data)
        
        df = pd.DataFrame([p.dict() for p in parsed.prices])
        if not df.empty:
            # Convert to Eastern Time
            df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('US/Eastern')
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            print(f"✅ Retrieved {len(df)} records for {start_date} to {end_date}")
            return df
            
        print(f"⚠️ No data for {start_date}-{end_date}")
        return None
        
    except Exception as e:
        print(f"⚠️ Exception for {start_date}-{end_date}: {str(e)}")
        return None

def get_intraday_5min_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch 5-minute intraday data with monthly chunking and retries
    """
    # Convert to datetime objects
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate monthly chunks
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
            df_chunk = get_intraday_5min_data_chunk(ticker, chunk_start, chunk_end)
            if df_chunk is not None:
                all_data.append(df_chunk)
                break
            retry += 1
            time.sleep(2 ** retry)  # Exponential backoff
        else:
            print(f"🚨 Failed to get data for {chunk_start} to {chunk_end} after 3 attempts")
        
        # Add delay between chunks to avoid rate limiting
        if i < len(chunks) - 1:
            time.sleep(1)
    
    if not all_data:
        return pd.DataFrame()
    
    return pd.concat(all_data).sort_index()

def main():
    try:
        ticker = "AAPL"
        start_date = "2025-01-02"
        end_date = "2025-06-20"
        
        print(f"\n{'='*50}")
        print(f"Fetching 5min data for {ticker} from {start_date} to {end_date}")
        print(f"{'='*50}\n")
        
        df = get_intraday_5min_data(ticker, start_date, end_date)
        
        if df.empty:
            print("\n❌ No data retrieved for the entire period")
            return
        
        # Calculate trading days
        trading_days = df.index.normalize().unique()
        
        print("\n" + "="*50)
        print(f"✅ Retrieval Complete!")
        print(f"Total records: {len(df):,}")
        print(f"Date range: {df.index.min()} to {df.index.max()} (ET)")
        print(f"Total trading days: {len(trading_days)}")
        print(f"Average records per day: {len(df)/len(trading_days):.1f}")
        print("="*50 + "\n")
        
        # Save to CSV
        filename = f"{ticker}_5min_{start_date.replace('-','')}_{end_date.replace('-','')}.csv"
        df.to_csv(filename)
        print(f"Data saved to {filename}")
        
        # Save to Parquet for efficiency
        parquet_name = f"{ticker}_5min_{start_date.replace('-','')}_{end_date.replace('-','')}.parquet"
        df.to_parquet(parquet_name)
        print(f"Data saved to {parquet_name} (Parquet format)")
        
        # Print sample
        print("\nFirst 5 records:")
        print(df.head())
        
        # Print last 5 records to verify end date coverage
        print("\nLast 5 records:")
        print(df.tail())
        
    except Exception as e:
        print("\n❌ Critical Error:", e)

if __name__ == "__main__":
    main()