import os
import time
import logging
import pickle
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
import pytz
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    filename='pattern_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())  # Also log to console

# Configuration
TICKER = "AAPL"
MODELS_DIR = "models"
MODEL_FILES = [
    "triple_bottom_model.pkl",
    "inverse_head_and_shoulders_model.pkl",
    "double_top_model.pkl",
    "double_bottom_model.pkl"
]
API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY", "your_api_key_here")
LOOKBACK_CANDLES = 20  # Number of candles to analyze

# Continuous monitoring configuration
MONITORING_ENABLED = True  # Set to True for continuous monitoring
MONITOR_FREQUENCY = 5 * 60  # Check every 5 minutes (in seconds)
TIMEFRAME = "5min"  # Supported: "5min", "15min", "30min", "60min", "daily"
MAX_DURATION = 24 * 60 * 60  # Maximum monitoring duration (24 hours in seconds)

# Pattern Detection Classifier Classes
class BasePatternClassifier:
    """Base class for pattern classifiers"""
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
    
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        """Predict pattern and return confidence"""
        # This should be implemented by each specific classifier
        # For now, return dummy values
        return False, 0.0

class TripleBottomClassifier(BasePatternClassifier):
    """Triple Bottom pattern classifier"""
    pass

class InverseHeadAndShouldersClassifier(BasePatternClassifier):
    """Inverse Head and Shoulders pattern classifier"""
    pass

class DoubleTopClassifier(BasePatternClassifier):
    """Double Top pattern classifier"""
    pass

class DoubleBottomClassifier(BasePatternClassifier):
    """Double Bottom pattern classifier"""
    pass

class Price(BaseModel):
    """Individual price data point"""
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class PriceResponse(BaseModel):
    """API response containing list of prices"""
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
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        parsed = PriceResponse(**data)
        
        df = pd.DataFrame([p.model_dump() for p in parsed.prices])
        if not df.empty:
            df["time"] = pd.to_datetime(df["time"]).dt.tz_convert('US/Eastern')
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
            return df
        return None
        
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return None

def get_recent_data(ticker: str, num_candles: int, timeframe: str) -> Optional[List[List[float]]]:
    """Get recent OHLCV data for pattern detection"""
    tz = pytz.timezone('US/Eastern')
    end_date = datetime.now(tz)
    
    # Calculate start date based on timeframe
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
    
    return candles

def load_models(models_dir: str, model_files: List[str]) -> Dict[str, Any]:
    """Load all pattern detection models with better error handling"""
    models = {}
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        logger.warning(f"Models directory '{models_dir}' not found. Creating it...")
        os.makedirs(models_dir)
        return models
    
    for model_file in model_files:
        model_path = os.path.join(models_dir, model_file)
        
        # Check if file exists
        if not os.path.exists(model_path):
            logger.warning(f"Model file '{model_file}' not found in {models_dir}")
            continue
            
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                pattern_name = model_file.split('_model')[0].replace('_', ' ').title()
                models[pattern_name] = model
                logger.info(f"Successfully loaded model: {pattern_name}")
        except AttributeError as e:
            logger.error(f"Class definition missing for {model_file}: {str(e)}")
            logger.info(f"You need to import or define the classifier classes before loading {model_file}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {str(e)}")
    
    if not models:
        logger.warning("No models were successfully loaded.")
        logger.info("To fix this:")
        logger.info("1. Make sure the model files exist in the 'models' directory")
        logger.info("2. Import the original classifier classes that were used to train these models")
        logger.info("3. Or retrain the models with the current script's classifier classes")
    
    return models

def analyze_patterns(candles: List[List[float]], models: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Run candles through all pattern detection models"""
    results = {}
    if not candles:
        return results
    
    # Log some basic candle info for debugging
    if candles:
        logger.info(f"Candle data summary:")
        logger.info(f"  First candle: Open={candles[0][0]:.2f}, High={candles[0][1]:.2f}, Low={candles[0][2]:.2f}, Close={candles[0][3]:.2f}")
        logger.info(f"  Last candle:  Open={candles[-1][0]:.2f}, High={candles[-1][1]:.2f}, Low={candles[-1][2]:.2f}, Close={candles[-1][3]:.2f}")
        price_change = ((candles[-1][3] - candles[0][0]) / candles[0][0]) * 100
        logger.info(f"  Price change: {price_change:.2f}%")
    
    for pattern_name, model in models.items():
        try:
            # Check if model has the expected predict method
            if not hasattr(model, 'predict'):
                logger.warning(f"{pattern_name} model doesn't have a 'predict' method")
                results[pattern_name] = {
                    'error': 'Model missing predict method'
                }
                continue
            
            # Try to get prediction
            prediction_result = model.predict(candles)
            
            # Handle different return formats
            if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                prediction, confidence = prediction_result
            elif isinstance(prediction_result, (list, np.ndarray)) and len(prediction_result) >= 2:
                prediction, confidence = prediction_result[0], prediction_result[1]
            else:
                # Handle single value or unexpected format
                prediction = bool(prediction_result)
                confidence = 50.0 if prediction else 0.0
                logger.warning(f"{pattern_name} returned unexpected format: {type(prediction_result)}")
            
            # Ensure confidence is a number
            if confidence is None or not isinstance(confidence, (int, float)):
                confidence = 0.0
            
            results[pattern_name] = {
                'detected': bool(prediction),
                'confidence': float(confidence),
                'model_type': type(model).__name__
            }
            
            logger.info(f"{pattern_name}: {'Detected' if prediction else 'Not detected'} "
                        f"(Confidence: {confidence}%)")
                        
        except Exception as e:
            logger.error(f"Error analyzing {pattern_name}: {str(e)}")
            logger.error(f"Model type: {type(model)}")
            logger.error(f"Model attributes: {dir(model)}")
            results[pattern_name] = {
                'error': str(e)
            }
    return results

def inspect_models(models: Dict[str, Any]) -> None:
    """Inspect loaded models to understand their structure"""
    logger.info("=" * 50)
    logger.info("MODEL INSPECTION")
    logger.info("=" * 50)
    
    for pattern_name, model in models.items():
        logger.info(f"\n{pattern_name} Model:")
        logger.info(f"  Type: {type(model).__name__}")
        logger.info(f"  Methods: {[method for method in dir(model) if not method.startswith('_')]}")
        
        # Check for common attributes
        common_attrs = ['model', 'scaler', 'features', 'threshold', 'n_features_']
        for attr in common_attrs:
            if hasattr(model, attr):
                attr_value = getattr(model, attr)
                logger.info(f"  {attr}: {type(attr_value)} - {str(attr_value)[:100]}...")
        
        # Try a test prediction to see what happens
        try:
            test_candles = [[100, 101, 99, 100.5, 1000]] * 5  # Simple test data
            test_result = model.predict(test_candles)
            logger.info(f"  Test prediction result: {test_result} (type: {type(test_result)})")
        except Exception as e:
            logger.info(f"  Test prediction failed: {str(e)}")
    
    logger.info("=" * 50)

def log_detection_results(results: Dict[str, Dict[str, Any]]) -> None:
    """Log detection results in a structured format"""
    header = f"\n{' Pattern Detection Results ':=^80}"
    logger.info(header)
    
    for pattern, data in results.items():
        if 'error' in data:
            logger.info(f"{pattern:>25}: ERROR - {data['error']}")
        else:
            status = "DETECTED" if data['detected'] else "Not detected"
            logger.info(f"{pattern:>25}: {status} ({data['confidence']}% confidence)")
    
    logger.info("=" * 80)

def run_detection_cycle(models: Dict[str, Any]) -> None:
    """Run a single detection cycle"""
    logger.info(f"\n{' Starting Detection Cycle ':=^80}")
    logger.info(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Timeframe: {TIMEFRAME}, Lookback: {LOOKBACK_CANDLES} candles")
    
    # 1. Get recent price data
    logger.info(f"Fetching recent data for {TICKER}...")
    candles = get_recent_data(TICKER, LOOKBACK_CANDLES, TIMEFRAME)
    
    if not candles:
        logger.error("Failed to get price data. Creating dummy data for testing...")
        candles = create_dummy_candles(LOOKBACK_CANDLES)
    
    logger.info(f"Retrieved {len(candles)} candles for analysis")
    
    # 2. Analyze patterns
    results = analyze_patterns(candles, models)
    
    # 3. Log results
    log_detection_results(results)
    logger.info(f"{' Cycle Completed ':=^80}\n")

def create_demo_models() -> Dict[str, Any]:
    """Create simple demo models for testing when real models can't be loaded"""
    demo_models = {}
    
    class DemoPatternDetector:
        def __init__(self, pattern_name: str) -> None:
            self.pattern_name = pattern_name
        
        def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
            # Simple demo logic - randomly detect patterns based on price movement
            if len(candles) < 3:
                return False, 0.0
            
            # Get recent price trend
            recent_closes = [candle[3] for candle in candles[-5:]]  # Last 5 closes
            price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0] * 100
            
            # Simple heuristics for demo
            if "bottom" in self.pattern_name.lower():
                # Bottom patterns more likely when price is declining
                detected = price_change < -1.0
                confidence = min(abs(price_change) * 10, 85) if detected else 15
            elif "top" in self.pattern_name.lower():
                # Top patterns more likely when price is rising then declining
                detected = price_change > 1.0
                confidence = min(abs(price_change) * 8, 80) if detected else 20
            else:
                # Other patterns
                detected = abs(price_change) > 0.5
                confidence = min(abs(price_change) * 12, 75) if detected else 25
            
            return detected, round(confidence, 1)
    
    patterns = ["Triple Bottom", "Inverse Head And Shoulders", "Double Top", "Double Bottom"]
    for pattern in patterns:
        demo_models[pattern] = DemoPatternDetector(pattern)
        logger.info(f"Created demo detector: {pattern}")
    
    return demo_models

def create_dummy_candles(num_candles: int) -> List[List[float]]:
    """Create dummy OHLCV data for testing"""
    import random
    
    base_price = 150.0
    candles = []
    
    for i in range(num_candles):
        # Create realistic OHLCV data
        open_price = base_price + random.uniform(-2, 2)
        close_price = open_price + random.uniform(-3, 3)
        high_price = max(open_price, close_price) + random.uniform(0, 2)
        low_price = min(open_price, close_price) - random.uniform(0, 1.5)
        volume = random.randint(100000, 1000000)
        
        candles.append([open_price, high_price, low_price, close_price, volume])
        base_price = close_price  # Next candle starts where this one ended
    
    return candles

def main() -> None:
    # 1. Load models
    logger.info(f"Starting pattern detection for {TICKER}")
    logger.info(f"Monitoring configuration:")
    logger.info(f"  Enabled: {'Yes' if MONITORING_ENABLED else 'No'}")
    logger.info(f"  Frequency: {MONITOR_FREQUENCY} seconds")
    logger.info(f"  Timeframe: {TIMEFRAME}")
    logger.info(f"  Max duration: {MAX_DURATION} seconds")
    
    models = load_models(MODELS_DIR, MODEL_FILES)
    
    if not models:
        logger.warning("No models loaded. Running in demo mode...")
        logger.info("Creating demo pattern detectors...")
        models = create_demo_models()
    else:
        # Inspect the loaded models
        inspect_models(models)
    
    # 2. Run detection cycles
    start_time = time.time()
    cycle_count = 0
    
    while MONITORING_ENABLED:
        cycle_count += 1
        logger.info(f"\n{' Starting Detection Cycle ':=^80}")
        logger.info(f"Cycle: {cycle_count}, Elapsed: {time.time() - start_time:.0f} seconds")
        
        try:
            run_detection_cycle(models)
        except Exception as e:
            logger.error(f"Error during detection cycle: {str(e)}")
        
        # Check if we've exceeded max duration
        if (time.time() - start_time) > MAX_DURATION:
            logger.info("Maximum monitoring duration reached. Stopping.")
            break
            
        # Wait for next cycle
        logger.info(f"Next detection in {MONITOR_FREQUENCY} seconds...")
        time.sleep(MONITOR_FREQUENCY)
    
    logger.info("Pattern detection completed")

if __name__ == "__main__":
    main()