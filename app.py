import os
import threading
import time
import logging
from datetime import datetime
from flask import Flask, render_template, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix
from utils.data_fetcher import get_recent_data
from utils.pattern_detector import PatternDetector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Global state for monitoring (thread-safe)
monitoring_state = {
    'is_monitoring': False,
    'ticker': None,
    'timeframe': None,
    'logs': [],
    'last_detection_results': None,
    'chart_data': None,
    'thread': None
}

# Lock for thread safety
monitor_lock = threading.Lock()

# Initialize pattern detector
pattern_detector = None

def get_monitoring_frequency_seconds(timeframe):
    """Convert timeframe to monitoring frequency in seconds"""
    frequency_map = {
        '5min': 5 * 60,
        '15min': 15 * 60, 
        '30min': 30 * 60,
        '60min': 60 * 60,
        'daily': 24 * 60 * 60
    }
    return frequency_map.get(timeframe, 5 * 60)

def monitoring_loop():
    """Background monitoring loop"""
    global monitoring_state, pattern_detector
    
    while True:
        with monitor_lock:
            if not monitoring_state['is_monitoring']:
                logger.info("Monitoring stopped")
                break
                
            ticker = monitoring_state['ticker']
            timeframe = monitoring_state['timeframe']
        
        try:
            # Fetch recent data (last 20 candles for pattern detection)
            logger.info(f"Fetching data for {ticker} - {timeframe}")
            candles = get_recent_data(ticker, 20, timeframe)
            
            if candles is None:
                error_msg = f"{datetime.now().strftime('%H:%M:%S')} - ERROR: Failed to fetch data for {ticker}"
                logger.error(error_msg)
                with monitor_lock:
                    monitoring_state['logs'].append({
                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                        'type': 'error',
                        'message': f"Failed to fetch data for {ticker}"
                    })
                time.sleep(30)  # Wait before retrying
                continue
            
            # Run pattern detection
            logger.info("Running pattern detection")
            results = pattern_detector.analyze_patterns(candles)
            
            # Update monitoring state
            with monitor_lock:
                monitoring_state['last_detection_results'] = results
                monitoring_state['chart_data'] = candles
                
                # Add detection logs
                timestamp = datetime.now().strftime('%H:%M:%S')
                for pattern_name, pattern_data in results.items():
                    if 'error' in pattern_data:
                        monitoring_state['logs'].append({
                            'timestamp': timestamp,
                            'type': 'error',
                            'message': f"{pattern_name}: {pattern_data['error']}"
                        })
                    else:
                        detected = pattern_data['detected']
                        confidence = pattern_data['confidence']
                        log_type = 'detected' if detected and confidence > 70 else ('low_confidence' if 30 <= confidence <= 70 else 'not_detected')
                        
                        monitoring_state['logs'].append({
                            'timestamp': timestamp,
                            'type': log_type,
                            'pattern': pattern_name,
                            'confidence': confidence,
                            'message': f"{pattern_name}: {'Detected' if detected else 'Not detected'} ({confidence:.1f}% confidence)"
                        })
                
                # Keep only last 100 log entries
                monitoring_state['logs'] = monitoring_state['logs'][-100:]
            
            logger.info(f"Detection cycle completed. Results: {len(results)} patterns analyzed")
            
        except Exception as e:
            error_msg = f"Error in monitoring loop: {str(e)}"
            logger.error(error_msg)
            with monitor_lock:
                monitoring_state['logs'].append({
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'type': 'error',
                    'message': error_msg
                })
        
        # Sleep based on timeframe
        frequency = get_monitoring_frequency_seconds(timeframe)
        logger.info(f"Sleeping for {frequency} seconds")
        time.sleep(frequency)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start real-time monitoring"""
    global monitoring_state
    
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    timeframe = data.get('timeframe', '5min')
    
    with monitor_lock:
        if monitoring_state['is_monitoring']:
            return jsonify({'status': 'error', 'message': 'Monitoring already running'})
        
        monitoring_state['is_monitoring'] = True
        monitoring_state['ticker'] = ticker
        monitoring_state['timeframe'] = timeframe
        monitoring_state['logs'] = []
        monitoring_state['last_detection_results'] = None
        monitoring_state['chart_data'] = None
        
        # Add start log
        monitoring_state['logs'].append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'info',
            'message': f'Started monitoring {ticker} on {timeframe} timeframe'
        })
    
    # Start monitoring thread
    thread = threading.Thread(target=monitoring_loop, daemon=True)
    thread.start()
    monitoring_state['thread'] = thread
    
    logger.info(f"Started monitoring {ticker} on {timeframe}")
    return jsonify({'status': 'success', 'message': f'Started monitoring {ticker} on {timeframe}'})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop real-time monitoring"""
    with monitor_lock:
        if not monitoring_state['is_monitoring']:
            return jsonify({'status': 'error', 'message': 'Monitoring not running'})
        
        monitoring_state['is_monitoring'] = False
        monitoring_state['logs'].append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'info',
            'message': 'Stopped monitoring'
        })
    
    logger.info("Stopped monitoring")
    return jsonify({'status': 'success', 'message': 'Monitoring stopped'})

@app.route('/get_monitoring_data')
def get_monitoring_data():
    """Get current monitoring data for frontend polling"""
    with monitor_lock:
        chart_data_formatted = None
        if monitoring_state['chart_data']:
            # Format chart data for Chart.js candlestick chart
            chart_data_formatted = []
            for i, candle in enumerate(monitoring_state['chart_data']):
                chart_data_formatted.append({
                    'x': i,  # Use index for x-axis
                    'o': candle[0],  # open
                    'h': candle[1],  # high
                    'l': candle[2],  # low
                    'c': candle[3]   # close
                })
        
        return jsonify({
            'is_monitoring': monitoring_state['is_monitoring'],
            'ticker': monitoring_state['ticker'],
            'timeframe': monitoring_state['timeframe'],
            'logs': monitoring_state['logs'],
            'results': monitoring_state['last_detection_results'],
            'chart_data': chart_data_formatted
        })

@app.route('/get_historical_analysis', methods=['POST'])
def get_historical_analysis():
    """Get historical pattern analysis for a ticker"""
    data = request.get_json()
    ticker = data.get('ticker', 'AAPL')
    timeframe = data.get('timeframe', '5min')
    
    try:
        # Fetch recent data for analysis
        candles = get_recent_data(ticker, 50, timeframe)  # Get more candles for historical analysis
        
        if candles is None:
            return jsonify({'status': 'error', 'message': 'Failed to fetch historical data'})
        
        # Run pattern detection
        results = pattern_detector.analyze_patterns(candles)
        
        # Format chart data
        chart_data = []
        for i, candle in enumerate(candles):
            chart_data.append({
                'x': i,
                'o': candle[0],
                'h': candle[1], 
                'l': candle[2],
                'c': candle[3]
            })
        
        return jsonify({
            'status': 'success',
            'results': results,
            'chart_data': chart_data,
            'ticker': ticker,
            'timeframe': timeframe
        })
        
    except Exception as e:
        logger.error(f"Error in historical analysis: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

def initialize_app():
    """Initialize the application components"""
    global pattern_detector
    
    try:
        pattern_detector = PatternDetector()
        logger.info("Pattern detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pattern detector: {str(e)}")
        pattern_detector = PatternDetector()  # Initialize with empty models

if __name__ == '__main__':
    initialize_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
