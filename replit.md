# Forex Pattern Recognition System

## Overview
This is a Flask web application that automates forex chart pattern recognition using machine learning models. The system fetches real-time and historical financial data, detects chart patterns using pre-trained ML models, and displays results through an interactive web interface with real-time monitoring capabilities.

## System Architecture

### Frontend Architecture
- **Technology Stack**: HTML5, CSS3, JavaScript with Bootstrap 5 (dark theme)
- **Charting Library**: Chart.js for interactive candlestick charts
- **Real-time Updates**: Polling-based mechanism (checks every 5 seconds)
- **Responsive Design**: Bootstrap grid system for mobile-friendly interface

### Backend Architecture
- **Framework**: Flask with threading support for background monitoring
- **Data Fetching**: REST API integration with Financial Datasets API
- **Pattern Detection**: Scikit-learn based ML models for pattern classification
- **Session Management**: Thread-safe global state management
- **API Design**: RESTful endpoints for data retrieval and control operations

### Data Storage Solutions
- **Runtime Storage**: In-memory data structures for monitoring state and logs
- **Model Storage**: Pickle files for pre-trained ML models (double_bottom, double_top, triple_bottom, triple_top)
- **No Database**: Currently uses in-memory storage; database can be added later for persistence

## Key Components

### 1. Flask Application (app.py)
- Main application server with routing and middleware
- Thread-safe monitoring state management
- ProxyFix middleware for deployment behind reverse proxies
- Environment-based configuration with secure session handling

### 2. Data Fetcher (utils/data_fetcher.py)
- **Purpose**: Centralized data retrieval from Financial Datasets API
- **Timeframe Support**: 5min, 15min, 30min, 60min, daily intervals
- **Data Validation**: Pydantic models for API response validation
- **Error Handling**: Robust error handling with logging

### 3. Pattern Detector (utils/pattern_detector.py)
- **ML Models**: Integration with scikit-learn models for pattern detection
- **Pattern Types**: Triple Bottom, Double Top, Double Bottom, Inverse Head & Shoulders
- **Confidence Scoring**: Returns detection confidence levels
- **Modular Design**: Base class architecture for extensibility

### 4. User Interface
- **Control Panel**: Ticker selection, timeframe configuration, monitoring controls
- **Chart Display**: Real-time updating candlestick charts
- **Logging System**: Categorized logs (detected, not detected, errors)
- **Status Indicators**: Visual feedback for monitoring state

### 5. Background Threading
- **Non-blocking Operations**: Background threads for continuous monitoring
- **Thread Safety**: Mutex locks for state synchronization
- **Configurable Frequency**: Monitoring intervals based on selected timeframe

## Data Flow

1. **User Input**: Selects ticker symbol and timeframe through web interface
2. **Monitoring Start**: Flask spawns background thread for data monitoring
3. **Data Retrieval**: Thread fetches latest data from Financial Datasets API
4. **Pattern Analysis**: Data processed through ML models for pattern detection
5. **Results Storage**: Detection results and logs stored in thread-safe structures
6. **Frontend Updates**: JavaScript polls backend for latest results and updates UI
7. **Chart Rendering**: Chart.js renders updated candlestick data with pattern annotations

## External Dependencies

### APIs and Services
- **Financial Datasets API**: Primary data source for forex and stock prices
- **API Key Management**: Environment variable based configuration
- **Rate Limiting**: Built-in request throttling to respect API limits

### Python Packages
- **Flask**: Web framework and routing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations for pattern detection
- **scikit-learn**: Machine learning model framework
- **requests**: HTTP client for API communication
- **pydantic**: Data validation and serialization
- **python-dotenv**: Environment variable management

### Frontend Libraries
- **Bootstrap 5**: UI framework with dark theme
- **Chart.js**: Interactive charting library
- **Font Awesome**: Icon library for UI elements

## Deployment Strategy

### Development Setup
- **Environment**: Python virtual environment with pip dependencies
- **Configuration**: .env file for API keys and secrets
- **Debug Mode**: Flask development server with auto-reload

### Production Considerations
- **WSGI Server**: Gunicorn or uWSGI for production deployment
- **Reverse Proxy**: Nginx for static file serving and load balancing
- **Process Management**: Supervisor or systemd for service management
- **Environment Variables**: Secure secret management
- **Monitoring**: Application logging and error tracking

### Security Measures
- **Session Security**: Configurable session secrets
- **API Key Protection**: Environment-based key management
- **Input Validation**: Pydantic models for data validation
- **CORS Handling**: Proper cross-origin request handling

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

- July 02, 2025: Fixed Windows compatibility issues with model loading
- July 02, 2025: Added support for 1-minute and 2-minute timeframes
- July 02, 2025: Implemented safe pickle loading system to handle ML model class definition issues
- July 02, 2025: Created working pattern detection classifiers with actual logic for all pattern types
- July 02, 2025: Added Windows setup guide with troubleshooting steps

## Changelog

Changelog:
- July 02, 2025. Initial setup and Windows compatibility fixes