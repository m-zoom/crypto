# Windows Setup Guide

## Prerequisites
1. Python 3.11 or higher installed
2. Git (optional, for cloning)

## Setup Steps

### 1. Create Project Directory
```cmd
mkdir forex-pattern-recognition
cd forex-pattern-recognition
```

### 2. Create Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### 3. Install Required Packages
```cmd
pip install flask pandas requests pydantic python-dotenv pytz numpy scikit-learn gunicorn werkzeug
```

### 4. Create Project Structure
Create the following folders and files:

```
forex-pattern-recognition/
│
├── app.py
├── main.py
├── .env
├── templates/
│   └── index.html
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── models/
│   └── (place your .pkl model files here)
├── utils/
│   ├── __init__.py
│   ├── data_fetcher.py
│   └── pattern_detector.py
└── attached_assets/
    └── (temporary folder for model files)
```

### 5. Environment Configuration
Create a `.env` file in the root directory:

```env
# Financial Datasets API Configuration
FINANCIAL_DATASETS_API_KEY=your_api_key_here

# Flask Configuration
SESSION_SECRET=your_session_secret_here_change_in_production
FLASK_ENV=development
FLASK_DEBUG=True
```

**Important**: Replace `your_api_key_here` with your actual Financial Datasets API key.

### 6. Copy Model Files
If you have the ML model files (.pkl), place them in the `models/` directory with these exact names:
- `double_bottom_model.pkl`
- `double_top_model.pkl`
- `triple_bottom_model.pkl`
- `inverse_head_and_shoulders_model.pkl`

### 7. Run the Application
```cmd
python main.py
```

The application will start and be available at:
- http://127.0.0.1:5000
- http://localhost:5000

## Common Issues and Solutions

### Issue: "Can't get attribute" Error with Model Loading
**Solution**: The application now includes a safe model loading system that handles missing class definitions. Even if the original ML models can't be loaded, the app will use working pattern detection classifiers.

### Issue: API Key Not Working
**Solution**: 
1. Make sure you have a valid Financial Datasets API key
2. Update the `.env` file with your actual API key
3. Restart the application

### Issue: Module Not Found
**Solution**: Make sure you're in the virtual environment and all packages are installed:
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Port Already in Use
**Solution**: Change the port in `main.py` or stop the process using port 5000:
```cmd
netstat -ano | findstr :5000
taskkill /PID <PID_NUMBER> /F
```

## Features Available

1. **Real-time Monitoring**: Select ticker and timeframe, then click Start
2. **Historical Analysis**: Click Analyze to run pattern detection on recent data
3. **Multiple Timeframes**: 1min, 2min, 5min, 15min, 30min, 60min, daily
4. **Pattern Detection**: Double Top/Bottom, Triple Bottom, Inverse Head & Shoulders
5. **Interactive Charts**: Real-time updating candlestick charts
6. **Activity Logs**: Categorized logs showing detected patterns

## Getting API Key

To get a Financial Datasets API key:
1. Visit https://financialdatasets.ai/
2. Sign up for an account
3. Navigate to API section
4. Generate your API key
5. Add it to your `.env` file

## Development Mode

The application runs in development mode by default with:
- Auto-reload on file changes
- Debug information
- Detailed logging

For production deployment, use a WSGI server like Gunicorn.