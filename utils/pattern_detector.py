import os
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class BasePatternClassifier:
    """Base class for pattern classifiers"""
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
    
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        """Predict pattern and return confidence"""
        # Default implementation returns no detection
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

class PatternDetector:
    """Main pattern detection class"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all pattern detection models with error handling"""
        model_files = [
            ("triple_bottom_model.pkl", "Triple Bottom"),
            ("inverse_head_and_shoulders_model.pkl", "Inverse Head And Shoulders"),
            ("double_top_model.pkl", "Double Top"),
            ("double_bottom_model.pkl", "Double Bottom")
        ]
        
        # Also check for files with timestamps in attached_assets
        for filename in os.listdir("attached_assets"):
            if filename.endswith(".pkl"):
                if "triple_bottom" in filename:
                    model_files.append((f"../attached_assets/{filename}", "Triple Bottom"))
                elif "inverse_head_and_shoulders" in filename:
                    model_files.append((f"../attached_assets/{filename}", "Inverse Head And Shoulders"))
                elif "double_top" in filename:
                    model_files.append((f"../attached_assets/{filename}", "Double Top"))
                elif "double_bottom" in filename:
                    model_files.append((f"../attached_assets/{filename}", "Double Bottom"))
        
        # Create models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory '{self.models_dir}' not found. Creating it...")
            os.makedirs(self.models_dir)
        
        for model_file, pattern_name in model_files:
            # Try models directory first, then attached_assets
            model_paths = [
                os.path.join(self.models_dir, model_file),
                model_file if model_file.startswith("../") else os.path.join("attached_assets", model_file)
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    try:
                        with open(model_path, 'rb') as f:
                            model = pickle.load(f)
                            self.models[pattern_name] = model
                            logger.info(f"Successfully loaded model: {pattern_name} from {model_path}")
                            model_loaded = True
                            break
                    except Exception as e:
                        logger.error(f"Error loading {model_path}: {str(e)}")
            
            if not model_loaded:
                logger.warning(f"Model file for '{pattern_name}' not found in any location")
        
        if not self.models:
            logger.warning("No models were successfully loaded. Creating dummy classifiers...")
            # Create dummy classifiers for demonstration
            self.models = {
                "Triple Bottom": TripleBottomClassifier(),
                "Inverse Head And Shoulders": InverseHeadAndShouldersClassifier(),
                "Double Top": DoubleTopClassifier(),
                "Double Bottom": DoubleBottomClassifier()
            }
    
    def analyze_patterns(self, candles: List[List[float]]) -> Dict[str, Dict[str, Any]]:
        """Run candles through all pattern detection models"""
        results = {}
        
        if not candles:
            logger.warning("No candle data provided for analysis")
            return results
        
        # Log basic candle info for debugging
        logger.debug(f"Analyzing {len(candles)} candles")
        if candles:
            logger.debug(f"First candle: O={candles[0][0]:.2f}, H={candles[0][1]:.2f}, L={candles[0][2]:.2f}, C={candles[0][3]:.2f}")
            logger.debug(f"Last candle: O={candles[-1][0]:.2f}, H={candles[-1][1]:.2f}, L={candles[-1][2]:.2f}, C={candles[-1][3]:.2f}")
            price_change = ((candles[-1][3] - candles[0][0]) / candles[0][0]) * 100
            logger.debug(f"Price change: {price_change:.2f}%")
        
        for pattern_name, model in self.models.items():
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
                
                logger.debug(f"{pattern_name}: {'Detected' if prediction else 'Not detected'} "
                           f"(Confidence: {confidence}%)")
                           
            except Exception as e:
                logger.error(f"Error analyzing {pattern_name}: {str(e)}")
                results[pattern_name] = {
                    'error': str(e)
                }
        
        return results
    
    def get_pattern_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """Get summary statistics of pattern detection results"""
        summary = {
            'detected': 0,
            'not_detected': 0,
            'errors': 0,
            'total': len(results)
        }
        
        for pattern_data in results.values():
            if 'error' in pattern_data:
                summary['errors'] += 1
            elif pattern_data.get('detected', False):
                summary['detected'] += 1
            else:
                summary['not_detected'] += 1
        
        return summary
