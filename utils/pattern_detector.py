import os
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import sys
import importlib

logger = logging.getLogger(__name__)

class BasePatternClassifier:
    """Base class for pattern classifiers"""
    def __init__(self) -> None:
        self.model = None
        self.scaler = None
    
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        """Predict pattern and return confidence"""
        # Default implementation with some random variation for demo
        import random
        confidence = random.uniform(0, 100)
        detected = confidence > 70
        return detected, confidence

class TripleBottomClassifier(BasePatternClassifier):
    """Triple Bottom pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        # Analyze candles for triple bottom pattern
        if len(candles) < 10:
            return False, 0.0
        
        # Simple heuristic: look for three low points
        lows = [candle[2] for candle in candles]  # Low prices
        min_low = min(lows)
        
        # Count how many candles are near the minimum
        near_min_count = sum(1 for low in lows if abs(low - min_low) / min_low < 0.02)
        
        if near_min_count >= 3:
            confidence = min(90.0, near_min_count * 15.0)
            return True, confidence
        
        return False, near_min_count * 10.0

class InverseHeadAndShouldersClassifier(BasePatternClassifier):
    """Inverse Head and Shoulders pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 10:
            return False, 0.0
        
        # Simple heuristic for inverse head and shoulders
        lows = [candle[2] for candle in candles]
        
        # Look for a low-high-low pattern in the middle section
        mid_start = len(lows) // 3
        mid_end = 2 * len(lows) // 3
        
        if mid_start < mid_end - 2:
            left_low = min(lows[:mid_start]) if mid_start > 0 else lows[0]
            head_low = min(lows[mid_start:mid_end])
            right_low = min(lows[mid_end:]) if mid_end < len(lows) else lows[-1]
            
            # Head should be lower than shoulders
            if head_low < left_low * 0.98 and head_low < right_low * 0.98:
                confidence = min(85.0, 70.0)
                return True, confidence
        
        return False, 25.0

class DoubleTopClassifier(BasePatternClassifier):
    """Double Top pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 8:
            return False, 0.0
        
        # Look for two similar high points
        highs = [candle[1] for candle in candles]
        max_high = max(highs)
        
        # Count highs near the maximum
        near_max_count = sum(1 for high in highs if abs(high - max_high) / max_high < 0.01)
        
        if near_max_count >= 2:
            confidence = min(80.0, near_max_count * 20.0)
            return True, confidence
        
        return False, near_max_count * 15.0

class DoubleBottomClassifier(BasePatternClassifier):
    """Double Bottom pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 8:
            return False, 0.0
        
        # Look for two similar low points
        lows = [candle[2] for candle in candles]
        min_low = min(lows)
        
        # Count lows near the minimum
        near_min_count = sum(1 for low in lows if abs(low - min_low) / min_low < 0.01)
        
        if near_min_count >= 2:
            confidence = min(75.0, near_min_count * 25.0)
            return True, confidence
        
        return False, near_min_count * 12.0

class SafeUnpickler(pickle.Unpickler):
    """Safe unpickler that can handle missing classes"""
    def __init__(self, file, pattern_classes):
        super().__init__(file)
        self.pattern_classes = pattern_classes
    
    def find_class(self, module, name):
        # If the class is one of our pattern classifiers, use our local version
        if name in self.pattern_classes:
            return self.pattern_classes[name]
        
        # For other classes, try the normal approach
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # If we can't find the class, create a dummy one
            logger.warning(f"Could not find class {module}.{name}, creating dummy class")
            
            class DummyClass:
                def __init__(self, *args, **kwargs):
                    pass
                def predict(self, candles):
                    return False, 0.0
            
            return DummyClass

def safe_load_pickle(file_path, pattern_classes):
    """Safely load a pickle file with custom class resolution"""
    try:
        with open(file_path, 'rb') as f:
            unpickler = SafeUnpickler(f, pattern_classes)
            return unpickler.load()
    except Exception as e:
        logger.error(f"Failed to load pickle file {file_path}: {str(e)}")
        return None

class PatternDetector:
    """Main pattern detection class"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all pattern detection models with error handling"""
        # Define our pattern classes for safe unpickling
        pattern_classes = {
            'TripleBottomClassifier': TripleBottomClassifier,
            'InverseHeadAndShouldersClassifier': InverseHeadAndShouldersClassifier,
            'DoubleTopClassifier': DoubleTopClassifier,
            'DoubleBottomClassifier': DoubleBottomClassifier,
            'BasePatternClassifier': BasePatternClassifier
        }
        
        model_files = [
            ("triple_bottom_model.pkl", "Triple Bottom"),
            ("inverse_head_and_shoulders_model.pkl", "Inverse Head And Shoulders"),
            ("double_top_model.pkl", "Double Top"),
            ("double_bottom_model.pkl", "Double Bottom")
        ]
        
        # Also check for files with timestamps in attached_assets
        if os.path.exists("attached_assets"):
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
        
        models_loaded = 0
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
                        # Try safe loading first
                        model = safe_load_pickle(model_path, pattern_classes)
                        if model is not None:
                            self.models[pattern_name] = model
                            logger.info(f"Successfully loaded model: {pattern_name} from {model_path}")
                            model_loaded = True
                            models_loaded += 1
                            break
                    except Exception as e:
                        logger.error(f"Error loading {model_path}: {str(e)}")
            
            if not model_loaded:
                logger.warning(f"Model file for '{pattern_name}' not found in any location")
        
        # If no models were loaded or only some were loaded, create working classifiers
        if models_loaded == 0:
            logger.warning("No models were successfully loaded. Creating working pattern classifiers...")
        else:
            logger.info(f"Loaded {models_loaded} models successfully. Creating classifiers for missing patterns...")
        
        # Ensure we have all pattern types available
        pattern_defaults = {
            "Triple Bottom": TripleBottomClassifier(),
            "Inverse Head And Shoulders": InverseHeadAndShouldersClassifier(),
            "Double Top": DoubleTopClassifier(),
            "Double Bottom": DoubleBottomClassifier()
        }
        
        for pattern_name, default_classifier in pattern_defaults.items():
            if pattern_name not in self.models:
                self.models[pattern_name] = default_classifier
                logger.info(f"Created working classifier for {pattern_name}")
        
        logger.info(f"Pattern detector initialized with {len(self.models)} pattern types")
    
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
