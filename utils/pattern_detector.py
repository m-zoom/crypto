import os
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
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
        if len(candles) < 15:
            return False, 0.0
        
        # Extract price data
        lows = [candle[2] for candle in candles]
        highs = [candle[1] for candle in candles]
        closes = [candle[3] for candle in candles]
        
        # Find local minima (potential bottoms)
        local_minima = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                local_minima.append((i, lows[i]))
        
        if len(local_minima) < 3:
            return False, 0.0
        
        # Check if we have three bottoms at similar levels
        sorted_minima = sorted(local_minima, key=lambda x: x[1])
        bottom_levels = [m[1] for m in sorted_minima[:3]]
        
        # Calculate if bottoms are at similar levels (within 3% of each other)
        max_bottom = max(bottom_levels)
        min_bottom = min(bottom_levels)
        
        if max_bottom > 0 and (max_bottom - min_bottom) / max_bottom < 0.03:
            # Check if pattern shows reversal (price moving up after third bottom)
            last_bottom_idx = max([m[0] for m in sorted_minima[:3]])
            if last_bottom_idx < len(closes) - 3:
                recent_trend = closes[-1] - closes[last_bottom_idx]
                if recent_trend > 0:
                    confidence = min(85.0, 60.0 + (3 - (max_bottom - min_bottom) / max_bottom * 100) * 8)
                    return True, confidence
        
        return False, len(local_minima) * 5.0

class InverseHeadAndShouldersClassifier(BasePatternClassifier):
    """Inverse Head and Shoulders pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 15:
            return False, 0.0
        
        # Extract price data
        lows = [candle[2] for candle in candles]
        highs = [candle[1] for candle in candles]
        closes = [candle[3] for candle in candles]
        
        # Find local minima for shoulders and head
        local_minima = []
        for i in range(3, len(lows) - 3):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                local_minima.append((i, lows[i]))
        
        if len(local_minima) < 3:
            return False, 0.0
        
        # Sort by position to find chronological order
        local_minima.sort(key=lambda x: x[0])
        
        # Look for left shoulder, head, right shoulder pattern
        for i in range(len(local_minima) - 2):
            left_shoulder = local_minima[i]
            head = local_minima[i + 1]
            right_shoulder = local_minima[i + 2]
            
            # Head should be significantly lower than both shoulders
            head_depth = (min(left_shoulder[1], right_shoulder[1]) - head[1]) / head[1]
            
            if head_depth > 0.015:  # Head at least 1.5% deeper
                # Shoulders should be at similar levels (within 2%)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / max(left_shoulder[1], right_shoulder[1])
                
                if shoulder_diff < 0.02:
                    # Check for upward breakout after right shoulder
                    if right_shoulder[0] < len(closes) - 3:
                        # Look for price moving above the neckline (shoulder level)
                        neckline = max(left_shoulder[1], right_shoulder[1])
                        recent_high = max(highs[right_shoulder[0]:])
                        
                        if recent_high > neckline * 1.01:  # Break above neckline by 1%
                            confidence = min(80.0, 45.0 + head_depth * 1000 + (2 - shoulder_diff * 100) * 10)
                            return True, confidence
        
        return False, len(local_minima) * 4.0

class DoubleTopClassifier(BasePatternClassifier):
    """Double Top pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 10:
            return False, 0.0
        
        # Extract price data
        highs = [candle[1] for candle in candles]
        lows = [candle[2] for candle in candles]
        closes = [candle[3] for candle in candles]
        
        # Find local maxima (potential tops)
        local_maxima = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
                local_maxima.append((i, highs[i]))
        
        if len(local_maxima) < 2:
            return False, 0.0
        
        # Check if we have two tops at similar levels
        sorted_maxima = sorted(local_maxima, key=lambda x: x[1], reverse=True)
        top_levels = [m[1] for m in sorted_maxima[:2]]
        
        # Calculate if tops are at similar levels (within 2% of each other)
        max_top = max(top_levels)
        min_top = min(top_levels)
        
        if max_top > 0 and (max_top - min_top) / max_top < 0.02:
            # Check for valley between the tops
            top_indices = [m[0] for m in sorted_maxima[:2]]
            min_idx, max_idx = min(top_indices), max(top_indices)
            valley_low = min(lows[min_idx:max_idx+1])
            
            # Valley should be significantly lower than tops (at least 2% drop)
            if (max_top - valley_low) / max_top > 0.02:
                # Check if price is declining after second top
                last_top_idx = max(top_indices)
                if last_top_idx < len(closes) - 2:
                    recent_trend = closes[-1] - closes[last_top_idx]
                    if recent_trend < 0:
                        confidence = min(80.0, 55.0 + (2 - (max_top - min_top) / max_top * 100) * 10)
                        return True, confidence
        
        return False, len(local_maxima) * 8.0

class DoubleBottomClassifier(BasePatternClassifier):
    """Double Bottom pattern classifier"""
    def predict(self, candles: List[List[float]]) -> Tuple[bool, float]:
        if len(candles) < 10:
            return False, 0.0
        
        # Extract price data
        lows = [candle[2] for candle in candles]
        highs = [candle[1] for candle in candles]
        closes = [candle[3] for candle in candles]
        
        # Find local minima (potential bottoms)
        local_minima = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
                local_minima.append((i, lows[i]))
        
        if len(local_minima) < 2:
            return False, 0.0
        
        # Check if we have two bottoms at similar levels
        sorted_minima = sorted(local_minima, key=lambda x: x[1])
        bottom_levels = [m[1] for m in sorted_minima[:2]]
        
        # Calculate if bottoms are at similar levels (within 2% of each other)
        max_bottom = max(bottom_levels)
        min_bottom = min(bottom_levels)
        
        if max_bottom > 0 and (max_bottom - min_bottom) / max_bottom < 0.02:
            # Check for peak between the bottoms
            bottom_indices = [m[0] for m in sorted_minima[:2]]
            min_idx, max_idx = min(bottom_indices), max(bottom_indices)
            peak_high = max(highs[min_idx:max_idx+1])
            
            # Peak should be significantly higher than bottoms (at least 2% rise)
            if (peak_high - min_bottom) / min_bottom > 0.02:
                # Check if price is rising after second bottom
                last_bottom_idx = max(bottom_indices)
                if last_bottom_idx < len(closes) - 2:
                    recent_trend = closes[-1] - closes[last_bottom_idx]
                    if recent_trend > 0:
                        confidence = min(75.0, 50.0 + (2 - (max_bottom - min_bottom) / max_bottom * 100) * 10)
                        return True, confidence
        
        return False, len(local_minima) * 7.0

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
                # Check if this is a loaded ML model or our fallback classifier
                model_type_str = str(type(model))
                is_ml_model = ('sklearn' in model_type_str or 
                              hasattr(model, '_sklearn_version') or 
                              hasattr(model, 'fit') or
                              'BaseEstimator' in str(type(model).__mro__))
                
                if is_ml_model and not hasattr(model, 'pattern_name'):
                    # This is a real ML model - use it properly
                    logger.info(f"Using ML model for {pattern_name}: {type(model)}")
                    
                    # Prepare features for ML model
                    features = self._prepare_features_for_ml_model(candles)
                    if features is not None:
                        try:
                            # Get prediction from ML model
                            if hasattr(model, 'predict_proba'):
                                # Classification model with probability
                                probabilities = model.predict_proba([features])
                                prediction = model.predict([features])[0]
                                confidence = float(max(probabilities[0]) * 100)
                            elif hasattr(model, 'predict'):
                                # Regular prediction
                                prediction = model.predict([features])[0]
                                confidence = float(abs(prediction) * 100) if isinstance(prediction, (int, float)) else 50.0
                            else:
                                raise AttributeError("Model has no predict method")
                            
                            detected = bool(prediction) if isinstance(prediction, (bool, np.bool_)) else prediction > 0.5
                            
                            results[pattern_name] = {
                                'detected': detected,
                                'confidence': min(confidence, 100.0),
                                'model_type': f'ML Model ({type(model).__name__})'
                            }
                            
                            logger.info(f"{pattern_name} ML Model: {'Detected' if detected else 'Not detected'} "
                                       f"(Confidence: {confidence:.1f}%)")
                        except Exception as ml_error:
                            logger.warning(f"ML model failed for {pattern_name}: {ml_error}, using fallback")
                            # Fall through to fallback classifier
                            is_ml_model = False
                    else:
                        logger.warning(f"Failed to prepare features for {pattern_name}, using fallback")
                        is_ml_model = False
                
                # Use fallback classifier if ML model failed or wasn't available
                if not is_ml_model and hasattr(model, 'predict'):
                    # This is our fallback classifier - use it as backup
                    logger.debug(f"Using fallback classifier for {pattern_name}")
                    prediction_result = model.predict(candles)
                    
                    # Handle different return formats
                    if isinstance(prediction_result, tuple) and len(prediction_result) == 2:
                        prediction, confidence = prediction_result
                    elif isinstance(prediction_result, (list, np.ndarray)) and len(prediction_result) >= 2:
                        prediction, confidence = prediction_result[0], prediction_result[1]
                    else:
                        prediction = bool(prediction_result)
                        confidence = 50.0 if prediction else 0.0
                    
                    results[pattern_name] = {
                        'detected': bool(prediction),
                        'confidence': float(confidence),
                        'model_type': f'Fallback ({type(model).__name__})'
                    }
                    
                    logger.debug(f"{pattern_name} Fallback: {'Detected' if prediction else 'Not detected'} "
                               f"(Confidence: {confidence}%)")
                else:
                    logger.warning(f"{pattern_name} model doesn't have a 'predict' method")
                    results[pattern_name] = {
                        'error': 'Model missing predict method'
                    }
                           
            except Exception as e:
                logger.error(f"Error analyzing {pattern_name}: {str(e)}")
                logger.error(f"Model type: {type(model)}")
                results[pattern_name] = {
                    'error': str(e)
                }
        
        return results
    
    def _prepare_features_for_ml_model(self, candles: List[List[float]]) -> Optional[List[float]]:
        """Prepare features from candle data for ML model input"""
        try:
            if len(candles) < 5:
                return None
            
            # Extract OHLCV data
            opens = [c[0] for c in candles]
            highs = [c[1] for c in candles]
            lows = [c[2] for c in candles]
            closes = [c[3] for c in candles]
            volumes = [c[4] for c in candles]
            
            features = []
            
            # Basic price features
            features.extend([
                opens[-1], highs[-1], lows[-1], closes[-1], volumes[-1]  # Latest OHLCV
            ])
            
            # Price ratios and changes
            if len(candles) >= 2:
                features.extend([
                    closes[-1] / closes[-2] - 1,  # Price change ratio
                    highs[-1] / lows[-1] - 1,     # High/Low ratio
                    (closes[-1] - opens[-1]) / opens[-1],  # Body size ratio
                ])
            
            # Moving averages and trends
            if len(candles) >= 5:
                ma5 = sum(closes[-5:]) / 5
                features.extend([
                    closes[-1] / ma5 - 1,  # Distance from MA5
                    max(highs[-5:]) / min(lows[-5:]) - 1,  # 5-period high/low range
                ])
            
            # Statistical features
            if len(candles) >= 10:
                recent_closes = closes[-10:]
                mean_price = sum(recent_closes) / len(recent_closes)
                variance = sum((p - mean_price) ** 2 for p in recent_closes) / len(recent_closes)
                std_dev = variance ** 0.5
                
                features.extend([
                    (closes[-1] - mean_price) / (std_dev + 1e-8),  # Z-score
                    std_dev / mean_price,  # Coefficient of variation
                ])
            
            # Pattern-specific features
            features.extend([
                min(lows) / max(highs),  # Overall low/high ratio
                sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1]) / (len(closes) - 1),  # Up days ratio
            ])
            
            # Pad features to ensure consistent length
            while len(features) < 20:
                features.append(0.0)
            
            # Truncate if too long
            features = features[:20]
            
            logger.debug(f"Prepared {len(features)} features for ML model")
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None
    
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
