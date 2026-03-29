"""
Lightweight Intrusion Detection System Models
Optimized for embedded systems (STM32, microcontrollers)
"""

import pickle
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

class LightweightIDS:
    """Base class for lightweight IDS models"""
    
    def __init__(self, name, model, scaler=None):
        self.name = name
        self.model = model
        self.scaler = scaler or StandardScaler()
        self.metrics = {}
    
    def get_model_size(self):
        """Estimate model size in KB"""
        model_bytes = pickle.dumps(self.model)
        return len(model_bytes) / 1024
    
    def save_model(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        print(f"✓ Saved {self.name} to {filepath}")
    
    def load_model(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        print(f"✓ Loaded {self.name} from {filepath}")

class TinyDecisionTree(LightweightIDS):
    """Ultra-lightweight Decision Tree - good for embedded systems"""
    def __init__(self):
        model = DecisionTreeClassifier(
            max_depth=5,  # Very shallow for small size
            min_samples_split=5,
            min_samples_leaf=2
        )
        super().__init__("TinyDecisionTree", model)

class LightRandomForest(LightweightIDS):
    """Lightweight Random Forest with limited trees"""
    def __init__(self):
        model = RandomForestClassifier(
            n_estimators=10,  # Very few trees
            max_depth=7,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        super().__init__("LightRandomForest", model)

class MicroXGBoost(LightweightIDS):
    """Micro-optimized XGBoost for embedded systems"""
    def __init__(self):
        model = XGBClassifier(
            n_estimators=5,  # Minimal trees
            max_depth=3,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            random_state=42,
            tree_method='hist',
            device='cpu'
        )
        super().__init__("MicroXGBoost", model)

class CompactExtraTrees(LightweightIDS):
    """Compact Extra Trees classifier"""
    def __init__(self):
        model = ExtraTreesClassifier(
            n_estimators=8,
            max_depth=6,
            min_samples_split=5,
            n_jobs=-1,
            random_state=42
        )
        super().__init__("CompactExtraTrees", model)

class TinyXGBoost(LightweightIDS):
    """Tiny XGBoost with extreme regularization"""
    def __init__(self):
        model = XGBClassifier(
            n_estimators=3,
            max_depth=2,
            learning_rate=0.5,
            reg_alpha=1.0,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            objective='binary:logistic',
            random_state=42
        )
        super().__init__("TinyXGBoost", model)

# Model registry for easy access
MODELS = {
    'tree': TinyDecisionTree,
    'rf': LightRandomForest,
    'xgb': MicroXGBoost,
    'et': CompactExtraTrees,
    'tiny_xgb': TinyXGBoost,
}

def get_model(model_name):
    """Get a model instance by name"""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    return MODELS[model_name]()

def list_available_models():
    """List all available models"""
    print("Available Lightweight IDS Models:")
    for name, model_class in MODELS.items():
        model = model_class()
        print(f"  {name:12} - {model.name}")

if __name__ == "__main__":
    list_available_models()
