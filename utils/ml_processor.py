"""
Machine Learning processor for astronomical data analysis.
Handles data preparation, model training, and predictions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class MLDataProcessor:
    """Data processor for ML operations on astronomical data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
    
    def prepare_features(self, df: pd.DataFrame, target_variable: str = "redshift") -> Dict[str, pd.DataFrame]:
        """Prepare features for machine learning."""
        try:
            # Define feature columns based on available data
            numeric_features = [
                'ra', 'dec', 'mag_g', 'mag_r', 'mag_i', 'mag_z',
                'redshift', 'distance', 'luminosity', 'stellar_mass',
                'metallicity', 'age', 'velocity_dispersion'
            ]
            
            categorical_features = [
                'object_type', 'catalog_source', 'galaxy_type',
                'spectral_type', 'morphology'
            ]
            
            # Select available features
            available_numeric = [col for col in numeric_features if col in df.columns]
            available_categorical = [col for col in categorical_features if col in df.columns]
            
            # Handle missing values
            for col in available_numeric:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
            
            for col in available_categorical:
                if df[col].isnull().sum() > 0:
                    df[col] = df[col].fillna('unknown')
            
            # Encode categorical variables
            X_encoded = df[available_numeric].copy()
            
            for col in available_categorical:
                if col in df.columns:
                    le = LabelEncoder()
                    X_encoded[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Remove target variable from features if present
            if target_variable in X_encoded.columns:
                X_encoded = X_encoded.drop(columns=[target_variable])
            
            # Prepare target variable
            y = None
            if target_variable in df.columns:
                y = df[target_variable].copy()
                # Handle missing values in target
                if y.isnull().sum() > 0:
                    y = y.fillna(y.median())
            else:
                # Create dummy target if not available
                y = pd.Series(np.random.normal(0.5, 0.1, len(df)), index=df.index)
                logger.warning(f"Target variable '{target_variable}' not found, using dummy data")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_encoded)
            X_scaled_df = pd.DataFrame(X_scaled, columns=X_encoded.columns, index=df.index)
            
            self.feature_columns = X_encoded.columns.tolist()
            
            return {
                "X": X_scaled_df,
                "y": y,
                "feature_names": self.feature_columns,
                "target_name": target_variable
            }
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def split_dataset(self, data: Dict[str, pd.DataFrame], 
                     test_size: float = 0.2, 
                     random_state: int = 42) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Split dataset into training and testing sets."""
        try:
            X = data["X"]
            y = data["y"]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            train_data = {
                "X": X_train,
                "y": y_train
            }
            
            test_data = {
                "X": X_test,
                "y": y_test
            }
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            raise

class ModelTrainer:
    """Trainer for machine learning models."""
    
    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor,
            "linear_regression": LinearRegression,
            "svr": SVR,
            "random_forest_classifier": RandomForestClassifier,
            "logistic_regression": LogisticRegression,
            "svc": SVC
        }
        
        self.default_params = {
            "random_forest": {"n_estimators": 100, "random_state": 42},
            "linear_regression": {},
            "svr": {"kernel": "rbf", "C": 1.0},
            "random_forest_classifier": {"n_estimators": 100, "random_state": 42},
            "logistic_regression": {"random_state": 42},
            "svc": {"kernel": "rbf", "C": 1.0}
        }
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   hyperparameters: Dict[str, Any] = None) -> Tuple[Any, Dict[str, float]]:
        """Train a machine learning model."""
        try:
            if model_type not in self.models:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Get model class and default parameters
            model_class = self.models[model_type]
            params = self.default_params[model_type].copy()
            
            # Update with custom hyperparameters
            if hyperparameters:
                params.update(hyperparameters)
            
            # Initialize and train model
            model = model_class(**params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred, model_type)
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, model_type: str) -> Dict[str, float]:
        """Calculate model performance metrics."""
        try:
            metrics = {}
            
            if "classifier" in model_type or model_type in ["logistic_regression", "svc"]:
                # Classification metrics
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                metrics["classification_report"] = classification_report(y_true, y_pred, output_dict=True)
            else:
                # Regression metrics
                metrics["mse"] = mean_squared_error(y_true, y_pred)
                metrics["rmse"] = np.sqrt(metrics["mse"])
                metrics["r2"] = r2_score(y_true, y_pred)
                metrics["mae"] = np.mean(np.abs(y_true - y_pred))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise

class PredictionEngine:
    """Engine for making predictions with trained models."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def predict(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """Make prediction using trained model."""
        try:
            # Ensure features are in correct format
            if not isinstance(features, pd.DataFrame):
                features = pd.DataFrame(features)
            
            # Make prediction
            prediction = model.predict(features)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_proba(self, model: Any, features: pd.DataFrame) -> np.ndarray:
        """Make probability prediction for classification models."""
        try:
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(features)
            else:
                raise ValueError("Model does not support probability predictions")
                
        except Exception as e:
            logger.error(f"Error making probability prediction: {e}")
            raise

class ModelEvaluator:
    """Evaluator for model performance and comparison."""
    
    @staticmethod
    def compare_models(models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compare multiple models on test data."""
        try:
            results = {}
            
            for model_name, model in models.items():
                y_pred = model.predict(X_test)
                metrics = ModelTrainer()._calculate_metrics(y_test, y_pred, model_name)
                results[model_name] = metrics
            
            return results
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            raise
    
    @staticmethod
    def cross_validate_model(model_type: str, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5) -> Dict[str, float]:
        """Perform cross-validation on model."""
        try:
            from sklearn.model_selection import cross_val_score
            
            model_class = ModelTrainer().models[model_type]
            model = model_class()
            
            # Perform cross-validation
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
            
            return {
                "mean_score": cv_scores.mean(),
                "std_score": cv_scores.std(),
                "scores": cv_scores.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error in cross-validation: {e}")
            raise