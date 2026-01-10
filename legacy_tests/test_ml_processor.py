"""
Tests for ML processor utilities.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from utils.ml_processor import MLDataProcessor, ModelTrainer, PredictionEngine, ModelEvaluator

class TestMLDataProcessor:
    """Test ML data processor."""
    
    def setup_method(self):
        """Set up test data."""
        self.processor = MLDataProcessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'ra': [150.0, 151.0, 152.0],
            'dec': [2.0, 2.1, 2.2],
            'redshift': [0.5, 0.6, 0.7],
            'mag_g': [20.0, 21.0, 22.0],
            'mag_r': [19.5, 20.5, 21.5],
            'object_type': ['galaxy', 'galaxy', 'quasar'],
            'catalog_source': ['SDSS', 'DESI', 'SDSS']
        })
    
    def test_prepare_features(self):
        """Test feature preparation."""
        result = self.processor.prepare_features(self.sample_data, 'redshift')
        
        assert 'X' in result
        assert 'y' in result
        assert 'feature_names' in result
        assert 'target_name' in result
        
        # Check that target variable is removed from features
        assert 'redshift' not in result['X'].columns
        
        # Check that features are scaled
        assert isinstance(result['X'], pd.DataFrame)
        assert len(result['X']) == len(self.sample_data)
    
    def test_prepare_features_with_missing_values(self):
        """Test feature preparation with missing values."""
        # Add missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, 'mag_g'] = np.nan
        data_with_missing.loc[1, 'object_type'] = None
        
        result = self.processor.prepare_features(data_with_missing, 'redshift')
        
        # Should handle missing values gracefully
        assert not result['X'].isnull().any().any()
        assert not result['y'].isnull().any()
    
    def test_split_dataset(self):
        """Test dataset splitting."""
        data = self.processor.prepare_features(self.sample_data, 'redshift')
        train_data, test_data = self.processor.split_dataset(data, test_size=0.5)
        
        assert 'X' in train_data
        assert 'y' in train_data
        assert 'X' in test_data
        assert 'y' in test_data
        
        # Check split sizes
        total_size = len(data['X'])
        expected_train_size = int(total_size * 0.5)
        expected_test_size = total_size - expected_train_size
        
        assert len(train_data['X']) == expected_train_size
        assert len(test_data['X']) == expected_test_size

class TestModelTrainer:
    """Test model trainer."""
    
    def setup_method(self):
        """Set up test data."""
        self.trainer = ModelTrainer()
        
        # Create sample data
        self.X_train = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        self.y_train = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5])
        self.X_test = pd.DataFrame({
            'feature1': [1.5, 2.5],
            'feature2': [3, 5]
        })
        self.y_test = pd.Series([0.75, 1.25])
    
    def test_train_random_forest(self):
        """Test training random forest model."""
        model, metrics = self.trainer.train_model(
            'random_forest',
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )
        
        assert model is not None
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert 'mae' in metrics
    
    def test_train_linear_regression(self):
        """Test training linear regression model."""
        model, metrics = self.trainer.train_model(
            'linear_regression',
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test
        )
        
        assert model is not None
        assert 'mse' in metrics
        assert 'r2' in metrics
    
    def test_train_with_custom_hyperparameters(self):
        """Test training with custom hyperparameters."""
        hyperparameters = {'n_estimators': 50, 'max_depth': 5}
        
        model, metrics = self.trainer.train_model(
            'random_forest',
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            hyperparameters=hyperparameters
        )
        
        assert model is not None
        assert model.n_estimators == 50
        assert model.max_depth == 5
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        with pytest.raises(ValueError, match="Unknown model type"):
            self.trainer.train_model(
                'invalid_model',
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test
            )

class TestPredictionEngine:
    """Test prediction engine."""
    
    def setup_method(self):
        """Set up test data."""
        self.engine = PredictionEngine()
        
        # Create a simple model
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()
        self.model.fit([[1], [2], [3]], [0.5, 1.0, 1.5])
        
        self.features = pd.DataFrame({'feature1': [2.5, 3.5]})
    
    def test_predict(self):
        """Test making predictions."""
        predictions = self.engine.predict(self.model, self.features)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(self.features)
    
    def test_predict_with_dict_features(self):
        """Test making predictions with dictionary features."""
        features_dict = {'feature1': [2.5, 3.5]}
        
        predictions = self.engine.predict(self.model, features_dict)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 2
    
    def test_predict_proba_with_classifier(self):
        """Test probability predictions with classifier."""
        from sklearn.ensemble import RandomForestClassifier
        
        classifier = RandomForestClassifier(n_estimators=10, random_state=42)
        classifier.fit([[1], [2], [3], [4]], [0, 0, 1, 1])
        
        features = pd.DataFrame({'feature1': [2.5, 3.5]})
        
        probabilities = self.engine.predict_proba(classifier, features)
        
        assert isinstance(probabilities, np.ndarray)
        assert probabilities.shape[1] == 2  # Two classes
    
    def test_predict_proba_with_regressor(self):
        """Test probability predictions with regressor should fail."""
        with pytest.raises(ValueError, match="Model does not support probability predictions"):
            self.engine.predict_proba(self.model, self.features)

class TestModelEvaluator:
    """Test model evaluator."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = ModelEvaluator()
        
        # Create sample models
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        self.model1 = LinearRegression()
        self.model1.fit([[1], [2], [3]], [0.5, 1.0, 1.5])
        
        self.model2 = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model2.fit([[1], [2], [3]], [0.5, 1.0, 1.5])
        
        self.X_test = pd.DataFrame({'feature1': [2.5, 3.5]})
        self.y_test = pd.Series([1.25, 1.75])
    
    def test_compare_models(self):
        """Test model comparison."""
        models = {
            'linear_regression': self.model1,
            'random_forest': self.model2
        }
        
        results = self.evaluator.compare_models(models, self.X_test, self.y_test)
        
        assert 'linear_regression' in results
        assert 'random_forest' in results
        assert 'mse' in results['linear_regression']
        assert 'r2' in results['linear_regression']
    
    def test_cross_validate_model(self):
        """Test cross-validation."""
        X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        
        results = self.evaluator.cross_validate_model('linear_regression', X, y, cv_folds=3)
        
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'scores' in results
        assert len(results['scores']) == 3

class TestIntegration:
    """Integration tests for ML pipeline."""
    
    def test_full_ml_pipeline(self):
        """Test complete ML pipeline."""
        # Create sample data
        data = pd.DataFrame({
            'ra': [150.0, 151.0, 152.0, 153.0, 154.0],
            'dec': [2.0, 2.1, 2.2, 2.3, 2.4],
            'redshift': [0.5, 0.6, 0.7, 0.8, 0.9],
            'mag_g': [20.0, 21.0, 22.0, 23.0, 24.0],
            'object_type': ['galaxy', 'galaxy', 'quasar', 'galaxy', 'quasar'],
            'catalog_source': ['SDSS', 'DESI', 'SDSS', 'DESI', 'SDSS']
        })
        
        # 1. Prepare features
        processor = MLDataProcessor()
        features = processor.prepare_features(data, 'redshift')
        
        # 2. Split dataset
        train_data, test_data = processor.split_dataset(features, test_size=0.4)
        
        # 3. Train model
        trainer = ModelTrainer()
        model, metrics = trainer.train_model(
            'random_forest',
            train_data['X'],
            train_data['y'],
            test_data['X'],
            test_data['y']
        )
        
        # 4. Make predictions
        engine = PredictionEngine()
        predictions = engine.predict(model, test_data['X'])
        
        # Assertions
        assert model is not None
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert len(predictions) == len(test_data['X'])
        assert all(isinstance(p, (int, float, np.integer, np.floating)) for p in predictions)
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        import joblib
        import tempfile
        
        # Create and train a model
        trainer = ModelTrainer()
        X_train = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_train = pd.Series([0.5, 1.0, 1.5, 2.0, 2.5])
        X_test = pd.DataFrame({'feature1': [2.5]})
        y_test = pd.Series([1.25])
        
        model, _ = trainer.train_model('random_forest', X_train, y_train, X_test, y_test)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
        
        try:
            joblib.dump(model, model_path)
            
            # Load model
            loaded_model = joblib.load(model_path)
            
            # Test predictions are the same
            engine = PredictionEngine()
            original_pred = engine.predict(model, X_test)
            loaded_pred = engine.predict(loaded_model, X_test)
            
            np.testing.assert_array_almost_equal(original_pred, loaded_pred)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.unlink(model_path)

if __name__ == "__main__":
    pytest.main([__file__])