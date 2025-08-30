"""
ML Analysis API module for astronomical data processing and machine learning.
Provides endpoints for dataset preparation, model training, and analysis.
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio
import pickle
import joblib
from pathlib import Path

# Import ML utilities
from utils.ml_processor import MLDataProcessor, ModelTrainer, PredictionEngine
from database.config import db

logger = logging.getLogger(__name__)
router = APIRouter()

# ML models storage
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Background jobs tracking
ml_jobs = {}

class MLAnalysisService:
    """Service for ML analysis operations."""
    
    def __init__(self):
        self.data_processor = MLDataProcessor()
        self.model_trainer = ModelTrainer()
        self.prediction_engine = PredictionEngine()
    
    async def prepare_dataset(self, catalog_sources: List[str] = None, 
                            target_variable: str = "redshift",
                            test_size: float = 0.2) -> Dict[str, Any]:
        """Prepare ML-ready dataset from astronomical catalogs."""
        try:
            # Get data from database
            objects = await db.get_astronomical_objects(limit=10000)
            if not objects:
                raise ValueError("No astronomical objects found in database")
            
            # Convert to DataFrame
            df = pd.DataFrame(objects)
            
            # Prepare features
            features = self.data_processor.prepare_features(df, target_variable)
            
            # Split dataset
            train_data, test_data = self.data_processor.split_dataset(
                features, test_size=test_size
            )
            
            return {
                "train_samples": len(train_data["X"]),
                "test_samples": len(test_data["X"]),
                "features": list(train_data["X"].columns),
                "target_variable": target_variable,
                "dataset_info": {
                    "total_objects": len(df),
                    "catalog_sources": df["catalog_source"].unique().tolist(),
                    "feature_count": len(train_data["X"].columns)
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {e}")
            raise HTTPException(status_code=500, detail=f"Dataset preparation failed: {str(e)}")
    
    async def train_model(self, model_type: str = "random_forest",
                         target_variable: str = "redshift",
                         hyperparameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train ML model on astronomical data."""
        try:
            # Prepare dataset
            dataset_info = await self.prepare_dataset(target_variable=target_variable)
            
            # Get prepared data
            objects = await db.get_astronomical_objects(limit=10000)
            df = pd.DataFrame(objects)
            features = self.data_processor.prepare_features(df, target_variable)
            train_data, test_data = self.data_processor.split_dataset(features)
            
            # Train model
            model, metrics = self.model_trainer.train_model(
                model_type=model_type,
                X_train=train_data["X"],
                y_train=train_data["y"],
                X_test=test_data["X"],
                y_test=test_data["y"],
                hyperparameters=hyperparameters
            )
            
            # Save model
            model_path = MODELS_DIR / f"{model_type}_{target_variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            joblib.dump(model, model_path)
            
            return {
                "model_path": str(model_path),
                "model_type": model_type,
                "target_variable": target_variable,
                "metrics": metrics,
                "dataset_info": dataset_info
            }
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")
    
    async def make_prediction(self, model_path: str, features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using trained model."""
        try:
            # Load model
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = joblib.load(model_path)
            
            # Prepare features
            feature_df = pd.DataFrame([features])
            
            # Make prediction
            prediction = self.prediction_engine.predict(model, feature_df)
            
            return {
                "prediction": prediction.tolist(),
                "model_path": model_path,
                "input_features": features,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize service
ml_service = MLAnalysisService()

@router.post("/prepare-dataset", summary="Prepare ML-ready dataset from astronomical catalogs")
async def prepare_ml_dataset(
    catalog_sources: Optional[List[str]] = Query(None, description="Catalog sources to include"),
    target_variable: str = Query("redshift", description="Target variable for ML"),
    test_size: float = Query(0.2, description="Test set size ratio")
):
    """Prepare machine learning dataset from astronomical catalogs."""
    return await ml_service.prepare_dataset(
        catalog_sources=catalog_sources,
        target_variable=target_variable,
        test_size=test_size
    )

@router.post("/train-model", summary="Train machine learning model")
async def train_ml_model(
    model_type: str = Query("random_forest", description="Type of ML model"),
    target_variable: str = Query("redshift", description="Target variable"),
    hyperparameters: Optional[Dict[str, Any]] = None
):
    """Train machine learning model on astronomical data."""
    return await ml_service.train_model(
        model_type=model_type,
        target_variable=target_variable,
        hyperparameters=hyperparameters
    )

@router.post("/predict", summary="Make prediction using trained model")
async def make_ml_prediction(
    model_path: str = Query(..., description="Path to trained model file"),
    features: Dict[str, float] = Query(..., description="Input features for prediction")
):
    """Make prediction using trained machine learning model."""
    return await ml_service.make_prediction(model_path, features)

@router.get("/models", summary="List available trained models")
async def list_models():
    """List all available trained models."""
    try:
        models = []
        for model_file in MODELS_DIR.glob("*.pkl"):
            models.append({
                "name": model_file.name,
                "path": str(model_file),
                "size": model_file.stat().st_size,
                "created": datetime.fromtimestamp(model_file.stat().st_ctime).isoformat()
            })
        
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.get("/dataset-statistics", summary="Get dataset statistics for ML")
async def get_dataset_statistics():
    """Get comprehensive statistics about available data for ML."""
    try:
        objects = await db.get_astronomical_objects(limit=1000)
        if not objects:
            return {"message": "No data available", "statistics": {}}
        
        df = pd.DataFrame(objects)
        
        stats = {
            "total_objects": len(df),
            "catalog_sources": df["catalog_source"].value_counts().to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist()
        }
        
        return {"statistics": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

@router.post("/upload-model", summary="Upload trained model file")
async def upload_model(
    model_file: UploadFile = File(...),
    model_type: str = Query(..., description="Type of model"),
    target_variable: str = Query(..., description="Target variable")
):
    """Upload a trained model file."""
    try:
        if not model_file.filename.endswith('.pkl'):
            raise HTTPException(status_code=400, detail="Only .pkl files are supported")
        
        # Save uploaded file
        model_path = MODELS_DIR / f"{model_type}_{target_variable}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(model_path, "wb") as f:
            content = await model_file.read()
            f.write(content)
        
        return {
            "message": "Model uploaded successfully",
            "model_path": str(model_path),
            "filename": model_file.filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading model: {str(e)}")

@router.delete("/models/{model_name}", summary="Delete trained model")
async def delete_model(model_name: str):
    """Delete a trained model file."""
    try:
        model_path = MODELS_DIR / model_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_path.unlink()
        return {"message": f"Model {model_name} deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting model: {str(e)}")