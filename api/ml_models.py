# api/ml_models.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Optional, Dict, Any, List
import logging
import os
import joblib
from datetime import datetime
import uuid

# Database integration
try:
    from database.config import db
    DB_AVAILABLE = True
except ImportError:
    db = None
    DB_AVAILABLE = False

logger = logging.getLogger(__name__)
router = APIRouter()

class MLRequest(BaseModel):
    data_source: str  # "database" or "file"
    file_path: Optional[str] = None  # path to CSV file if data_source is "file"
    catalog_source: Optional[str] = None  # SDSS, DESI, etc. if data_source is "database"
    target_column: str
    model_type: str = "classification"  # "classification" or "regression"
    algorithm: str = "random_forest"  # "random_forest", "logistic_regression", "linear_regression"
    test_size: float = 0.2
    max_samples: int = 10000  # Limit for database queries

class MLResponse(BaseModel):
    task_id: str
    status: str
    message: str

class MLResultResponse(BaseModel):
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    best_params: dict
    feature_importance: Optional[Dict[str, float]] = None
    model_path: Optional[str] = None

# Global storage for ML tasks
ml_tasks_status: Dict[str, Dict[str, Any]] = {}

async def load_data_from_database(catalog_source: Optional[str] = None, max_samples: int = 10000) -> pd.DataFrame:
    """Load astronomical data from database."""
    if not DB_AVAILABLE or not db:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        # Connect to database if not connected
        if not (db.mongo_client or db.sql_connection):
            await db.connect()
        
        # Get astronomical objects
        objects = await db.get_astronomical_objects(
            limit=max_samples,
            catalog_source=catalog_source
        )
        
        if not objects:
            raise HTTPException(status_code=404, detail="No astronomical data found in database")
        
        df = pd.DataFrame(objects)
        logger.info(f"Loaded {len(df)} objects from database")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def preprocess_astronomical_data(df: pd.DataFrame, target_column: str) -> tuple:
    """Preprocess astronomical data for ML."""
    try:
        # Select relevant features for astronomical analysis
        feature_columns = []
        
        # Coordinate features
        if 'ra' in df.columns and 'dec' in df.columns:
            feature_columns.extend(['ra', 'dec'])
        
        # Cartesian coordinates if available
        if all(col in df.columns for col in ['X', 'Y', 'Z']):
            feature_columns.extend(['X', 'Y', 'Z'])
        
        # Magnitude features
        mag_columns = [col for col in df.columns if col.startswith('mag_')]
        feature_columns.extend(mag_columns)
        
        # Other astronomical features
        other_features = ['redshift', 'distance', 'luminosity', 'stellar_mass', 'metallicity']
        for feature in other_features:
            if feature in df.columns:
                feature_columns.append(feature)
        
        # Parse magnitudes from JSON if stored as JSON
        if 'magnitudes_json' in df.columns:
            try:
                import json
                for idx, mag_json in df['magnitudes_json'].items():
                    if pd.notna(mag_json):
                        mag_data = json.loads(mag_json) if isinstance(mag_json, str) else mag_json
                        for mag_key, mag_val in mag_data.items():
                            if mag_key not in df.columns:
                                df[mag_key] = np.nan
                            df.at[idx, mag_key] = mag_val
                            if mag_key not in feature_columns:
                                feature_columns.append(mag_key)
            except Exception as e:
                logger.warning(f"Error parsing magnitudes_json: {e}")
        
        # Ensure we have some features
        if not feature_columns:
            # Use all numeric columns as fallback
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_column in feature_columns:
                feature_columns.remove(target_column)
        
        if not feature_columns:
            raise ValueError("No suitable features found for ML training")
        
        # Prepare features and target
        X = df[feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Prepare target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = df[target_column].copy()
        y = y.fillna(y.median() if y.dtype in ['int64', 'float64'] else y.mode()[0] if len(y.mode()) > 0 else 0)
        
        return X, y, feature_columns
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

async def train_ml_model_task(task_id: str, request: MLRequest):
    """Background task for training ML model."""
    try:
        ml_tasks_status[task_id]["status"] = "loading_data"
        ml_tasks_status[task_id]["message"] = "Loading data..."
        
        # Load data
        if request.data_source == "database":
            df = await load_data_from_database(request.catalog_source, request.max_samples)
        elif request.data_source == "file":
            if not request.file_path or not os.path.exists(request.file_path):
                raise FileNotFoundError(f"File not found: {request.file_path}")
            df = pd.read_csv(request.file_path)
        else:
            raise ValueError("data_source must be 'database' or 'file'")
        
        ml_tasks_status[task_id]["status"] = "preprocessing"
        ml_tasks_status[task_id]["message"] = "Preprocessing data..."
        
        # Preprocess data
        X, y, feature_columns = preprocess_astronomical_data(df, request.target_column)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42
        )
        
        ml_tasks_status[task_id]["status"] = "training"
        ml_tasks_status[task_id]["message"] = "Training model..."
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Select model and parameters
        if request.model_type == "classification":
            if request.algorithm == "random_forest":
                model = RandomForestClassifier(random_state=42)
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            elif request.algorithm == "logistic_regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
                param_grid = {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["liblinear", "lbfgs"]
                }
            else:
                raise ValueError(f"Unsupported classification algorithm: {request.algorithm}")
        else:  # regression
            if request.algorithm == "random_forest":
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5]
                }
            elif request.algorithm == "linear_regression":
                model = LinearRegression()
                param_grid = {}  # No hyperparameters to tune
            else:
                raise ValueError(f"Unsupported regression algorithm: {request.algorithm}")
        
        # Train with GridSearch if parameters available
        if param_grid:
            scoring = "accuracy" if request.model_type == "classification" else "r2"
            gs = GridSearchCV(model, param_grid, cv=3, scoring=scoring, n_jobs=-1)
            gs.fit(X_train_scaled, y_train)
            best_model = gs.best_estimator_
            best_params = gs.best_params_
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = {}
        
        ml_tasks_status[task_id]["status"] = "evaluating"
        ml_tasks_status[task_id]["message"] = "Evaluating model..."
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Calculate metrics
        results = {"best_params": best_params}
        
        if request.model_type == "classification":
            results["accuracy"] = float(accuracy_score(y_test, y_pred))
            results["f1_score"] = float(f1_score(y_test, y_pred, average="weighted"))
            
            # Calculate ROC AUC if possible
            try:
                if hasattr(best_model, 'predict_proba'):
                    y_proba = best_model.predict_proba(X_test_scaled)
                    if y_proba.shape[1] == 2:  # Binary classification
                        results["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
                    else:  # Multiclass
                        results["roc_auc"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr"))
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
                results["roc_auc"] = None
        else:  # regression
            results["mse"] = float(mean_squared_error(y_test, y_pred))
            results["r2_score"] = float(r2_score(y_test, y_pred))
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            importance_dict = dict(zip(feature_columns, best_model.feature_importances_))
            # Sort by importance
            results["feature_importance"] = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Save model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_filename = f"{task_id}_model.joblib"
        model_path = os.path.join(model_dir, model_filename)
        
        # Save both model and scaler
        model_data = {
            "model": best_model,
            "scaler": scaler,
            "feature_columns": feature_columns,
            "model_type": request.model_type,
            "algorithm": request.algorithm,
            "trained_at": datetime.utcnow().isoformat()
        }
        joblib.dump(model_data, model_path)
        results["model_path"] = model_path
        
        ml_tasks_status[task_id]["status"] = "completed"
        ml_tasks_status[task_id]["message"] = "Model training completed successfully"
        ml_tasks_status[task_id]["results"] = results
        ml_tasks_status[task_id]["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"ML training task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Error in ML training task {task_id}: {e}", exc_info=True)
        ml_tasks_status[task_id]["status"] = "failed"
        ml_tasks_status[task_id]["message"] = f"Training failed: {str(e)}"
        ml_tasks_status[task_id]["error"] = str(e)
        ml_tasks_status[task_id]["completed_at"] = datetime.utcnow().isoformat()

@router.post("/train", response_model=MLResponse)
async def train_model(request: MLRequest, background_tasks: BackgroundTasks):
    """
    Train a machine learning model on astronomical data.
    Supports both database and file data sources.
    """
    try:
        # Validate request
        if request.model_type not in ["classification", "regression"]:
            raise HTTPException(status_code=400, detail="model_type must be 'classification' or 'regression'")
        
        if request.data_source not in ["database", "file"]:
            raise HTTPException(status_code=400, detail="data_source must be 'database' or 'file'")
        
        if request.data_source == "database" and not DB_AVAILABLE:
            raise HTTPException(status_code=503, detail="Database not available")
        
        if request.data_source == "file" and not request.file_path:
            raise HTTPException(status_code=400, detail="file_path required when data_source is 'file'")
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Initialize task status
        ml_tasks_status[task_id] = {
            "status": "started",
            "message": "Initializing ML training task...",
            "started_at": datetime.utcnow().isoformat(),
            "request": request.dict()
        }
        
        # Start background task
        background_tasks.add_task(train_ml_model_task, task_id, request)
        
        return MLResponse(
            task_id=task_id,
            status="started",
            message="ML training task started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting ML training: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

@router.get("/task/{task_id}")
async def get_training_status(task_id: str):
    """Get the status of a training task."""
    if task_id not in ml_tasks_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return ml_tasks_status[task_id]

@router.get("/models")
async def list_models():
    """List all trained models."""
    model_dir = "models"
    if not os.path.exists(model_dir):
        return {"models": []}
    
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith(".joblib"):
            model_path = os.path.join(model_dir, filename)
            try:
                model_data = joblib.load(model_path)
                models.append({
                    "filename": filename,
                    "model_type": model_data.get("model_type"),
                    "algorithm": model_data.get("algorithm"),
                    "trained_at": model_data.get("trained_at"),
                    "feature_count": len(model_data.get("feature_columns", [])),
                    "path": model_path
                })
            except Exception as e:
                logger.warning(f"Could not load model {filename}: {e}")
    
    return {"models": models}

@router.delete("/models/{filename}")
async def delete_model(filename: str):
    """Delete a trained model."""
    model_dir = "models"
    model_path = os.path.join(model_dir, filename)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        os.remove(model_path)
        return {"message": f"Model {filename} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting model {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")

@router.post("/predict/{model_filename}")
async def predict_with_model(model_filename: str, features: Dict[str, float]):
    """Make predictions using a trained model."""
    model_dir = "models"
    model_path = os.path.join(model_dir, model_filename)
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # Load model
        model_data = joblib.load(model_path)
        model = model_data["model"]
        scaler = model_data["scaler"]
        feature_columns = model_data["feature_columns"]
        
        # Prepare features
        feature_values = []
        for col in feature_columns:
            if col in features:
                feature_values.append(features[col])
            else:
                # Use 0 as default for missing features
                feature_values.append(0.0)
                logger.warning(f"Feature {col} not provided, using default value 0.0")
        
        # Scale features
        features_array = np.array([feature_values])
        features_scaled = scaler.transform(features_array)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability if available
        prediction_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(features_scaled)[0]
                if len(proba) == 2:  # Binary classification
                    prediction_proba = {"class_0": float(proba[0]), "class_1": float(proba[1])}
                else:  # Multiclass
                    prediction_proba = {f"class_{i}": float(p) for i, p in enumerate(proba)}
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {e}")
        
        return {
            "prediction": float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            "prediction_proba": prediction_proba,
            "model_info": {
                "filename": model_filename,
                "model_type": model_data.get("model_type"),
                "algorithm": model_data.get("algorithm"),
                "trained_at": model_data.get("trained_at")
            }
        }
        
    except Exception as e:
        logger.error(f"Error making prediction with model {model_filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
