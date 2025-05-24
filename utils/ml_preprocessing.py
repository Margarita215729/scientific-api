"""
Machine Learning preprocessing module for astronomical data.
Prepares datasets for training ML models with proper feature engineering,
normalization, and train/test splits.
"""

import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
import zipfile
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ML_DIR = "galaxy_data/ml_ready"
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler, 
    "robust": RobustScaler
}

class AstronomicalMLPreprocessor:
    """Main class for ML preprocessing of astronomical data."""
    
    def __init__(self):
        os.makedirs(ML_DIR, exist_ok=True)
        self.feature_columns = []
        self.target_column = None
        self.scaler = None
        self.imputer = None
        
    def load_data(self) -> pd.DataFrame:
        """Load astronomical data from processed files."""
        logger.info("Loading astronomical data for ML preprocessing...")
        
        # Try to load merged data first
        merged_path = "galaxy_data/processed/merged_real_galaxies.csv"
        
        if os.path.exists(merged_path):
            df = pd.read_csv(merged_path)
            logger.info(f"Loaded merged data: {len(df)} objects")
        else:
            # Load individual catalogs
            individual_files = [
                "galaxy_data/processed/sdss_real.csv",
                "galaxy_data/processed/euclid_real.csv", 
                "galaxy_data/processed/desi_real.csv",
                "galaxy_data/processed/des_real.csv"
            ]
            
            dataframes = []
            for file_path in individual_files:
                if os.path.exists(file_path):
                    catalog_df = pd.read_csv(file_path)
                    dataframes.append(catalog_df)
                    logger.info(f"Loaded {os.path.basename(file_path)}: {len(catalog_df)} objects")
            
            if not dataframes:
                raise Exception("No astronomical data files found")
            
            df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Combined data: {len(df)} total objects")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for machine learning."""
        logger.info("Engineering features for ML...")
        
        # Basic astronomical features
        feature_df = df.copy()
        
        # Color indices (magnitude differences)
        if "magnitude_g" in df.columns and "magnitude_r" in df.columns:
            feature_df["color_g_r"] = df["magnitude_g"] - df["magnitude_r"]
        
        if "magnitude_r" in df.columns and "magnitude_i" in df.columns:
            feature_df["color_r_i"] = df["magnitude_r"] - df["magnitude_i"]
        
        if "magnitude_g" in df.columns and "magnitude_i" in df.columns:
            feature_df["color_g_i"] = df["magnitude_g"] - df["magnitude_i"]
        
        # Distance-related features
        if "redshift" in df.columns:
            # Logarithmic redshift
            feature_df["log_redshift"] = np.log10(df["redshift"] + 0.001)
            
            # Redshift bins
            feature_df["redshift_bin"] = pd.cut(df["redshift"], 
                                              bins=[0, 0.1, 0.3, 0.7, 1.0, 2.0, 5.0],
                                              labels=[0, 1, 2, 3, 4, 5])
        
        # Angular features
        if "RA" in df.columns:
            feature_df["ra_sin"] = np.sin(np.radians(df["RA"]))
            feature_df["ra_cos"] = np.cos(np.radians(df["RA"]))
            feature_df["ra_normalized"] = df["RA"] / 360.0
        
        if "DEC" in df.columns:
            feature_df["dec_sin"] = np.sin(np.radians(df["DEC"]))
            feature_df["dec_cos"] = np.cos(np.radians(df["DEC"]))
            feature_df["dec_normalized"] = (df["DEC"] + 90) / 180.0
        
        # 3D position features (if available)
        for coord in ["X", "Y", "Z"]:
            if coord in df.columns:
                feature_df[f"{coord}_normalized"] = df[coord] / (df[coord].abs().max() + 1e-8)
                feature_df[f"log_{coord}"] = np.sign(df[coord]) * np.log10(np.abs(df[coord]) + 1)
        
        # Distance from origin
        if all(col in df.columns for col in ["X", "Y", "Z"]):
            feature_df["distance_3d"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
            feature_df["log_distance_3d"] = np.log10(feature_df["distance_3d"] + 1)
        
        # Magnitude-related features
        magnitude_cols = [col for col in df.columns if "magnitude" in col.lower()]
        if magnitude_cols:
            # Mean magnitude
            feature_df["magnitude_mean"] = df[magnitude_cols].mean(axis=1)
            
            # Magnitude standard deviation (color diversity)
            feature_df["magnitude_std"] = df[magnitude_cols].std(axis=1)
            
            # Brightest and faintest magnitudes
            feature_df["magnitude_min"] = df[magnitude_cols].min(axis=1)
            feature_df["magnitude_max"] = df[magnitude_cols].max(axis=1)
            feature_df["magnitude_range"] = feature_df["magnitude_max"] - feature_df["magnitude_min"]
        
        # Source encoding
        if "source" in df.columns:
            source_dummies = pd.get_dummies(df["source"], prefix="source")
            feature_df = pd.concat([feature_df, source_dummies], axis=1)
        
        # Galactic coordinates (simplified)
        if "RA" in df.columns and "DEC" in df.columns:
            # Convert to galactic-like coordinates (simplified)
            feature_df["galactic_l"] = np.arctan2(df["RA"] - 180, df["DEC"]) * 180 / np.pi
            feature_df["galactic_b"] = np.arcsin(np.sin(np.radians(df["DEC"]))) * 180 / np.pi
        
        logger.info(f"Feature engineering complete. Features: {len(feature_df.columns)}")
        return feature_df
    
    def select_features(self, df: pd.DataFrame, target_column: str, 
                       feature_selection_method: str = "correlation", 
                       n_features: int = 20) -> List[str]:
        """Select the most relevant features for the target variable."""
        logger.info(f"Selecting features for target: {target_column}")
        
        # Exclude non-numeric and target columns
        exclude_cols = [target_column, "source"] + [col for col in df.columns if df[col].dtype == 'object']
        feature_candidates = [col for col in df.columns if col not in exclude_cols]
        
        # Remove columns with too many NaN values
        valid_features = []
        for col in feature_candidates:
            if df[col].notna().sum() / len(df) > 0.5:  # At least 50% non-null
                valid_features.append(col)
        
        logger.info(f"Valid feature candidates: {len(valid_features)}")
        
        if feature_selection_method == "correlation":
            # Correlation-based selection
            correlations = df[valid_features + [target_column]].corr()[target_column].abs()
            selected_features = correlations.nlargest(n_features + 1).index.tolist()
            selected_features.remove(target_column)
        
        elif feature_selection_method == "mutual_info":
            # Mutual information-based selection
            X = df[valid_features].fillna(0)
            y = df[target_column].fillna(0)
            
            selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(valid_features)))
            selector.fit(X, y)
            
            selected_features = [valid_features[i] for i in selector.get_support(indices=True)]
        
        elif feature_selection_method == "variance":
            # Select features with highest variance (most information)
            variances = df[valid_features].var().sort_values(ascending=False)
            selected_features = variances.head(n_features).index.tolist()
        
        else:
            # Default: use all valid features (limited to n_features)
            selected_features = valid_features[:n_features]
        
        logger.info(f"Selected {len(selected_features)} features: {selected_features[:5]}...")
        return selected_features
    
    def prepare_dataset(self, config: Dict) -> Dict[str, Any]:
        """Prepare the complete ML dataset."""
        logger.info("Preparing ML dataset...")
        
        # Load and engineer features
        df = self.load_data()
        df = self.engineer_features(df)
        
        # Set target variable
        target_variable = config.get("target_variable", "redshift")
        self.target_column = target_variable
        
        if target_variable not in df.columns:
            raise ValueError(f"Target variable '{target_variable}' not found in data")
        
        # Feature selection
        include_features = config.get("include_features")
        if include_features:
            selected_features = [f for f in include_features if f in df.columns]
        else:
            selected_features = self.select_features(
                df, target_variable, 
                feature_selection_method="correlation",
                n_features=config.get("n_features", 20)
            )
        
        self.feature_columns = selected_features
        
        # Prepare feature matrix and target vector
        X = df[selected_features].copy()
        y = df[target_variable].copy()
        
        # Handle missing values
        logger.info("Handling missing values...")
        imputer_method = config.get("imputation", "median")
        
        if imputer_method == "knn":
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=imputer_method)
        
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Remove samples with missing target values
        valid_indices = y.notna()
        X_final = X_imputed[valid_indices]
        y_final = y[valid_indices]
        
        logger.info(f"Final dataset: {len(X_final)} samples, {len(X_final.columns)} features")
        
        # Train/test split
        test_size = config.get("test_size", 0.2)
        random_state = config.get("random_state", 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_final, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self._create_stratification_labels(y_final) if len(y_final) > 100 else None
        )
        
        # Normalization
        normalization = config.get("normalization", "standard")
        if normalization in SCALERS:
            self.scaler = SCALERS[normalization]()
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Prepare metadata
        metadata = {
            "target_variable": target_variable,
            "n_samples": len(X_final),
            "n_features": len(selected_features),
            "feature_names": selected_features,
            "test_size": test_size,
            "normalization": normalization,
            "imputation": imputer_method,
            "target_statistics": {
                "min": float(y_final.min()),
                "max": float(y_final.max()),
                "mean": float(y_final.mean()),
                "std": float(y_final.std())
            },
            "feature_statistics": {
                col: {
                    "mean": float(X_final[col].mean()),
                    "std": float(X_final[col].std()),
                    "min": float(X_final[col].min()),
                    "max": float(X_final[col].max())
                } for col in selected_features
            },
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "metadata": metadata,
            "scaler": self.scaler,
            "imputer": self.imputer
        }
    
    def _create_stratification_labels(self, y: pd.Series) -> pd.Series:
        """Create stratification labels for train/test split."""
        # Create bins for continuous target variables
        if y.dtype in ['float64', 'float32']:
            n_bins = min(10, len(y.unique()))
            return pd.cut(y, bins=n_bins, labels=False, duplicates='drop')
        else:
            return y
    
    def save_dataset(self, dataset: Dict[str, Any], task_id: str) -> str:
        """Save the prepared dataset to files."""
        logger.info(f"Saving ML dataset for task {task_id}...")
        
        # Create task directory
        task_dir = os.path.join(ML_DIR, f"dataset_{task_id}")
        os.makedirs(task_dir, exist_ok=True)
        
        # Save data files
        dataset["X_train"].to_csv(os.path.join(task_dir, "X_train.csv"), index=False)
        dataset["X_test"].to_csv(os.path.join(task_dir, "X_test.csv"), index=False)
        dataset["y_train"].to_csv(os.path.join(task_dir, "y_train.csv"), index=False)
        dataset["y_test"].to_csv(os.path.join(task_dir, "y_test.csv"), index=False)
        
        # Save preprocessors
        if dataset["scaler"]:
            joblib.dump(dataset["scaler"], os.path.join(task_dir, "scaler.pkl"))
        
        if dataset["imputer"]:
            joblib.dump(dataset["imputer"], os.path.join(task_dir, "imputer.pkl"))
        
        # Save metadata
        import json
        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(dataset["metadata"], f, indent=2)
        
        # Create README
        readme_content = self._generate_readme(dataset["metadata"])
        with open(os.path.join(task_dir, "README.md"), "w") as f:
            f.write(readme_content)
        
        # Create ZIP file
        zip_path = os.path.join(ML_DIR, f"ml_dataset_{task_id}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(task_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, task_dir)
                    zipf.write(file_path, arcname)
        
        logger.info(f"Dataset saved to {zip_path}")
        return zip_path
    
    def _generate_readme(self, metadata: Dict) -> str:
        """Generate README file for the dataset."""
        readme = f"""# Astronomical ML Dataset

## Dataset Information

- **Target Variable**: {metadata['target_variable']}
- **Number of Samples**: {metadata['n_samples']:,}
- **Number of Features**: {metadata['n_features']}
- **Test Set Size**: {metadata['test_size']:.1%}
- **Created**: {metadata['created_at']}

## Preprocessing

- **Normalization**: {metadata['normalization']}
- **Imputation**: {metadata['imputation']}

## Target Variable Statistics

- **Range**: {metadata['target_statistics']['min']:.3f} - {metadata['target_statistics']['max']:.3f}
- **Mean**: {metadata['target_statistics']['mean']:.3f}
- **Standard Deviation**: {metadata['target_statistics']['std']:.3f}

## Features

{chr(10).join(f"- {name}" for name in metadata['feature_names'])}

## Files

- `X_train.csv`: Training features
- `X_test.csv`: Test features  
- `y_train.csv`: Training targets
- `y_test.csv`: Test targets
- `scaler.pkl`: Fitted feature scaler (if used)
- `imputer.pkl`: Fitted imputer (if used)
- `metadata.json`: Complete dataset metadata

## Usage Example

```python
import pandas as pd
import joblib

# Load data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Load preprocessors
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')

# Train your model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train.values.ravel())

# Evaluate
score = model.score(X_test, y_test)
print(f"Model R² score: {{score:.3f}}")
```

## Data Sources

This dataset combines data from multiple astronomical surveys:
- SDSS (Sloan Digital Sky Survey) DR17
- DESI (Dark Energy Spectroscopic Instrument) DR1
- DES (Dark Energy Survey) Y6
- Euclid Q1 (if available)

## Citation

If you use this dataset in your research, please cite the original survey data sources.
"""
        return readme

# Main function for the background task
def prepare_ml_dataset(config: Dict) -> Dict[str, Any]:
    """Main function to prepare ML dataset (for background tasks)."""
    try:
        processor = AstronomicalMLPreprocessor()
        
        # Prepare dataset
        dataset = processor.prepare_dataset(config)
        
        # Generate task ID if not provided
        task_id = config.get("task_id", f"auto_{int(datetime.now().timestamp())}")
        
        # Save dataset
        zip_path = processor.save_dataset(dataset, task_id)
        
        return {
            "status": "success",
            "file_path": zip_path,
            "metadata": dataset["metadata"],
            "message": f"ML dataset prepared successfully with {dataset['metadata']['n_samples']} samples"
        }
        
    except Exception as e:
        logger.error(f"Error preparing ML dataset: {e}")
        return {
            "status": "error",
            "error": str(e)
        } 