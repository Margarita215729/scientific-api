"""
Data processing module for custom astronomical data cleaning and normalization.
Handles various input formats and provides comprehensive data processing pipelines.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PROCESSING_OUTPUT_DIR = "galaxy_data/processed"
SUPPORTED_FORMATS = ["csv", "fits", "txt", "json", "parquet"]

class DataProcessor:
    """Main class for astronomical data processing."""
    
    def __init__(self):
        os.makedirs(PROCESSING_OUTPUT_DIR, exist_ok=True)
        
    def process_astronomical_data(self, config: Dict) -> Dict[str, Any]:
        """Main processing function."""
        processing_type = config.get("processing_type", "clean")
        input_format = config.get("input_format", "csv")
        output_format = config.get("output_format", "csv")
        
        logger.info(f"Processing data: {processing_type}, {input_format} -> {output_format}")
        
        if processing_type == "clean":
            return self._clean_data(config)
        elif processing_type == "normalize":
            return self._normalize_data(config)
        elif processing_type == "feature_engineering":
            return self._engineer_features(config)
        else:
            raise ValueError(f"Unknown processing type: {processing_type}")
    
    def _clean_data(self, config: Dict) -> Dict[str, Any]:
        """Clean astronomical data."""
        logger.info("Cleaning astronomical data...")
        
        # Load existing data
        df = self._load_available_data()
        
        if df.empty:
            raise Exception("No data available for processing")
        
        original_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates()
        logger.info(f"Removed {original_count - len(df)} duplicate rows")
        
        # Clean coordinate columns
        if "RA" in df.columns:
            # Remove invalid RA values
            df = df[(df["RA"] >= 0) & (df["RA"] <= 360)]
            # Remove extreme outliers
            ra_q1, ra_q3 = df["RA"].quantile([0.01, 0.99])
            df = df[(df["RA"] >= ra_q1) & (df["RA"] <= ra_q3)]
        
        if "DEC" in df.columns:
            # Remove invalid DEC values
            df = df[(df["DEC"] >= -90) & (df["DEC"] <= 90)]
            # Remove extreme outliers
            dec_q1, dec_q3 = df["DEC"].quantile([0.01, 0.99])
            df = df[(df["DEC"] >= dec_q1) & (df["DEC"] <= dec_q3)]
        
        # Clean redshift values
        if "redshift" in df.columns:
            # Remove negative or extremely large redshifts
            df = df[(df["redshift"] >= 0) & (df["redshift"] <= 10)]
            # Remove extreme outliers (>99.5 percentile)
            z_max = df["redshift"].quantile(0.995)
            df = df[df["redshift"] <= z_max]
        
        # Clean magnitude columns
        magnitude_cols = [col for col in df.columns if "magnitude" in col.lower()]
        for mag_col in magnitude_cols:
            # Remove impossible magnitude values
            df = df[(df[mag_col] >= 5) & (df[mag_col] <= 35)]
            
            # Remove extreme outliers
            mag_q1, mag_q3 = df[mag_col].quantile([0.005, 0.995])
            df = df[(df[mag_col] >= mag_q1) & (df[mag_col] <= mag_q3)]
        
        # Remove rows with too many missing values
        missing_threshold = 0.5  # Remove rows with >50% missing values
        df = df.dropna(thresh=int(len(df.columns) * missing_threshold))
        
        # Clean 3D coordinates if present
        coord_3d = ["X", "Y", "Z"]
        for coord in coord_3d:
            if coord in df.columns:
                # Remove infinite values
                df = df[np.isfinite(df[coord])]
                
                # Remove extreme outliers
                coord_q1, coord_q3 = df[coord].quantile([0.005, 0.995])
                df = df[(df[coord] >= coord_q1) & (df[coord] <= coord_q3)]
        
        # Save cleaned data
        output_path = os.path.join(PROCESSING_OUTPUT_DIR, "cleaned_data.csv")
        df.to_csv(output_path, index=False)
        
        cleaning_report = {
            "original_rows": original_count,
            "cleaned_rows": len(df),
            "removed_rows": original_count - len(df),
            "removal_percentage": ((original_count - len(df)) / original_count) * 100,
            "columns_processed": list(df.columns),
            "output_file": output_path
        }
        
        logger.info(f"Data cleaning complete: {len(df)} rows remaining ({cleaning_report['removal_percentage']:.1f}% removed)")
        
        return {
            "status": "success",
            "processing_type": "clean",
            "report": cleaning_report,
            "output_path": output_path
        }
    
    def _normalize_data(self, config: Dict) -> Dict[str, Any]:
        """Normalize astronomical data."""
        logger.info("Normalizing astronomical data...")
        
        # Load data (prefer cleaned data if available)
        cleaned_path = os.path.join(PROCESSING_OUTPUT_DIR, "cleaned_data.csv")
        if os.path.exists(cleaned_path):
            df = pd.read_csv(cleaned_path)
        else:
            df = self._load_available_data()
        
        if df.empty:
            raise Exception("No data available for normalization")
        
        original_df = df.copy()
        
        # Normalize coordinates
        if "RA" in df.columns:
            df["RA_normalized"] = df["RA"] / 360.0
        
        if "DEC" in df.columns:
            df["DEC_normalized"] = (df["DEC"] + 90) / 180.0
        
        # Normalize redshift
        if "redshift" in df.columns:
            # Log normalization for redshift (better for ML)
            df["redshift_log"] = np.log10(df["redshift"] + 0.001)
            df["redshift_normalized"] = (df["redshift"] - df["redshift"].min()) / (df["redshift"].max() - df["redshift"].min())
        
        # Normalize magnitudes
        magnitude_cols = [col for col in df.columns if "magnitude" in col.lower()]
        for mag_col in magnitude_cols:
            if mag_col in df.columns:
                # Z-score normalization
                df[f"{mag_col}_zscore"] = (df[mag_col] - df[mag_col].mean()) / df[mag_col].std()
                
                # Min-max normalization
                df[f"{mag_col}_minmax"] = (df[mag_col] - df[mag_col].min()) / (df[mag_col].max() - df[mag_col].min())
        
        # Normalize 3D coordinates
        coord_3d = ["X", "Y", "Z"]
        for coord in coord_3d:
            if coord in df.columns:
                # Center and scale
                df[f"{coord}_centered"] = df[coord] - df[coord].mean()
                df[f"{coord}_normalized"] = df[f"{coord}_centered"] / df[f"{coord}_centered"].abs().max()
                
                # Distance from origin
                if all(c in df.columns for c in coord_3d):
                    df["distance_3d"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
                    df["distance_3d_log"] = np.log10(df["distance_3d"] + 1)
        
        # Add angular separations and features
        if "RA" in df.columns and "DEC" in df.columns:
            # Convert to radians for calculations
            ra_rad = np.radians(df["RA"])
            dec_rad = np.radians(df["DEC"])
            
            # Trigonometric features
            df["RA_sin"] = np.sin(ra_rad)
            df["RA_cos"] = np.cos(ra_rad)
            df["DEC_sin"] = np.sin(dec_rad)
            df["DEC_cos"] = np.cos(dec_rad)
        
        # Save normalized data
        output_path = os.path.join(PROCESSING_OUTPUT_DIR, "normalized_data.csv")
        df.to_csv(output_path, index=False)
        
        normalization_report = {
            "original_columns": len(original_df.columns),
            "normalized_columns": len(df.columns),
            "added_features": len(df.columns) - len(original_df.columns),
            "normalization_methods": ["z-score", "min-max", "log", "trigonometric"],
            "output_file": output_path
        }
        
        logger.info(f"Data normalization complete: {len(df.columns)} total columns")
        
        return {
            "status": "success",
            "processing_type": "normalize",
            "report": normalization_report,
            "output_path": output_path
        }
    
    def _engineer_features(self, config: Dict) -> Dict[str, Any]:
        """Engineer advanced features from astronomical data."""
        logger.info("Engineering advanced features...")
        
        # Load data (prefer normalized data if available)
        normalized_path = os.path.join(PROCESSING_OUTPUT_DIR, "normalized_data.csv")
        cleaned_path = os.path.join(PROCESSING_OUTPUT_DIR, "cleaned_data.csv")
        
        if os.path.exists(normalized_path):
            df = pd.read_csv(normalized_path)
        elif os.path.exists(cleaned_path):
            df = pd.read_csv(cleaned_path)
        else:
            df = self._load_available_data()
        
        if df.empty:
            raise Exception("No data available for feature engineering")
        
        original_columns = len(df.columns)
        
        # Color indices (magnitude differences)
        magnitude_cols = [col for col in df.columns if "magnitude" in col.lower() and "_" not in col]
        if len(magnitude_cols) >= 2:
            for i, mag1 in enumerate(magnitude_cols):
                for mag2 in magnitude_cols[i+1:]:
                    color_name = f"color_{mag1.split('_')[-1]}_{mag2.split('_')[-1]}"
                    df[color_name] = df[mag1] - df[mag2]
        
        # Spectral features
        if "redshift" in df.columns:
            # Redshift bins
            df["redshift_bin_coarse"] = pd.cut(df["redshift"], 
                                             bins=[0, 0.1, 0.5, 1.0, 2.0, 10.0],
                                             labels=["nearby", "intermediate", "distant", "high_z", "extreme"])
            
            df["redshift_bin_fine"] = pd.cut(df["redshift"], bins=20, labels=False)
            
            # Velocity features
            c = 299792.458  # Speed of light in km/s
            df["velocity_recession"] = df["redshift"] * c  # Simplified recession velocity
        
        # Morphological features
        if all(col in df.columns for col in ["RA", "DEC"]):
            # Sky density features (local density estimation)
            df["galactic_latitude"] = 90 - np.abs(df["DEC"])  # Distance from galactic plane (simplified)
            
            # Clustering features (simplified)
            # Group by sky regions
            df["sky_region_ra"] = pd.cut(df["RA"], bins=36, labels=False)  # 10-degree bins
            df["sky_region_dec"] = pd.cut(df["DEC"], bins=18, labels=False)  # 10-degree bins
            df["sky_region"] = df["sky_region_ra"] * 18 + df["sky_region_dec"]
        
        # Distance-related features
        if "distance_3d" in df.columns or all(col in df.columns for col in ["X", "Y", "Z"]):
            if "distance_3d" not in df.columns:
                df["distance_3d"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
            
            # Volume density features
            df["volume_shell"] = pd.cut(df["distance_3d"], bins=20, labels=False)
            
            # Angular size features (if we have physical size estimates)
            if "redshift" in df.columns:
                # Simplified angular diameter distance
                df["angular_diameter_distance"] = df["distance_3d"] / (1 + df["redshift"])
        
        # Statistical features per source
        if "source" in df.columns:
            # Group statistics
            for col in ["redshift", "RA", "DEC"] + magnitude_cols:
                if col in df.columns:
                    source_stats = df.groupby("source")[col].agg(["mean", "std", "count"])
                    source_stats.columns = [f"{col}_source_{stat}" for stat in ["mean", "std", "count"]]
                    df = df.merge(source_stats, left_on="source", right_index=True, how="left")
        
        # Interaction features
        if "redshift" in df.columns and magnitude_cols:
            # Magnitude vs redshift interactions
            for mag_col in magnitude_cols[:3]:  # Limit to avoid too many features
                if mag_col in df.columns:
                    df[f"{mag_col}_per_redshift"] = df[mag_col] / (df["redshift"] + 0.001)
                    df[f"{mag_col}_redshift_product"] = df[mag_col] * df["redshift"]
        
        # Environmental features
        if all(col in df.columns for col in ["X", "Y", "Z"]):
            # Simplified local density (count neighbors in radius)
            # This is computationally expensive, so we'll do a simplified version
            
            # Grid-based density
            x_bins = pd.cut(df["X"], bins=20, labels=False)
            y_bins = pd.cut(df["Y"], bins=20, labels=False)
            z_bins = pd.cut(df["Z"], bins=20, labels=False)
            
            df["grid_cell"] = x_bins * 400 + y_bins * 20 + z_bins
            grid_counts = df["grid_cell"].value_counts()
            df["local_density_grid"] = df["grid_cell"].map(grid_counts)
        
        # Remove temporary columns
        temp_columns = [col for col in df.columns if "bin" in col and col.endswith("_fine")]
        df = df.drop(columns=temp_columns, errors="ignore")
        
        # Save feature-engineered data
        output_path = os.path.join(PROCESSING_OUTPUT_DIR, "feature_engineered_data.csv")
        df.to_csv(output_path, index=False)
        
        feature_report = {
            "original_columns": original_columns,
            "final_columns": len(df.columns),
            "engineered_features": len(df.columns) - original_columns,
            "feature_types": {
                "color_indices": len([col for col in df.columns if col.startswith("color_")]),
                "redshift_features": len([col for col in df.columns if "redshift" in col]),
                "distance_features": len([col for col in df.columns if "distance" in col or "volume" in col]),
                "angular_features": len([col for col in df.columns if "sin" in col or "cos" in col]),
                "statistical_features": len([col for col in df.columns if "_source_" in col]),
                "interaction_features": len([col for col in df.columns if "_per_" in col or "_product" in col])
            },
            "output_file": output_path
        }
        
        logger.info(f"Feature engineering complete: {feature_report['engineered_features']} new features")
        
        return {
            "status": "success",
            "processing_type": "feature_engineering",
            "report": feature_report,
            "output_path": output_path
        }
    
    def _load_available_data(self) -> pd.DataFrame:
        """Load any available astronomical data."""
        # Try to load in order of preference
        file_patterns = [
            "merged_real_galaxies.csv",
            "sdss_real.csv",
            "desi_real.csv", 
            "des_real.csv",
            "euclid_real.csv",
            "merged_galaxies.csv"  # Fallback to demo data
        ]
        
        for pattern in file_patterns:
            file_path = os.path.join(PROCESSING_OUTPUT_DIR, pattern)
            if os.path.exists(file_path):
                logger.info(f"Loading data from {file_path}")
                return pd.read_csv(file_path)
        
        logger.warning("No astronomical data files found")
        return pd.DataFrame()

# Main function for background tasks
def process_astronomical_data(config: Dict) -> Dict[str, Any]:
    """Main function for processing astronomical data (background task)."""
    try:
        processor = DataProcessor()
        result = processor.process_astronomical_data(config)
        
        return {
            "status": "success",
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error processing astronomical data: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 