"""
Advanced Data Cleaning and Transformation API
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime
import logging
from enum import Enum

# Import database
from database.config import db

# Import data processing utilities
try:
    from utils.data_processing import clean_astronomical_data, normalize_features
    PROCESSING_AVAILABLE = True
except ImportError:
    PROCESSING_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cleaning", tags=["data_cleaning"])

class CleaningOperation(str, Enum):
    REMOVE_DUPLICATES = "remove_duplicates"
    HANDLE_MISSING = "handle_missing"
    NORMALIZE_VALUES = "normalize_values"
    REMOVE_OUTLIERS = "remove_outliers"
    STANDARDIZE_FORMATS = "standardize_formats"
    VALIDATE_RANGES = "validate_ranges"

class MissingValueStrategy(str, Enum):
    DROP = "drop"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_ZERO = "fill_zero"
    INTERPOLATE = "interpolate"

@router.post("/analyze-issues")
async def analyze_dataset_issues(
    dataset_id: str,
    sample_size: int = Query(10000, ge=100, le=100000)
):
    """Analyze dataset for quality issues and suggest cleaning operations"""
    if db.mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        collection = db.mongo_db[dataset_id]
        
        # Get sample of records
        records = await collection.find().limit(sample_size).to_list(sample_size)
        
        if not records:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(records)
        
        # Remove MongoDB metadata columns for analysis
        analysis_df = df.drop(columns=[col for col in df.columns if col.startswith('_')], errors='ignore')
        
        issues = []
        recommendations = []
        
        # 1. Missing Values Analysis
        missing_analysis = analyze_missing_values(analysis_df)
        if missing_analysis["total_missing"] > 0:
            issues.append(missing_analysis)
            recommendations.extend(missing_analysis["recommendations"])
        
        # 2. Duplicate Analysis
        duplicate_analysis = analyze_duplicates(analysis_df)
        if duplicate_analysis["duplicate_count"] > 0:
            issues.append(duplicate_analysis)
            recommendations.extend(duplicate_analysis["recommendations"])
        
        # 3. Outlier Analysis
        outlier_analysis = analyze_outliers(analysis_df)
        if outlier_analysis["outlier_count"] > 0:
            issues.append(outlier_analysis)
            recommendations.extend(outlier_analysis["recommendations"])
        
        # 4. Data Type Analysis
        dtype_analysis = analyze_data_types(analysis_df)
        if dtype_analysis["inconsistent_types"]:
            issues.append(dtype_analysis)
            recommendations.extend(dtype_analysis["recommendations"])
        
        # 5. Value Range Analysis
        range_analysis = analyze_value_ranges(analysis_df)
        if range_analysis["out_of_range_count"] > 0:
            issues.append(range_analysis)
            recommendations.extend(range_analysis["recommendations"])
        
        # Calculate overall health score
        health_score = calculate_dataset_health_score(issues, len(analysis_df))
        
        return {
            "dataset_id": dataset_id,
            "total_records": len(df),
            "analyzed_records": len(analysis_df),
            "total_columns": len(analysis_df.columns),
            "health_score": health_score,
            "issues": issues,
            "recommendations": recommendations,
            "suggested_operations": get_suggested_operations(issues)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing dataset {dataset_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def analyze_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze missing values in the dataset"""
    missing_counts = df.isnull().sum()
    total_cells = len(df) * len(df.columns)
    total_missing = missing_counts.sum()
    
    missing_by_column = {}
    recommendations = []
    
    for col, count in missing_counts.items():
        if count > 0:
            percentage = (count / len(df)) * 100
            missing_by_column[col] = {
                "count": int(count),
                "percentage": round(percentage, 2)
            }
            
            # Generate recommendations based on missing percentage
            if percentage > 50:
                recommendations.append(f"Consider dropping column '{col}' (>{percentage:.1f}% missing)")
            elif percentage > 20:
                recommendations.append(f"Column '{col}' has significant missing data - consider imputation or investigation")
            elif percentage > 5:
                recommendations.append(f"Fill missing values in '{col}' using appropriate strategy")
    
    return {
        "type": "missing_values",
        "severity": "high" if (total_missing / total_cells) > 0.1 else "medium" if total_missing > 0 else "low",
        "total_missing": int(total_missing),
        "missing_percentage": round((total_missing / total_cells) * 100, 2),
        "missing_by_column": missing_by_column,
        "recommendations": recommendations
    }

def analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze duplicate rows in the dataset"""
    duplicate_mask = df.duplicated()
    duplicate_count = duplicate_mask.sum()
    
    recommendations = []
    if duplicate_count > 0:
        percentage = (duplicate_count / len(df)) * 100
        recommendations.append(f"Remove {duplicate_count} duplicate rows ({percentage:.1f}% of data)")
        
        # Check for partial duplicates (same values in key columns)
        if len(df.columns) > 3:
            # Check duplicates in first few columns (likely identifiers)
            key_columns = df.columns[:3]
            partial_duplicates = df.duplicated(subset=key_columns).sum()
            if partial_duplicates > duplicate_count:
                recommendations.append(f"Found {partial_duplicates - duplicate_count} partial duplicates in key columns")
    
    return {
        "type": "duplicates",
        "severity": "medium" if duplicate_count > len(df) * 0.05 else "low",
        "duplicate_count": int(duplicate_count),
        "duplicate_percentage": round((duplicate_count / len(df)) * 100, 2),
        "recommendations": recommendations
    }

def analyze_outliers(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze outliers in numerical columns"""
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    outlier_info = {}
    total_outliers = 0
    recommendations = []
    
    for col in numeric_columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
            
        # Use IQR method for outlier detection
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            total_outliers += outlier_count
            percentage = (outlier_count / len(df)) * 100
            
            outlier_info[col] = {
                "count": outlier_count,
                "percentage": round(percentage, 2),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "min_outlier": df[col].min(),
                "max_outlier": df[col].max()
            }
            
            if percentage > 10:
                recommendations.append(f"Column '{col}' has many outliers ({percentage:.1f}%) - investigate data quality")
            elif percentage > 1:
                recommendations.append(f"Consider capping or transforming outliers in '{col}'")
    
    return {
        "type": "outliers",
        "severity": "medium" if total_outliers > len(df) * 0.05 else "low",
        "outlier_count": total_outliers,
        "outliers_by_column": outlier_info,
        "recommendations": recommendations
    }

def analyze_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze data type consistency"""
    type_issues = {}
    recommendations = []
    
    for col in df.columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
        
        # Check if column should be numeric but has mixed types
        try:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            non_numeric_count = numeric_series.isnull().sum() - df[col].isnull().sum()
            
            if non_numeric_count > 0 and non_numeric_count < len(df) * 0.5:
                type_issues[col] = {
                    "issue": "mixed_numeric",
                    "non_numeric_count": int(non_numeric_count),
                    "percentage": round((non_numeric_count / len(df)) * 100, 2)
                }
                recommendations.append(f"Column '{col}' appears numeric but has {non_numeric_count} non-numeric values")
                
        except Exception:
            pass
        
        # Check for inconsistent date formats
        if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
            try:
                # Try to parse as datetime
                pd.to_datetime(df[col], errors='coerce')
                # If successful, check for parsing failures
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                failed_parses = parsed_dates.isnull().sum() - df[col].isnull().sum()
                
                if failed_parses > 0:
                    type_issues[col] = {
                        "issue": "inconsistent_dates",
                        "failed_parses": int(failed_parses),
                        "percentage": round((failed_parses / len(df)) * 100, 2)
                    }
                    recommendations.append(f"Column '{col}' has inconsistent date formats")
                    
            except Exception:
                pass
    
    return {
        "type": "data_types",
        "severity": "medium" if type_issues else "low",
        "inconsistent_types": type_issues,
        "recommendations": recommendations
    }

def analyze_value_ranges(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze if values are within expected ranges for scientific data"""
    range_issues = {}
    recommendations = []
    total_out_of_range = 0
    
    # Define expected ranges for common astronomical columns
    expected_ranges = {
        'ra': (0, 360),          # Right Ascension in degrees
        'dec': (-90, 90),        # Declination in degrees
        'z': (0, 10),            # Redshift (reasonable upper bound)
        'mag': (-30, 30),        # Magnitude values
        'flux': (0, None),       # Flux should be positive
        'mass': (0, None),       # Mass should be positive
        'radius': (0, None),     # Radius should be positive
        'distance': (0, None),   # Distance should be positive
        'temperature': (0, None), # Temperature should be positive
        'luminosity': (0, None)  # Luminosity should be positive
    }
    
    for col in df.columns:
        if col.startswith('_'):  # Skip metadata columns
            continue
        
        # Check if column name matches known patterns
        col_lower = col.lower()
        matching_range = None
        
        for pattern, range_values in expected_ranges.items():
            if pattern in col_lower:
                matching_range = range_values
                break
        
        if matching_range and pd.api.types.is_numeric_dtype(df[col]):
            min_val, max_val = matching_range
            
            out_of_range = 0
            if min_val is not None:
                out_of_range += (df[col] < min_val).sum()
            if max_val is not None:
                out_of_range += (df[col] > max_val).sum()
            
            if out_of_range > 0:
                total_out_of_range += out_of_range
                percentage = (out_of_range / len(df)) * 100
                
                range_issues[col] = {
                    "count": int(out_of_range),
                    "percentage": round(percentage, 2),
                    "expected_range": matching_range,
                    "actual_range": (df[col].min(), df[col].max())
                }
                
                recommendations.append(f"Column '{col}' has {out_of_range} values outside expected range {matching_range}")
    
    return {
        "type": "value_ranges",
        "severity": "high" if total_out_of_range > len(df) * 0.1 else "medium" if total_out_of_range > 0 else "low",
        "out_of_range_count": total_out_of_range,
        "range_issues": range_issues,
        "recommendations": recommendations
    }

def calculate_dataset_health_score(issues: List[Dict], total_records: int) -> float:
    """Calculate overall dataset health score (0-100)"""
    if not issues:
        return 100.0
    
    score = 100.0
    
    for issue in issues:
        severity = issue.get("severity", "low")
        
        if issue["type"] == "missing_values":
            missing_pct = issue.get("missing_percentage", 0)
            if severity == "high":
                score -= min(30, missing_pct)
            else:
                score -= min(15, missing_pct / 2)
                
        elif issue["type"] == "duplicates":
            dup_pct = issue.get("duplicate_percentage", 0)
            score -= min(20, dup_pct)
            
        elif issue["type"] == "outliers":
            if severity == "medium":
                score -= 10
            else:
                score -= 5
                
        elif issue["type"] == "data_types":
            score -= len(issue.get("inconsistent_types", {})) * 5
            
        elif issue["type"] == "value_ranges":
            if severity == "high":
                score -= 20
            else:
                score -= 10
    
    return max(0.0, min(100.0, score))

def get_suggested_operations(issues: List[Dict]) -> List[Dict]:
    """Get suggested cleaning operations based on detected issues"""
    operations = []
    
    for issue in issues:
        if issue["type"] == "missing_values":
            missing_pct = issue.get("missing_percentage", 0)
            if missing_pct > 50:
                operations.append({
                    "operation": CleaningOperation.HANDLE_MISSING,
                    "strategy": MissingValueStrategy.DROP,
                    "priority": "high",
                    "description": "Drop columns with >50% missing values"
                })
            else:
                operations.append({
                    "operation": CleaningOperation.HANDLE_MISSING,
                    "strategy": MissingValueStrategy.FILL_MEDIAN,
                    "priority": "medium",
                    "description": "Fill missing values with median for numeric columns"
                })
        
        elif issue["type"] == "duplicates":
            operations.append({
                "operation": CleaningOperation.REMOVE_DUPLICATES,
                "priority": "high",
                "description": "Remove duplicate rows keeping first occurrence"
            })
        
        elif issue["type"] == "outliers":
            operations.append({
                "operation": CleaningOperation.REMOVE_OUTLIERS,
                "priority": "medium",
                "description": "Remove or cap extreme outliers using IQR method"
            })
        
        elif issue["type"] == "data_types":
            operations.append({
                "operation": CleaningOperation.STANDARDIZE_FORMATS,
                "priority": "medium",
                "description": "Standardize data types and formats"
            })
        
        elif issue["type"] == "value_ranges":
            operations.append({
                "operation": CleaningOperation.VALIDATE_RANGES,
                "priority": "high",
                "description": "Validate and correct values outside expected ranges"
            })
    
    # Always suggest normalization for ML readiness
    operations.append({
        "operation": CleaningOperation.NORMALIZE_VALUES,
        "priority": "low",
        "description": "Normalize numerical features for machine learning"
    })
    
    return operations

@router.post("/apply-cleaning")
async def apply_cleaning_operations(
    dataset_id: str,
    operations: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    create_backup: bool = True
):
    """Apply cleaning operations to a dataset"""
    if db.mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    # Create cleaning task
    task_id = f"clean_{dataset_id}_{datetime.utcnow().timestamp()}"
    
    # Start background cleaning task
    background_tasks.add_task(
        apply_cleaning_background,
        task_id,
        dataset_id,
        operations,
        create_backup
    )
    
    return {
        "status": "started",
        "task_id": task_id,
        "dataset_id": dataset_id,
        "operations": [op.get("operation") for op in operations],
        "message": "Data cleaning started in background"
    }

async def apply_cleaning_background(
    task_id: str,
    dataset_id: str,
    operations: List[Dict[str, Any]],
    create_backup: bool = True
):
    """Background task to apply cleaning operations"""
    try:
        # Update task status
        await db.mongo_update_one(
            "cleaning_tasks",
            {"_id": task_id},
            {
                "status": "processing",
                "started_at": datetime.utcnow(),
                "dataset_id": dataset_id,
                "operations": operations
            },
            upsert=True
        )
        
        # Get dataset
        collection = db.mongo_db[dataset_id]
        records = await collection.find().to_list(None)  # Get all records
        
        if not records:
            raise ValueError("Dataset is empty")
        
        # Create backup if requested
        if create_backup:
            backup_collection = f"{dataset_id}_backup_{int(datetime.utcnow().timestamp())}"
            await db.mongo_db[backup_collection].insert_many(records.copy())
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        original_shape = df.shape
        
        # Apply cleaning operations
        cleaning_log = []
        
        for operation in operations:
            op_type = operation.get("operation")
            
            if op_type == CleaningOperation.REMOVE_DUPLICATES:
                before_count = len(df)
                df = df.drop_duplicates()
                removed = before_count - len(df)
                cleaning_log.append(f"Removed {removed} duplicate rows")
            
            elif op_type == CleaningOperation.HANDLE_MISSING:
                strategy = operation.get("strategy", MissingValueStrategy.FILL_MEDIAN)
                
                if strategy == MissingValueStrategy.DROP:
                    # Drop columns with >50% missing
                    missing_pct = df.isnull().sum() / len(df)
                    cols_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
                    if cols_to_drop:
                        df = df.drop(columns=cols_to_drop)
                        cleaning_log.append(f"Dropped columns with >50% missing: {cols_to_drop}")
                
                elif strategy == MissingValueStrategy.FILL_MEDIAN:
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if df[col].isnull().any():
                            median_val = df[col].median()
                            filled_count = df[col].isnull().sum()
                            df[col] = df[col].fillna(median_val)
                            cleaning_log.append(f"Filled {filled_count} missing values in '{col}' with median ({median_val})")
            
            elif op_type == CleaningOperation.REMOVE_OUTLIERS:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outliers_removed = 0
                
                for col in numeric_cols:
                    if col.startswith('_'):
                        continue
                    
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    before_count = len(df)
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    removed = before_count - len(df)
                    outliers_removed += removed
                
                if outliers_removed > 0:
                    cleaning_log.append(f"Removed {outliers_removed} outlier rows")
            
            elif op_type == CleaningOperation.NORMALIZE_VALUES:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                normalized_cols = []
                
                for col in numeric_cols:
                    if col.startswith('_'):
                        continue
                    
                    min_val = df[col].min()
                    max_val = df[col].max()
                    
                    if max_val != min_val:  # Avoid division by zero
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        normalized_cols.append(col)
                
                if normalized_cols:
                    cleaning_log.append(f"Normalized columns: {normalized_cols}")
        
        # Store cleaned data back to database
        # Clear existing data
        await collection.delete_many({})
        
        # Insert cleaned data
        cleaned_records = df.to_dict('records')
        if cleaned_records:
            await collection.insert_many(cleaned_records)
        
        # Update task status
        await db.mongo_update_one(
            "cleaning_tasks",
            {"_id": task_id},
            {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "original_shape": original_shape,
                "final_shape": df.shape,
                "cleaning_log": cleaning_log
            }
        )
        
    except Exception as e:
        logger.error(f"Cleaning task {task_id} failed: {e}")
        await db.mongo_update_one(
            "cleaning_tasks",
            {"_id": task_id},
            {
                "status": "failed",
                "error": str(e),
                "failed_at": datetime.utcnow()
            }
        )

@router.get("/cleaning-status/{task_id}")
async def get_cleaning_status(task_id: str):
    """Get status of a cleaning task"""
    task = await db.mongo_find_one("cleaning_tasks", {"_id": task_id})
    
    if not task:
        raise HTTPException(status_code=404, detail="Cleaning task not found")
    
    return {
        "task_id": task_id,
        "status": task.get("status"),
        "dataset_id": task.get("dataset_id"),
        "started_at": task.get("started_at"),
        "completed_at": task.get("completed_at"),
        "original_shape": task.get("original_shape"),
        "final_shape": task.get("final_shape"),
        "cleaning_log": task.get("cleaning_log", []),
        "error": task.get("error")
    }

@router.get("/preview-cleaning")
async def preview_cleaning_operations(
    dataset_id: str,
    operations: str = Query(..., description="JSON string of operations to preview")
):
    """Preview the effects of cleaning operations without applying them"""
    if db.mongo_db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        operations_list = json.loads(operations)
        
        # Get sample of dataset
        collection = db.mongo_db[dataset_id]
        records = await collection.find().limit(1000).to_list(1000)
        
        if not records:
            raise HTTPException(status_code=404, detail="Dataset not found or empty")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        original_shape = df.shape
        
        # Apply operations to preview
        preview_log = []
        
        # This is a simplified preview - just show what would happen
        for operation in operations_list:
            op_type = operation.get("operation")
            
            if op_type == CleaningOperation.REMOVE_DUPLICATES:
                duplicates = df.duplicated().sum()
                preview_log.append(f"Would remove {duplicates} duplicate rows")
            
            elif op_type == CleaningOperation.HANDLE_MISSING:
                missing = df.isnull().sum().sum()
                preview_log.append(f"Would handle {missing} missing values")
            
            elif op_type == CleaningOperation.REMOVE_OUTLIERS:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                outlier_count = 0
                for col in numeric_cols:
                    if not col.startswith('_'):
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                        outlier_count += outliers
                preview_log.append(f"Would remove approximately {outlier_count} outlier rows")
        
        return {
            "dataset_id": dataset_id,
            "original_shape": original_shape,
            "preview_operations": preview_log,
            "sample_size": len(df),
            "operations": operations_list
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid operations JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
