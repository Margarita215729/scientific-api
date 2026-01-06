"""
Binary Classification Models for Graph Analysis.

This module provides functions to train and evaluate classifiers that
distinguish cosmological graphs from quantum graphs based on extracted features.

Key Functions:
--------------
- train_classifiers: Train multiple classifier models
- evaluate_classifiers: Evaluate classifier performance with cross-validation
- predict_graph_type: Predict graph type from features
- save_classifier / load_classifier: Model persistence

Dependencies:
-------------
- scikit-learn for classification algorithms
- pandas for feature tables
- joblib for model serialization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def prepare_classification_data(
    feature_table: pd.DataFrame,
    target_column: str = "system_type",
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    scale_features: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Prepare feature table for classification.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table with graph features and labels.
    target_column : str, default="system_type"
        Column name containing labels ("cosmology" or "quantum").
    feature_columns : list of str, optional
        List of feature column names. If None, uses all numeric columns except target.
    test_size : float, default=0.2
        Fraction of data to use for testing.
    random_state : int, default=42
        Random seed for reproducibility.
    scale_features : bool, default=True
        Whether to standardize features (mean=0, std=1).

    Returns
    -------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Testing features.
    y_train : np.ndarray
        Training labels (0 for cosmology, 1 for quantum).
    y_test : np.ndarray
        Testing labels.
    scaler : StandardScaler or None
        Fitted scaler if scale_features=True, else None.

    Notes
    -----
    - Automatically converts string labels to binary (0/1)
    - Removes NaN/inf values
    - Logs data statistics
    """
    logger.info("Preparing classification data...")

    # Select features
    if feature_columns is None:
        # Use all numeric columns except target and metadata
        exclude_cols = {target_column, "graph_id", "system_type"}
        feature_columns = [
            col for col in feature_table.columns
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

    logger.info(f"Using {len(feature_columns)} features for classification")

    # Extract features and labels
    X = feature_table[feature_columns].values
    y_raw = feature_table[target_column].values

    # Convert labels to binary
    # Assuming "cosmology" -> 0, "quantum" -> 1
    y = np.array([1 if label == "quantum" else 0 for label in y_raw])

    # Handle missing/invalid values
    valid_mask = np.all(np.isfinite(X), axis=1)
    if not np.all(valid_mask):
        n_invalid = np.sum(~valid_mask)
        logger.warning(f"Removing {n_invalid} rows with NaN/inf values")
        X = X[valid_mask]
        y = y[valid_mask]

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: cosmology={np.sum(y == 0)}, quantum={np.sum(y == 1)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature scaling
    scaler = None
    if scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        logger.info("Features standardized (mean=0, std=1)")

    return X_train, X_test, y_train, y_test, scaler


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train multiple classification models.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    models : dict, optional
        Dictionary of model names to sklearn estimators.
        If None, uses default models (LogisticRegression, RandomForest, GradientBoosting).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    trained_models : dict
        Dictionary of model names to fitted estimators.

    Notes
    -----
    Default models:
    - Logistic Regression: Simple linear baseline
    - Random Forest: Ensemble decision trees
    - Gradient Boosting: Boosted decision trees (usually best performance)
    """
    logger.info("Training classifiers...")

    if models is None:
        # Default models
        models = {
            "logistic_regression": LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver="lbfgs",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
            ),
        }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} training complete")

    logger.info(f"Trained {len(trained_models)} models")
    return trained_models


def evaluate_classifiers(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Evaluate classifier performance.

    Parameters
    ----------
    models : dict
        Dictionary of model names to fitted estimators.
    X_test : np.ndarray
        Testing features.
    y_test : np.ndarray
        Testing labels.
    X_train : np.ndarray, optional
        Training features for cross-validation.
    y_train : np.ndarray, optional
        Training labels for cross-validation.
    cv_folds : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    results_df : pd.DataFrame
        Evaluation metrics for each model.
        Columns: accuracy, precision, recall, f1_score, roc_auc, cv_score_mean, cv_score_std

    Notes
    -----
    - Reports test set performance for all models
    - Optionally reports cross-validation scores on training set
    - Logs detailed classification reports
    """
    logger.info("Evaluating classifiers...")

    results = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"{name} - Confusion Matrix:\n{cm}")

        # Classification report
        report = classification_report(y_test, y_pred, target_names=["cosmology", "quantum"])
        logger.info(f"{name} - Classification Report:\n{report}")

        # Cross-validation
        cv_score_mean = np.nan
        cv_score_std = np.nan
        if X_train is not None and y_train is not None:
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv_folds, scoring="accuracy", n_jobs=-1
            )
            cv_score_mean = np.mean(cv_results["test_score"])
            cv_score_std = np.std(cv_results["test_score"])
            logger.info(f"{name} - CV Accuracy: {cv_score_mean:.4f} ± {cv_score_std:.4f}")

        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "cv_score_mean": cv_score_mean,
            "cv_score_std": cv_score_std,
        })

    results_df = pd.DataFrame(results)
    logger.info(f"Evaluation complete. Results:\n{results_df}")

    return results_df


def predict_graph_type(
    model: Any,
    features: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    return_proba: bool = False,
) -> np.ndarray:
    """
    Predict graph type from features.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier.
    features : np.ndarray
        Feature array (n_samples, n_features).
    scaler : StandardScaler, optional
        Fitted scaler for feature normalization.
    return_proba : bool, default=False
        If True, return probability estimates instead of class labels.

    Returns
    -------
    predictions : np.ndarray
        Class labels (0 for cosmology, 1 for quantum) or probabilities.

    Notes
    -----
    - If scaler is provided, features are scaled before prediction
    - Probability output requires model with predict_proba method
    """
    # Scale features if scaler provided
    if scaler is not None:
        features = scaler.transform(features)

    # Predict
    if return_proba:
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features)[:, 1]
        else:
            logger.warning("Model does not support probability estimates, returning class labels")
            return model.predict(features)
    else:
        return model.predict(features)


def save_classifier(
    model: Any,
    scaler: Optional[StandardScaler],
    feature_columns: List[str],
    output_path: Path,
) -> None:
    """
    Save trained classifier and metadata.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier.
    scaler : StandardScaler or None
        Fitted scaler.
    feature_columns : list of str
        List of feature column names.
    output_path : Path
        Output file path (.joblib).

    Notes
    -----
    - Saves model, scaler, and feature columns in single file
    - Use .joblib extension for efficiency
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classifier_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": feature_columns,
    }

    joblib.dump(classifier_data, output_path)
    logger.info(f"Classifier saved to {output_path}")


def load_classifier(input_path: Path) -> Tuple[Any, Optional[StandardScaler], List[str]]:
    """
    Load trained classifier and metadata.

    Parameters
    ----------
    input_path : Path
        Input file path (.joblib).

    Returns
    -------
    model : sklearn estimator
        Trained classifier.
    scaler : StandardScaler or None
        Fitted scaler.
    feature_columns : list of str
        List of feature column names.
    """
    input_path = Path(input_path)
    classifier_data = joblib.load(input_path)

    model = classifier_data["model"]
    scaler = classifier_data.get("scaler")
    feature_columns = classifier_data["feature_columns"]

    logger.info(f"Classifier loaded from {input_path}")
    return model, scaler, feature_columns


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging
    from ml.features.feature_table import load_feature_table

    setup_logging()

    # Example: Load feature table and train classifiers
    # Assuming feature table was generated by build_feature_table_from_directory()
    feature_table_path = Path("data/processed/features/feature_table.parquet")

    if feature_table_path.exists():
        logger.info(f"Loading feature table from {feature_table_path}...")
        feature_table = load_feature_table(feature_table_path)

        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_classification_data(
            feature_table, test_size=0.2, random_state=42
        )

        # Train models
        trained_models = train_classifiers(X_train, y_train)

        # Evaluate models
        results = evaluate_classifiers(
            trained_models, X_test, y_test, X_train, y_train, cv_folds=5
        )

        print("\n=== Classification Results ===")
        print(results)

        # Save best model (e.g., gradient_boosting)
        best_model_name = results.loc[results["f1_score"].idxmax(), "model"]
        best_model = trained_models[best_model_name]

        feature_columns = [
            col for col in feature_table.columns
            if col not in {"system_type", "graph_id"} and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

        output_dir = Path("data/models")
        save_classifier(
            best_model,
            scaler,
            feature_columns,
            output_dir / f"classifier_{best_model_name}.joblib"
        )

        logger.info(f"Best model: {best_model_name} (F1={results.loc[results['f1_score'].idxmax(), 'f1_score']:.4f})")
    else:
        logger.error(f"Feature table not found at {feature_table_path}")
        logger.info("Run build_feature_table_from_directory() first to generate feature table")
