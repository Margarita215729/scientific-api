"""
Graph Similarity Regression Models.

This module provides functions to train regression models that predict
pairwise similarity between graphs based on feature differences.

The similarity can be measured using graph distance metrics (e.g., graph edit distance,
spectral distance) and regressed against pairwise feature representations.

Key Functions:
--------------
- build_pairwise_features: Construct pairwise feature representations
- train_similarity_regressors: Train regression models for similarity prediction
- evaluate_similarity_regressors: Evaluate regression performance
- predict_similarity: Predict similarity between graphs
- save_regressor / load_regressor: Model persistence

Dependencies:
-------------
- scikit-learn for regression algorithms
- pandas for feature tables
- joblib for model serialization
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def build_pairwise_features(
    feature_table: pd.DataFrame,
    distance_matrix: Optional[np.ndarray] = None,
    pairwise_method: str = "difference",
    max_pairs: Optional[int] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct pairwise feature representations and similarity targets.

    Parameters
    ----------
    feature_table : pd.DataFrame
        Feature table with graph features.
        Must have columns: graph_id, system_type, and numeric features.
    distance_matrix : np.ndarray, optional
        Precomputed distance matrix (n_graphs, n_graphs).
        If None, uses simple feature-based similarity (1 / (1 + L2 distance)).
    pairwise_method : str, default="difference"
        Method for constructing pairwise features:
        - "difference": Absolute difference |f1 - f2|
        - "concat": Concatenation [f1, f2]
        - "product": Element-wise product f1 * f2
        - "combined": Concatenation of all above
    max_pairs : int, optional
        Maximum number of pairs to generate. If None, uses all pairs.
        Useful for large datasets to avoid quadratic explosion.
    random_state : int, default=42
        Random seed for pair sampling.

    Returns
    -------
    X_pairwise : np.ndarray
        Pairwise feature array (n_pairs, n_pairwise_features).
    y_similarity : np.ndarray
        Similarity targets (n_pairs,).
        Higher values indicate more similar graphs.

    Notes
    -----
    - For n graphs, generates n*(n-1)/2 pairs (upper triangle)
    - Similarity is inverse of distance: sim = 1 / (1 + dist)
    - If distance_matrix not provided, uses Euclidean distance in feature space
    """
    logger.info("Building pairwise features...")

    # Extract numeric features
    feature_cols = [
        col
        for col in feature_table.columns
        if col not in {"graph_id", "system_type"}
        and pd.api.types.is_numeric_dtype(feature_table[col])
    ]
    features = feature_table[feature_cols].values
    n_graphs = len(features)

    logger.info(f"Using {len(feature_cols)} features for {n_graphs} graphs")

    # Generate all pairs
    pairs = []
    for i in range(n_graphs):
        for j in range(i + 1, n_graphs):
            pairs.append((i, j))

    n_pairs_total = len(pairs)
    logger.info(f"Total pairs: {n_pairs_total}")

    # Sample pairs if needed
    if max_pairs is not None and n_pairs_total > max_pairs:
        rng = np.random.RandomState(random_state)
        sampled_indices = rng.choice(n_pairs_total, size=max_pairs, replace=False)
        pairs = [pairs[i] for i in sampled_indices]
        logger.info(f"Sampled {max_pairs} pairs from {n_pairs_total}")
    else:
        max_pairs = n_pairs_total

    # Build pairwise features
    X_pairwise_list = []

    for i, j in pairs:
        f1 = features[i]
        f2 = features[j]

        if pairwise_method == "difference":
            pairwise_feat = np.abs(f1 - f2)
        elif pairwise_method == "concat":
            pairwise_feat = np.concatenate([f1, f2])
        elif pairwise_method == "product":
            pairwise_feat = f1 * f2
        elif pairwise_method == "combined":
            pairwise_feat = np.concatenate(
                [
                    np.abs(f1 - f2),  # difference
                    f1 * f2,  # product
                    (f1 + f2) / 2,  # average
                ]
            )
        else:
            raise ValueError(f"Unknown pairwise_method: {pairwise_method}")

        X_pairwise_list.append(pairwise_feat)

    X_pairwise = np.array(X_pairwise_list)

    # Build similarity targets
    if distance_matrix is not None:
        # Use provided distance matrix
        y_distance = np.array([distance_matrix[i, j] for i, j in pairs])
    else:
        # Use Euclidean distance in feature space
        y_distance = np.array(
            [np.linalg.norm(features[i] - features[j]) for i, j in pairs]
        )

    # Convert distance to similarity: sim = 1 / (1 + dist)
    y_similarity = 1.0 / (1.0 + y_distance)

    logger.info(f"Pairwise features: {X_pairwise.shape}")
    logger.info(
        f"Similarity range: [{y_similarity.min():.4f}, {y_similarity.max():.4f}]"
    )

    return X_pairwise, y_similarity


def train_similarity_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train multiple regression models for similarity prediction.

    Parameters
    ----------
    X_train : np.ndarray
        Training pairwise features.
    y_train : np.ndarray
        Training similarity targets.
    models : dict, optional
        Dictionary of model names to sklearn regressors.
        If None, uses default models (Ridge, RandomForest, GradientBoosting).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    trained_models : dict
        Dictionary of model names to fitted regressors.

    Notes
    -----
    Default models:
    - Ridge Regression: Linear regression with L2 regularization
    - Random Forest: Ensemble decision trees
    - Gradient Boosting: Boosted decision trees (usually best performance)
    """
    logger.info("Training similarity regressors...")

    if models is None:
        # Default models
        models = {
            "ridge": Ridge(
                alpha=1.0,
                random_state=random_state,
            ),
            "random_forest": RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(
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


def evaluate_similarity_regressors(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Evaluate similarity regression performance.

    Parameters
    ----------
    models : dict
        Dictionary of model names to fitted regressors.
    X_test : np.ndarray
        Testing pairwise features.
    y_test : np.ndarray
        Testing similarity targets.
    X_train : np.ndarray, optional
        Training features for cross-validation.
    y_train : np.ndarray, optional
        Training targets for cross-validation.
    cv_folds : int, default=5
        Number of cross-validation folds.

    Returns
    -------
    results_df : pd.DataFrame
        Evaluation metrics for each model.
        Columns: mse, rmse, mae, r2, cv_r2_mean, cv_r2_std

    Notes
    -----
    - Reports test set performance for all models
    - Optionally reports cross-validation R² on training set
    - Lower MSE/MAE is better, higher R² is better
    """
    logger.info("Evaluating similarity regressors...")

    results = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        # Predictions
        y_pred = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        logger.info(
            f"{name} - MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}"
        )

        # Cross-validation
        cv_r2_mean = np.nan
        cv_r2_std = np.nan
        if X_train is not None and y_train is not None:
            cv_results = cross_validate(
                model, X_train, y_train, cv=cv_folds, scoring="r2", n_jobs=-1
            )
            cv_r2_mean = np.mean(cv_results["test_score"])
            cv_r2_std = np.std(cv_results["test_score"])
            logger.info(f"{name} - CV R²: {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")

        results.append(
            {
                "model": name,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
            }
        )

    results_df = pd.DataFrame(results)
    logger.info(f"Evaluation complete. Results:\n{results_df}")

    return results_df


def predict_similarity(
    model: Any,
    features1: np.ndarray,
    features2: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    pairwise_method: str = "difference",
) -> np.ndarray:
    """
    Predict similarity between two sets of graph features.

    Parameters
    ----------
    model : sklearn regressor
        Trained similarity regressor.
    features1 : np.ndarray
        Feature array for first set of graphs (n_graphs, n_features).
    features2 : np.ndarray
        Feature array for second set of graphs (n_graphs, n_features).
        Must have same shape as features1.
    scaler : StandardScaler, optional
        Fitted scaler for feature normalization.
    pairwise_method : str, default="difference"
        Method for constructing pairwise features (must match training).

    Returns
    -------
    similarities : np.ndarray
        Predicted similarity scores (n_graphs,).

    Notes
    -----
    - features1 and features2 must have same number of rows
    - Uses same pairwise_method as during training
    """
    if features1.shape != features2.shape:
        raise ValueError("features1 and features2 must have same shape")

    n_pairs = len(features1)

    # Build pairwise features
    X_pairwise_list = []
    for i in range(n_pairs):
        f1 = features1[i]
        f2 = features2[i]

        if pairwise_method == "difference":
            pairwise_feat = np.abs(f1 - f2)
        elif pairwise_method == "concat":
            pairwise_feat = np.concatenate([f1, f2])
        elif pairwise_method == "product":
            pairwise_feat = f1 * f2
        elif pairwise_method == "combined":
            pairwise_feat = np.concatenate(
                [
                    np.abs(f1 - f2),
                    f1 * f2,
                    (f1 + f2) / 2,
                ]
            )
        else:
            raise ValueError(f"Unknown pairwise_method: {pairwise_method}")

        X_pairwise_list.append(pairwise_feat)

    X_pairwise = np.array(X_pairwise_list)

    # Scale if scaler provided
    if scaler is not None:
        X_pairwise = scaler.transform(X_pairwise)

    # Predict
    similarities = model.predict(X_pairwise)

    return similarities


def save_regressor(
    model: Any,
    scaler: Optional[StandardScaler],
    pairwise_method: str,
    feature_columns: List[str],
    output_path: Path,
) -> None:
    """
    Save trained regressor and metadata.

    Parameters
    ----------
    model : sklearn regressor
        Trained regressor.
    scaler : StandardScaler or None
        Fitted scaler.
    pairwise_method : str
        Pairwise feature construction method.
    feature_columns : list of str
        List of feature column names.
    output_path : Path
        Output file path (.joblib).

    Notes
    -----
    - Saves model, scaler, pairwise_method, and feature columns in single file
    - Use .joblib extension for efficiency
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    regressor_data = {
        "model": model,
        "scaler": scaler,
        "pairwise_method": pairwise_method,
        "feature_columns": feature_columns,
    }

    joblib.dump(regressor_data, output_path)
    logger.info(f"Regressor saved to {output_path}")


def load_regressor(
    input_path: Path,
) -> Tuple[Any, Optional[StandardScaler], str, List[str]]:
    """
    Load trained regressor and metadata.

    Parameters
    ----------
    input_path : Path
        Input file path (.joblib).

    Returns
    -------
    model : sklearn regressor
        Trained regressor.
    scaler : StandardScaler or None
        Fitted scaler.
    pairwise_method : str
        Pairwise feature construction method.
    feature_columns : list of str
        List of feature column names.
    """
    input_path = Path(input_path)
    regressor_data = joblib.load(input_path)

    model = regressor_data["model"]
    scaler = regressor_data.get("scaler")
    pairwise_method = regressor_data["pairwise_method"]
    feature_columns = regressor_data["feature_columns"]

    logger.info(f"Regressor loaded from {input_path}")
    return model, scaler, pairwise_method, feature_columns


# Example usage
if __name__ == "__main__":
    from app.core.logging import setup_logging
    from scientific_api.ml.features.feature_table import load_feature_table

    setup_logging()

    # Example: Load feature table and train similarity regressors
    feature_table_path = Path("data/processed/features/feature_table.parquet")

    if feature_table_path.exists():
        logger.info(f"Loading feature table from {feature_table_path}...")
        feature_table = load_feature_table(feature_table_path)

        # Build pairwise features
        X_pairwise, y_similarity = build_pairwise_features(
            feature_table,
            pairwise_method="combined",  # Use combined features
            max_pairs=10000,  # Limit to 10k pairs for speed
            random_state=42,
        )

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pairwise, y_similarity, test_size=0.2, random_state=42
        )

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train models
        trained_models = train_similarity_regressors(X_train, y_train)

        # Evaluate models
        results = evaluate_similarity_regressors(
            trained_models, X_test, y_test, X_train, y_train, cv_folds=5
        )

        print("\n=== Similarity Regression Results ===")
        print(results)

        # Save best model (e.g., gradient_boosting)
        best_model_name = results.loc[results["r2"].idxmax(), "model"]
        best_model = trained_models[best_model_name]

        feature_cols = [
            col
            for col in feature_table.columns
            if col not in {"graph_id", "system_type"}
            and pd.api.types.is_numeric_dtype(feature_table[col])
        ]

        output_dir = Path("data/models")
        save_regressor(
            best_model,
            scaler,
            "combined",
            feature_cols,
            output_dir / f"regressor_{best_model_name}.joblib",
        )

        logger.info(
            f"Best model: {best_model_name} (R²={results.loc[results['r2'].idxmax(), 'r2']:.4f})"
        )
    else:
        logger.error(f"Feature table not found at {feature_table_path}")
        logger.info(
            "Run build_feature_table_from_directory() first to generate feature table"
        )
