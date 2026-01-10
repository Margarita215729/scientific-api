"""Training utilities for notebook-friendly callable API."""

from __future__ import annotations

from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scientific_api.storage.paths import ensure_dirs, get_outputs_dir, get_reports_dir


def _prepare_features(
    df: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in features dataframe")

    y = df[target].astype(str).values
    drop_cols = {"graph_id", "system_type", "preset", "source", target}
    feature_cols = [
        c
        for c in df.columns
        if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns available for training")

    X = df[feature_cols].copy()
    X = X.fillna(X.mean())
    return X.values, y, feature_cols


def train_and_evaluate(
    features_df: pd.DataFrame, target: str = "source", seed: int = 42
) -> Dict:
    """Train baseline classifiers (LogReg + RF) and save artifacts.

    Returns a metrics dictionary with paths to saved models and metrics tables.
    """

    X, y, feature_cols = _prepare_features(features_df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logreg = LogisticRegression(max_iter=500, random_state=seed)
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=seed, n_jobs=-1, max_depth=None
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    metrics = {
        "target": target,
        "logreg": {
            "accuracy": float(accuracy_score(y_test, y_pred_lr)),
            "f1_macro": float(f1_score(y_test, y_pred_lr, average="macro")),
            "report": classification_report(y_test, y_pred_lr, output_dict=True),
        },
        "random_forest": {
            "accuracy": float(accuracy_score(y_test, y_pred_rf)),
            "f1_macro": float(f1_score(y_test, y_pred_rf, average="macro")),
            "report": classification_report(y_test, y_pred_rf, output_dict=True),
        },
    }

    models_dir = get_outputs_dir() / "models"
    reports_dir = get_reports_dir() / "tables"
    ensure_dirs([models_dir, reports_dir])

    lr_path = models_dir / f"{target}_logreg.joblib"
    rf_path = models_dir / f"{target}_rf.joblib"
    joblib.dump({"model": logreg, "scaler": scaler, "features": feature_cols}, lr_path)
    joblib.dump({"model": rf, "features": feature_cols}, rf_path)

    metrics_path = reports_dir / f"metrics_{target}.json"
    pd.DataFrame(
        [
            {
                "model": "logreg",
                **{k: v for k, v in metrics["logreg"].items() if k != "report"},
            },
            {
                "model": "random_forest",
                **{k: v for k, v in metrics["random_forest"].items() if k != "report"},
            },
        ]
    ).to_json(metrics_path, orient="records", indent=2)

    metrics["artifacts"] = {
        "logreg_path": str(lr_path),
        "rf_path": str(rf_path),
        "metrics_json": str(metrics_path),
    }
    return metrics
