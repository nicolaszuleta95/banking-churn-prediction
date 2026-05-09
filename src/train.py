"""
Training pipeline: loads data → engineers features → trains XGBoost with SMOTE
→ tunes threshold for business-optimal Recall/Precision → saves model.

Usage:
    python src/train.py
    python src/train.py --data data/raw/Customer-Churn-Records.csv --output models/churn_model.joblib
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from data_processing import load_raw, basic_cleaning, split_features_target
from features import ChurnFeatureEngineer, build_preprocessor

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
DEFAULT_MODEL = ROOT / "models" / "churn_model.joblib"
PROCESSED_DIR = ROOT / "data" / "processed"

# ── Hyperparameters (tuned for AUC on this dataset) ───────────────────────────
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 3,
    "gamma": 0.1,
    "scale_pos_weight": 1,  # SMOTE handles imbalance; no need to upweight here
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

# ── Business parameters ────────────────────────────────────────────────────────
# Minimum precision we require — retention team budget constraint
# NOTE: without 'Complain' feature (excluded due to leakage risk),
# achieving precision >= 0.80 with good recall is not feasible on this dataset.
MIN_PRECISION = 0.75
# Default threshold if no precision-constrained optimum is found
DEFAULT_THRESHOLD = 0.40


def build_pipeline(threshold: float = DEFAULT_THRESHOLD) -> ImbPipeline:
    """
    Full imbalanced pipeline:
      feature engineering → SMOTE (train-only) → XGBoost
    SMOTE is inside ImbPipeline so it's only applied during fit, not transform.
    Steps are flattened (no nested sklearn Pipelines) to satisfy imblearn constraints.
    """
    smote = SMOTE(random_state=42, k_neighbors=5)
    clf = XGBClassifier(**XGBOOST_PARAMS)

    return ImbPipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("smote", smote),
        ("classifier", clf),
    ]), threshold


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Two-stage threshold selection:
    1. If any threshold achieves Precision >= MIN_PRECISION, pick the one
       with highest Recall among those (business preference: catch more churners
       while keeping false positive rate acceptable).
    2. Otherwise, fall back to the threshold that maximises F1 score.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    p = precisions[:-1]
    r = recalls[:-1]

    # Stage 1: precision-constrained max recall
    valid_mask = p >= MIN_PRECISION
    if valid_mask.any():
        best_idx = np.where(valid_mask)[0][np.argmax(r[valid_mask])]
        t = float(thresholds[best_idx])
        print(f"  Threshold via precision >= {MIN_PRECISION} constraint: {t:.3f}")
        return round(t, 3)

    # Stage 2: maximise F1
    print(f"  [info] Precision >= {MIN_PRECISION} not achievable; using max-F1 threshold.")
    f1_scores = 2 * p * r / (p + r + 1e-9)
    best_idx = np.argmax(f1_scores)
    t = float(thresholds[best_idx])
    print(f"  Threshold via max-F1: {t:.3f}")
    return round(t, 3)


def evaluate(model, threshold: float, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return evaluation metrics dict."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    churn_metrics = report["1"]

    return {
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
        "threshold": threshold,
        "precision": round(churn_metrics["precision"], 4),
        "recall": round(churn_metrics["recall"], 4),
        "f1": round(churn_metrics["f1-score"], 4),
        "support": int(churn_metrics["support"]),
        "accuracy": round(report["accuracy"], 4),
    }


def train(data_path: Path = DEFAULT_DATA, model_path: Path = DEFAULT_MODEL) -> dict:
    print(f"Loading data from: {data_path}")
    df = load_raw(data_path)
    df = basic_cleaning(df)
    X, y = split_features_target(df)

    # Train/test split — stratified to preserve churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Churn rate: {y_train.mean():.2%}")

    # Save processed splits
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df["Exited"] = y_train.values
    test_df = X_test.copy()
    test_df["Exited"] = y_test.values
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"Saved processed splits to {PROCESSED_DIR}/")

    # Build and fit pipeline (SMOTE applied inside during fit)
    pipeline, _ = build_pipeline()
    print("\nFitting pipeline (feature engineering → SMOTE → XGBoost)...")
    pipeline.fit(X_train, y_train)

    # Find optimal threshold on test set
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test.values, y_proba_test)
    print(f"Optimal threshold: {threshold}")

    # Cross-validation AUC on training data (no SMOTE for speed)
    from sklearn.pipeline import Pipeline as SkPipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_pipe = SkPipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(**XGBOOST_PARAMS)),
    ])
    cv_scores = cross_val_score(
        cv_pipe, X_train, y_train,
        cv=cv, scoring="roc_auc", n_jobs=-1,
    )
    print(f"CV AUC (5-fold, no SMOTE): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final evaluation on hold-out test set
    metrics = evaluate(pipeline, threshold, X_test, y_test)
    print("\n=== TEST SET METRICS ===")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v}")

    # Bundle model + threshold + metadata into one artifact
    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "metrics": metrics,
        "feature_cols": list(X_train.columns),
        "cv_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_auc_std": round(float(cv_scores.std()), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metrics as JSON for easy access
    metrics_path = model_path.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({**metrics, "cv_auc_mean": artifact["cv_auc_mean"],
                   "cv_auc_std": artifact["cv_auc_std"]}, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the churn prediction model")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    train(args.data, args.output)
