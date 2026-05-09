"""
Inference functions used by the Streamlit app and any external consumer.

Loads the trained artifact (pipeline + threshold + metadata) and exposes:
  - load_model()        → loads artifact from disk
  - predict_proba()     → churn probability for one or more customers
  - predict()           → binary label using stored threshold
  - explain()           → SHAP values for a single customer
  - business_impact()   → revenue-at-risk projection for a cohort
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "churn_model.joblib"

# Business assumptions (overridable)
DEFAULT_CLV = 800          # Average Customer Lifetime Value (USD)
DEFAULT_CONTACT_COST = 15  # Cost per retention contact (USD)
DEFAULT_RETENTION_RATE = 0.30  # % of contacted at-risk customers retained


def load_model(model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    """
    Load the model artifact from disk.
    Returns dict with keys: pipeline, threshold, metrics, feature_cols, cv_auc_mean, cv_auc_std
    Raises FileNotFoundError if model has not been trained yet.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python src/train.py` to train the model first."
        )
    return joblib.load(model_path)


def predict_proba(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Return churn probability for each row in X (values in [0, 1])."""
    return artifact["pipeline"].predict_proba(X)[:, 1]


def predict(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Return binary churn prediction using the stored threshold."""
    proba = predict_proba(artifact, X)
    return (proba >= artifact["threshold"]).astype(int)


def predict_single(artifact: dict, customer: dict) -> dict:
    """
    Predict churn for a single customer dict.
    Returns: {probability, label, threshold, risk_level}
    """
    X = pd.DataFrame([customer])
    proba = float(predict_proba(artifact, X)[0])
    threshold = artifact["threshold"]
    label = int(proba >= threshold)

    if proba < 0.30:
        risk_level = "Low"
    elif proba < threshold:
        risk_level = "Medium"
    elif proba < 0.70:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        "probability": round(proba, 4),
        "label": label,
        "threshold": threshold,
        "risk_level": risk_level,
    }


def explain(artifact: dict, customer: dict, max_display: int = 10) -> dict:
    """
    Compute SHAP values for a single customer.
    Returns: {shap_values, expected_value, feature_names, feature_values}

    Uses TreeExplainer on the XGBoost classifier inside the pipeline.
    The input is transformed through the full feature pipeline before SHAP.
    """
    X_raw = pd.DataFrame([customer])

    # Transform through feature engineering + preprocessing (without classifier)
    pipeline = artifact["pipeline"]
    # Build a sub-pipeline with all steps except the classifier
    from imblearn.pipeline import Pipeline as ImbPipeline
    steps_without_clf = [(name, step) for name, step in pipeline.steps
                         if name not in ("smote", "classifier")]
    from sklearn.pipeline import Pipeline as SkPipeline
    transform_pipe = SkPipeline(steps_without_clf)
    X_transformed = transform_pipe.transform(X_raw)

    classifier = pipeline.named_steps["classifier"]

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)

    # Get feature names from the preprocessor
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Sort by absolute SHAP value
    shap_vals = shap_values[0]
    top_idx = np.argsort(np.abs(shap_vals))[::-1][:max_display]

    return {
        "shap_values": shap_vals[top_idx].tolist(),
        "feature_names": [feature_names[i] for i in top_idx],
        "feature_values": X_transformed[0, top_idx].tolist(),
        "expected_value": float(explainer.expected_value),
        "all_shap_values": shap_vals.tolist(),
        "all_feature_names": feature_names,
    }


def business_impact(
    artifact: dict,
    X: pd.DataFrame,
    clv: float = DEFAULT_CLV,
    contact_cost: float = DEFAULT_CONTACT_COST,
    retention_rate: float = DEFAULT_RETENTION_RATE,
) -> dict:
    """
    Compute business impact metrics for a cohort of customers.

    Returns:
        total_customers: total cohort size
        predicted_churners: customers flagged by model
        revenue_at_risk: CLV * predicted_churners
        recoverable_revenue: revenue_at_risk * retention_rate
        intervention_cost: contact_cost * predicted_churners
        net_value: recoverable_revenue - intervention_cost
        churn_rate_predicted: fraction of cohort flagged
    """
    proba = predict_proba(artifact, X)
    threshold = artifact["threshold"]
    flagged = (proba >= threshold).sum()
    total = len(X)

    revenue_at_risk = flagged * clv
    recoverable = revenue_at_risk * retention_rate
    cost = flagged * contact_cost
    net = recoverable - cost

    return {
        "total_customers": int(total),
        "predicted_churners": int(flagged),
        "churn_rate_predicted": round(float(flagged) / total, 4),
        "revenue_at_risk": round(float(revenue_at_risk), 2),
        "recoverable_revenue": round(float(recoverable), 2),
        "intervention_cost": round(float(cost), 2),
        "net_value": round(float(net), 2),
        "assumptions": {
            "clv": clv,
            "contact_cost": contact_cost,
            "retention_rate": retention_rate,
        },
    }
