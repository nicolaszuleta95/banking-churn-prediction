"""
Feature engineering as a sklearn-compatible transformer.

All transformations are derived from EDA insights (notebook 01_eda.ipynb):
- age_segment: Senior customers (51-60) churn 3× more than young customers
- balance_per_product: financial engagement ratio
- zero_balance: zero-balance customers churn less (13.8% vs 24.1%)
- products_3plus: NumOfProducts >= 3 → churn rate 82-100%
- is_german: Germany churn rate 32% vs France/Spain ~16%
- engagement_score: composite activity signal
- risk_profile: combined balance + products + activity risk

NOTE on 'Complain': Complain=1 predicts 99.5% churn. This feature is included
but should be understood as potentially concurrent with churn (not strictly
prior), which may inflate model performance. See model_card.md for discussion.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer


# Card type ordinal ranking (from lowest to highest tier)
CARD_ORDER = [["SILVER", "GOLD", "PLATINUM", "DIAMOND"]]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds derived features before encoding/scaling.
    Designed to be the first step in the sklearn Pipeline.
    Input: raw DataFrame with original columns (no ID cols).
    Output: DataFrame with original + engineered columns.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # --- Age segment ---
        X["age_segment"] = pd.cut(
            X["Age"],
            bins=[0, 35, 50, 60, 200],
            labels=["young", "mid", "senior", "elder"],
        ).astype(str)

        # --- Balance features ---
        X["zero_balance"] = (X["Balance"] == 0).astype(int)
        # Avoid division by zero; products=0 shouldn't happen but guard anyway
        X["balance_per_product"] = X["Balance"] / X["NumOfProducts"].clip(lower=1)

        # --- Product risk flag ---
        X["products_3plus"] = (X["NumOfProducts"] >= 3).astype(int)

        # --- Geography flag ---
        X["is_german"] = (X["Geography"] == "Germany").astype(int)

        # --- Engagement score ---
        # Combines activity, tenure, and balance presence
        X["engagement_score"] = (
            X["IsActiveMember"]
            * (1 - X["zero_balance"])
            * np.log1p(X["Tenure"])
        )

        # --- Risk profile ---
        # High balance + inactive + Germany → high risk
        X["risk_profile"] = (
            (1 - X["IsActiveMember"])
            * X["is_german"]
            + X["products_3plus"]
            + X["zero_balance"].apply(lambda v: 0 if v else 1)
        ).clip(upper=3)

        return X

    def get_feature_names_out(self, input_features=None):
        engineered = [
            "age_segment", "zero_balance", "balance_per_product",
            "products_3plus", "is_german", "engagement_score", "risk_profile",
        ]
        if input_features is not None:
            return list(input_features) + engineered
        return engineered


def build_preprocessor() -> ColumnTransformer:
    """
    Build the ColumnTransformer that encodes and scales all features
    after ChurnFeatureEngineer has added derived columns.
    """
    numeric_features = [
        "CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary",
        "Point Earned", "balance_per_product", "engagement_score",
    ]

    binary_passthrough = [
        "NumOfProducts", "HasCrCard", "IsActiveMember",
        "zero_balance", "products_3plus", "is_german", "risk_profile",
        "Satisfaction Score",
    ]

    categorical_ohe = ["Geography", "Gender"]

    ordinal_card = ["Card Type"]

    age_segment_ohe = ["age_segment"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("pass", "passthrough", binary_passthrough),
            ("ohe_cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_ohe),
            ("ord_card", OrdinalEncoder(categories=CARD_ORDER, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_card),
            ("ohe_age", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), age_segment_ohe),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_feature_pipeline() -> Pipeline:
    """
    Returns the full feature pipeline:
    ChurnFeatureEngineer → ColumnTransformer (encode + scale)
    This pipeline outputs a numpy array ready for a classifier.
    """
    return Pipeline([
        ("engineer", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
    ])
