"""
Data loading, validation, and cleaning for the banking churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Columns that are identifiers — not features
_ID_COLS = ["RowNumber", "CustomerId", "Surname"]

TARGET = "Exited"

# 'Complain' is excluded from production features: in this dataset, complaints
# are recorded concurrently with churn (99.5% of complainers churn), making it
# a near-perfect but potentially leaky predictor. See model_card.md for details.
FEATURE_COLS = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure",
    "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Satisfaction Score",
    "Card Type", "Point Earned",
]

# Full feature set including Complain — for research/analysis only
FEATURE_COLS_WITH_COMPLAIN = FEATURE_COLS + ["Complain"]

EXPECTED_COLUMNS = _ID_COLS + FEATURE_COLS + [TARGET]

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary", "Point Earned",
]

CATEGORICAL_FEATURES = ["Geography", "Gender", "Card Type"]

BINARY_FEATURES = ["HasCrCard", "IsActiveMember"]

ORDINAL_FEATURES = ["Satisfaction Score"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and drop identifier columns."""
    df = pd.read_csv(path)
    _validate(df)
    df = df.drop(columns=_ID_COLS)
    return df


def _validate(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    if df[TARGET].isna().any():
        raise ValueError("Target column 'Exited' contains null values.")

    if df[TARGET].nunique() != 2:
        raise ValueError(f"Target must be binary, found: {df[TARGET].unique()}")


def split_features_target(df: pd.DataFrame):
    """Return (X, y) splitting features from target."""
    return df[FEATURE_COLS], df[TARGET]


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal cleaning to raw data (after dropping ID cols).
    - Clip CreditScore to valid range [300, 850]
    - Clip Age to [18, 100]
    - Ensure non-negative Balance
    """
    df = df.copy()
    df["CreditScore"] = df["CreditScore"].clip(300, 850)
    df["Age"] = df["Age"].clip(18, 100)
    df["Balance"] = df["Balance"].clip(lower=0)
    return df
