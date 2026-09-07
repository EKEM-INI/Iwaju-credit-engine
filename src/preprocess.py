"""
preprocess.py
-------------
Handles all data cleaning and feature engineering for Pezesha loan data.
Called by both train_model.py (batch) and app.py (single borrower).
"""

import pandas as pd
import numpy as np
from datetime import datetime


# -------------------------------------------------------------------
# Constants — must stay identical between training and inference
# -------------------------------------------------------------------
BUSINESS_TYPES = [
    "retail_shop", "food_vendor", "salon_barber", "transport",
    "tailoring", "electronics_repair", "agro_dealer", "pharmacy",
    "hardware_store", "mobile_money_agent"
]

LOCATIONS = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Thika", "Malindi", "Kitale", "Garissa", "Nyeri"
]

# Medians learned at training time — populated by train_model.py and
# saved alongside the model so the app can load them.
FILL_MEDIANS: dict = {}


# -------------------------------------------------------------------
# Public helpers
# -------------------------------------------------------------------

def clean_dataframe(df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
    """
    Clean and engineer features from raw loan data.

    Parameters
    ----------
    df   : raw DataFrame (as loaded from CSV or built from form inputs)
    fit  : True when called from training (computes medians);
           False when called from inference (uses saved FILL_MEDIANS).

    Returns
    -------
    feature DataFrame ready for XGBoost (no target column).
    """
    df = df.copy()

    # 1. Parse loan_date → age in days since 2020-01-01
    reference_date = datetime(2020, 1, 1)
    df["loan_date"] = pd.to_datetime(df["loan_date"], errors="coerce")
    df["loan_age_days"] = (df["loan_date"] - reference_date).dt.days
    df.drop(columns=["loan_date"], inplace=True, errors="ignore")

    # 2. Drop identifier columns that leak or are irrelevant
    df.drop(columns=["borrower_id", "repayment_history"], inplace=True, errors="ignore")

    # 3. Numeric columns — fill missing with median
    numeric_cols = ["loan_amount", "years_in_business",
                    "mobile_money_activity_score", "loan_age_days"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0
        if fit:
            FILL_MEDIANS[col] = float(df[col].median())
        median_val = FILL_MEDIANS.get(col, 0)
        df[col] = df[col].fillna(median_val)

    # 4. Log-transform skewed loan amount
    df["loan_amount_log"] = np.log1p(df["loan_amount"])

    # 5. Categorical columns — one-hot encode
    df["business_type"] = df["business_type"].fillna("unknown")
    df["business_location"] = df["business_location"].fillna("unknown")

    df = pd.get_dummies(df, columns=["business_type", "business_location"], dtype=int)

    # Ensure all expected dummy columns are present (important at inference)
    for bt in BUSINESS_TYPES:
        col = f"business_type_{bt}"
        if col not in df.columns:
            df[col] = 0
    for loc in LOCATIONS:
        col = f"business_location_{loc}"
        if col not in df.columns:
            df[col] = 0

    # Drop any stray unknown columns created at inference
    for c in list(df.columns):
        if "business_type_" in c and c not in [f"business_type_{x}" for x in BUSINESS_TYPES]:
            df.drop(columns=[c], inplace=True)
        if "business_location_" in c and c not in [f"business_location_{x}" for x in LOCATIONS]:
            df.drop(columns=[c], inplace=True)

    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """
    Convert repayment_history text → binary target.
      defaulted → 1  (bad)
      late / on_time → 0  (good)
    """
    mapping = {"defaulted": 1, "late": 0, "on_time": 0}
    return df["repayment_history"].map(mapping).astype(int)


def get_feature_columns(X: pd.DataFrame) -> list:
    """Return sorted list of feature column names (for consistent ordering)."""
    return sorted(X.columns.tolist())
