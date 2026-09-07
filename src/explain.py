"""
explain.py
----------
Uses SHAP values to produce 3 plain-English reasons for each risk score.
Called by the Streamlit app (app.py).
"""

import numpy as np

# -------------------------------------------------------------------
# Human-readable templates for every feature
# -------------------------------------------------------------------
FEATURE_TEMPLATES = {
    # Numeric features
    "mobile_money_activity_score": {
        "high":  "Strong mobile money activity suggests reliable cash flow",
        "low":   "Low mobile money activity — limited transaction history detected",
    },
    "years_in_business": {
        "high":  "Long-established business reduces default risk",
        "low":   "New or young business — limited track record available",
    },
    "loan_amount": {
        "high":  "Large loan amount increases repayment pressure",
        "low":   "Small loan amount is generally easier to repay",
    },
    "loan_amount_log": {
        "high":  "Loan size is on the higher end relative to typical borrowers",
        "low":   "Loan size is modest and manageable",
    },
    "loan_age_days": {
        "high":  "Loan originated recently — limited payment history to assess",
        "low":   "Older loan cohort — more historical data available",
    },
}

# Prefix-based templates for one-hot encoded columns
LOCATION_TEMPLATE = {
    "high": "Borrower is located in {loc}, which has higher default rates historically",
    "low":  "Borrower is located in {loc}, which has lower default rates historically",
}

BIZTYPE_TEMPLATE = {
    "high": "The {biz} sector has shown elevated default risk in the portfolio",
    "low":  "The {biz} sector has shown lower default risk in the portfolio",
}


def _feature_to_english(feature_name: str, shap_value: float) -> str:
    """Convert a single (feature, shap_value) pair into a sentence."""
    direction = "high" if shap_value > 0 else "low"

    # Exact numeric feature match
    if feature_name in FEATURE_TEMPLATES:
        return FEATURE_TEMPLATES[feature_name][direction]

    # One-hot: business_type_X
    if feature_name.startswith("business_type_"):
        biz = feature_name.replace("business_type_", "").replace("_", " ").title()
        template = BIZTYPE_TEMPLATE[direction]
        return template.format(biz=biz)

    # One-hot: business_location_X
    if feature_name.startswith("business_location_"):
        loc = feature_name.replace("business_location_", "")
        template = LOCATION_TEMPLATE[direction]
        return template.format(loc=loc)

    # Fallback — clean up underscores
    readable = feature_name.replace("_", " ").capitalize()
    verb = "increases" if shap_value > 0 else "decreases"
    return f"{readable} {verb} the predicted risk"


def get_top3_reasons(shap_values: np.ndarray, feature_names: list) -> list[str]:
    """
    Given a 1-D array of SHAP values for ONE borrower and the matching
    feature names, return the 3 most impactful plain-English reasons.

    Parameters
    ----------
    shap_values   : 1-D numpy array, length == len(feature_names)
    feature_names : list of strings matching column order

    Returns
    -------
    List of 3 strings (most → least impactful)
    """
    abs_vals = np.abs(shap_values)
    top_idx  = np.argsort(abs_vals)[::-1][:3]

    reasons = []
    for idx in top_idx:
        fname   = feature_names[idx]
        sval    = float(shap_values[idx])
        reasons.append(_feature_to_english(fname, sval))

    return reasons
