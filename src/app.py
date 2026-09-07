"""
app.py  —  Pezesha Loan Risk Dashboard
---------------------------------------
Streamlit web app for loan officers to assess borrower default risk.

Run with:
    streamlit run src/app.py
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# Allow importing sibling modules
sys.path.insert(0, os.path.dirname(__file__))
import preprocess as pp
from explain import get_top3_reasons

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT  = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH    = os.path.join(MODELS_DIR, "xgb_model.pkl")
MEDIANS_PATH  = os.path.join(MODELS_DIR, "fill_medians.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")
EXPLAINER_PATH= os.path.join(MODELS_DIR, "shap_explainer.pkl")

# -------------------------------------------------------------------
# Page config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Pezesha · Loan Risk Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------------------------
# Custom CSS — clean, professional, Africa-inspired palette
# -------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ---- Page background ---- */
.stApp {
    background: linear-gradient(145deg, #0f1923 0%, #132233 60%, #0d2b1e 100%);
    min-height: 100vh;
}

/* ---- Header ---- */
.pez-header {
    background: linear-gradient(90deg, #00b96b 0%, #00e88a 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 18px;
}
.pez-header h1 {
    color: #0f1923;
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0;
    line-height: 1.15;
}
.pez-header p {
    color: #0f1923cc;
    margin: 0;
    font-size: 0.92rem;
    font-weight: 500;
}

/* ---- Card ---- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 14px;
    padding: 26px 28px;
    margin-bottom: 20px;
}
.card-title {
    color: #00e88a;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ---- Risk badge ---- */
.risk-green  { background:#003d22; border:1.5px solid #00b96b; color:#00e88a; }
.risk-amber  { background:#3d2a00; border:1.5px solid #e8a200; color:#ffcc30; }
.risk-red    { background:#3d0000; border:1.5px solid #e83000; color:#ff6b6b; }
.risk-badge {
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    margin: 20px 0 10px;
}
.risk-badge .score {
    font-size: 4rem;
    font-weight: 800;
    font-family: 'DM Mono', monospace;
    line-height: 1;
}
.risk-badge .label {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 8px;
}
.risk-badge .sublabel {
    font-size: 0.82rem;
    margin-top: 6px;
    opacity: 0.75;
}

/* ---- Reasons ---- */
.reason-item {
    background: rgba(255,255,255,0.04);
    border-left: 3px solid #00e88a;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #d4e8dd;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ---- Streamlit overrides ---- */
label, .stSelectbox label, .stNumberInput label, .stSlider label {
    color: #a8c4b4 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
}
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #e8f4ee !important;
    border-radius: 8px !important;
}
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #00b96b, #00e88a) !important;
    color: #0f1923 !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 0 !important;
    margin-top: 8px;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88 !important; }
div[data-testid="stForm"] { background: transparent !important; border: none !important; }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# Load model artefacts (cached so they load only once)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading risk model…")
def load_artefacts():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None

    with open(MODEL_PATH,     "rb") as f: model     = pickle.load(f)
    with open(MEDIANS_PATH,   "r")  as f: medians   = json.load(f)
    with open(FEATURES_PATH,  "r")  as f: feat_cols = json.load(f)
    with open(EXPLAINER_PATH, "rb") as f: explainer = pickle.load(f)

    # Inject medians back into the preprocess module
    pp.FILL_MEDIANS.update(medians)
    return model, medians, feat_cols, explainer


model, medians, feat_cols, explainer = load_artefacts()

# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown("""
<div class="pez-header">
  <div style="font-size:2.4rem">🏦</div>
  <div>
    <h1>Pezesha · Loan Risk Predictor</h1>
    <p>AI-powered default risk assessment for loan officers</p>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Guard: model not trained yet
# -------------------------------------------------------------------
if model is None:
    st.error("⚠️  No trained model found in `/models/`.  "
             "Please run `python src/train_model.py` first, then refresh this page.")
    st.stop()

# -------------------------------------------------------------------
# Input form
# -------------------------------------------------------------------
st.markdown('<div class="card"><div class="card-title">📋 Borrower Details</div>', unsafe_allow_html=True)

with st.form("borrower_form"):
    col1, col2 = st.columns(2)

    with col1:
        loan_amount = st.number_input(
            "Loan Amount (KES)", min_value=500.0, max_value=500_000.0,
            value=15_000.0, step=500.0
        )
        loan_date = st.date_input("Loan Date")
        business_type = st.selectbox("Business Type", options=pp.BUSINESS_TYPES)

    with col2:
        business_location = st.selectbox(
            "Country / Business Location",
            options=pp.ALL_AFRICAN_COUNTRIES,
            index=pp.ALL_AFRICAN_COUNTRIES.index("Kenya") if "Kenya" in pp.ALL_AFRICAN_COUNTRIES else 0
        )
        years_in_business = st.number_input(
            "Years in Business", min_value=0.1, max_value=30.0,
            value=2.0, step=0.5
        )
        mobile_score = st.slider(
            "Mobile Money Activity Score  (0 = none · 100 = very active)",
            min_value=0.0, max_value=100.0, value=55.0, step=0.5
        )

    submitted = st.form_submit_button("🔍  Assess Risk")

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------
if submitted:
    # Build a single-row DataFrame matching the raw CSV format
    row = pd.DataFrame([{
        "borrower_id":                "NEWAPPLICANT",
        "loan_amount":                loan_amount,
        "loan_date":                  str(loan_date),
        "business_type":              business_type,
        "business_location":          business_location,
        "years_in_business":          years_in_business,
        "repayment_history":          "on_time",   # placeholder — won't be used
        "mobile_money_activity_score": mobile_score,
    }])

    # Clean (fit=False → use saved medians, not recalculate)
    X = pp.clean_dataframe(row, fit=False)

    # Align columns exactly to training order; add any missing as 0
    for c in feat_cols:
        if c not in X.columns:
            X[c] = 0
    X = X[feat_cols]

    # Predict probability → scale to 0-100 risk score
    prob       = float(model.predict_proba(X)[0, 1])
    risk_score = int(round(prob * 100))

    # Risk tier
    if risk_score < 35:
        tier, css_class, description = "LOW RISK", "risk-green", "Loan can proceed with standard terms."
    elif risk_score < 65:
        tier, css_class, description = "MEDIUM RISK", "risk-amber", "Proceed with caution — consider reduced amount or guarantor."
    else:
        tier, css_class, description = "HIGH RISK", "risk-red", "High probability of default — escalate for manual review."

    # SHAP explanations
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):   # binary classifier returns list of 2
        shap_vals = shap_vals[1]
    reasons = get_top3_reasons(shap_vals[0], feat_cols)

    # --- Display result ---
    st.markdown(f"""
    <div class="risk-badge {css_class}">
      <div class="score">{risk_score}</div>
      <div class="label">{tier}</div>
      <div class="sublabel">{description}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="card-title">🔍 Top 3 Reasons Behind This Score</div>', unsafe_allow_html=True)
    for i, reason in enumerate(reasons, 1):
        st.markdown(f'<div class="reason-item"><strong>#{i}</strong> &nbsp; {reason}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Summary table ---
    st.markdown('<div class="card"><div class="card-title">📊 Applicant Summary</div>', unsafe_allow_html=True)
    summary = {
        "Loan Amount (KES)": f"{loan_amount:,.0f}",
        "Business Type": business_type.replace("_", " ").title(),
        "Location": business_location,
        "Years in Business": years_in_business,
        "Mobile Money Score": f"{mobile_score:.1f} / 100",
        "Risk Score": f"{risk_score} / 100",
        "Risk Tier": tier,
    }
    df_summary = pd.DataFrame(list(summary.items()), columns=["Field", "Value"])
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("""
<div style="text-align:center;color:#3d6b52;font-size:0.78rem;margin-top:40px;padding-bottom:20px;">
  Pezesha AI Risk Engine · Powered by XGBoost + SHAP · For authorised loan officers only
</div>
""", unsafe_allow_html=True)
