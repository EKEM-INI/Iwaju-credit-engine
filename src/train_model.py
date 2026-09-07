"""
train_model.py
--------------
Trains the XGBoost default-prediction model on historical loan data,
evaluates it, and saves the model + supporting artefacts to /models/.

Usage:
    python src/train_model.py                        # uses default CSV
    python src/train_model.py data/my_loans.csv      # custom CSV path
"""

import sys
import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier
import shap

# Allow importing from src/ when running from project root
sys.path.insert(0, os.path.dirname(__file__))
import preprocess as pp

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH    = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROJECT_ROOT, "data", "historical_loans.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH      = os.path.join(MODELS_DIR, "xgb_model.pkl")
MEDIANS_PATH    = os.path.join(MODELS_DIR, "fill_medians.json")
FEATURES_PATH   = os.path.join(MODELS_DIR, "feature_columns.json")
EXPLAINER_PATH  = os.path.join(MODELS_DIR, "shap_explainer.pkl")


# -------------------------------------------------------------------
# 1. Load raw data
# -------------------------------------------------------------------
print("\n📂  Loading data from:", DATA_PATH)
df_raw = pd.read_csv(DATA_PATH)
print(f"    Rows: {len(df_raw)}   Columns: {list(df_raw.columns)}")

# -------------------------------------------------------------------
# 2. Build target before cleaning (cleaning drops repayment_history)
# -------------------------------------------------------------------
y = pp.build_target(df_raw)
print(f"\n🎯  Target distribution:\n{y.value_counts().rename({0:'no default',1:'default'})}")

# -------------------------------------------------------------------
# 3. Clean & feature-engineer  (fit=True → learns medians)
# -------------------------------------------------------------------
X = pp.clean_dataframe(df_raw, fit=True)
feature_cols = pp.get_feature_columns(X)
X = X[feature_cols]  # enforce consistent ordering

print(f"\n🔧  Features after engineering ({len(feature_cols)} total):")
print("   ", feature_cols)

# -------------------------------------------------------------------
# 4. Train / test split (stratified so both classes appear in test)
# -------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\n📊  Train: {len(X_train)} rows   Test: {len(X_test)} rows")

# -------------------------------------------------------------------
# 5. Train XGBoost
# -------------------------------------------------------------------
print("\n🚀  Training XGBoost model …")

# Class imbalance: weight the minority (default) class
scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale,
    use_label_encoder=False,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# -------------------------------------------------------------------
# 6. Evaluate
# -------------------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.50).astype(int)
auc    = roc_auc_score(y_test, y_prob)

print(f"\n✅  ROC-AUC on test set: {auc:.4f}")
print("\n📋  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
print("Confusion Matrix (rows=actual, cols=predicted):")
print(confusion_matrix(y_test, y_pred))

# -------------------------------------------------------------------
# 7. Build SHAP explainer (TreeExplainer is fast and exact for XGBoost)
# -------------------------------------------------------------------
print("\n🔍  Building SHAP explainer …")
explainer = shap.TreeExplainer(model)

# -------------------------------------------------------------------
# 8. Save everything
# -------------------------------------------------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"💾  Model saved        → {MODEL_PATH}")

with open(MEDIANS_PATH, "w") as f:
    json.dump(pp.FILL_MEDIANS, f, indent=2)
print(f"💾  Fill medians saved → {MEDIANS_PATH}")

with open(FEATURES_PATH, "w") as f:
    json.dump(feature_cols, f, indent=2)
print(f"💾  Feature list saved → {FEATURES_PATH}")

with open(EXPLAINER_PATH, "wb") as f:
    pickle.dump(explainer, f)
print(f"💾  SHAP explainer     → {EXPLAINER_PATH}")

print("\n🎉  Training complete!  Run the dashboard with:  streamlit run src/app.py\n")
