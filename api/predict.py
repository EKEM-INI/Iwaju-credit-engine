import os
import json
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler

# Base directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_JSON_PATH = os.path.join(MODELS_DIR, "xgb_model.json")
MEDIANS_PATH = os.path.join(MODELS_DIR, "fill_medians.json")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

# Cache model and metadata in memory across warm serverless invocations
_MODEL_CACHE = None
_MEDIANS_CACHE = None
_FEATURES_CACHE = None

def get_model_assets():
    global _MODEL_CACHE, _MEDIANS_CACHE, _FEATURES_CACHE
    if _MODEL_CACHE is None:
        if os.path.exists(MODEL_JSON_PATH):
            with open(MODEL_JSON_PATH, "r", encoding="utf-8") as f:
                _MODEL_CACHE = json.load(f)
        if os.path.exists(MEDIANS_PATH):
            with open(MEDIANS_PATH, "r", encoding="utf-8") as f:
                _MEDIANS_CACHE = json.load(f)
        if os.path.exists(FEATURES_PATH):
            with open(FEATURES_PATH, "r", encoding="utf-8") as f:
                _FEATURES_CACHE = json.load(f)
    return _MODEL_CACHE, _MEDIANS_CACHE, _FEATURES_CACHE

# Constants
BUSINESS_TYPES = [
    "retail_shop", "food_vendor", "salon_barber", "transport",
    "tailoring", "electronics_repair", "agro_dealer", "pharmacy",
    "hardware_store", "mobile_money_agent"
]

LOCATIONS = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Thika", "Malindi", "Kitale", "Garissa", "Nyeri"
]

FEATURE_TEMPLATES = {
    "mobile_money_activity_score": {
        "high": "Low mobile money activity — limited transaction history detected",
        "low": "Strong mobile money activity suggests reliable cash flow",
    },
    "years_in_business": {
        "high": "New or young business — limited track record available",
        "low": "Long-established business reduces default risk",
    },
    "loan_amount": {
        "high": "Large loan amount increases repayment pressure",
        "low": "Small loan amount is generally easier to repay",
    },
    "loan_amount_log": {
        "high": "Loan size is on the higher end relative to typical borrowers",
        "low": "Loan size is modest and manageable",
    },
    "loan_age_days": {
        "high": "Loan originated recently — limited payment history to assess",
        "low": "Older loan cohort — more historical data available",
    },
}

def feature_to_english(feature_name: str, impact_value: float) -> str:
    # positive impact pushes default probability up ('high' risk factor)
    direction = "high" if impact_value > 0 else "low"
    
    if feature_name in FEATURE_TEMPLATES:
        return FEATURE_TEMPLATES[feature_name][direction]
    
    if feature_name.startswith("business_type_"):
        biz = feature_name.replace("business_type_", "").replace("_", " ").title()
        if direction == "high":
            return f"The {biz} sector has shown elevated default risk in the portfolio"
        else:
            return f"The {biz} sector has shown lower default risk in the portfolio"
            
    if feature_name.startswith("business_location_"):
        loc = feature_name.replace("business_location_", "")
        if direction == "high":
            return f"Borrower is located in {loc}, which has higher default rates historically"
        else:
            return f"Borrower is located in {loc}, which has lower default rates historically"
            
    readable = feature_name.replace("_", " ").title()
    verb = "increases default risk" if impact_value > 0 else "reduces default risk"
    return f"{readable} {verb}"

def run_prediction(payload):
    model_json, medians, feature_cols = get_model_assets()
    if not model_json:
        raise ValueError("Model assets not loaded")

    # Extract input fields
    loan_amount = float(payload.get("loan_amount", medians.get("loan_amount", 10000)))
    years_in_biz = float(payload.get("years_in_business", medians.get("years_in_business", 3.0)))
    mm_score = float(payload.get("mobile_money_activity_score", medians.get("mobile_money_activity_score", 50.0)))
    biz_type = str(payload.get("business_type", "retail_shop"))
    biz_loc = str(payload.get("business_location", "Nairobi"))
    loan_date_str = str(payload.get("loan_date", datetime.today().strftime("%Y-%m-%d")))

    # Feature Engineering
    try:
        loan_dt = datetime.strptime(loan_date_str, "%Y-%m-%d")
    except Exception:
        loan_dt = datetime.today()
    
    ref_dt = datetime(2020, 1, 1)
    loan_age_days = float((loan_dt - ref_dt).days)
    loan_amount_log = float(math.log1p(loan_amount))

    # Construct complete feature vector matching 25 columns
    features = {}
    for col in feature_cols:
        features[col] = 0.0

    features["loan_amount"] = loan_amount
    features["loan_amount_log"] = loan_amount_log
    features["years_in_business"] = years_in_biz
    features["mobile_money_activity_score"] = mm_score
    features["loan_age_days"] = loan_age_days

    # One-hot encoding
    bt_col = f"business_type_{biz_type}"
    if bt_col in features:
        features[bt_col] = 1.0

    loc_col = f"business_location_{biz_loc}"
    if loc_col in features:
        features[loc_col] = 1.0

    # Evaluate decision trees
    trees = model_json["learner"]["gradient_booster"]["model"]["trees"]
    feat_names = model_json["learner"]["feature_names"]
    
    total_margin = 0.0
    feature_contributions = {f: 0.0 for f in feat_names}

    for tree in trees:
        left = tree["left_children"]
        right = tree["right_children"]
        splits = tree["split_indices"]
        conds = tree["split_conditions"]
        weights = tree["base_weights"]
        def_left = tree["default_left"]

        node = 0
        while left[node] != -1:
            fid = splits[node]
            fname = feat_names[fid]
            val = features.get(fname, 0.0)
            parent_weight = weights[node]
            
            if math.isnan(val):
                next_node = left[node] if def_left[node] == 1 else right[node]
            elif val < conds[node]:
                next_node = left[node]
            else:
                next_node = right[node]
                
            child_weight = weights[next_node]
            feature_contributions[fname] += (child_weight - parent_weight)
            node = next_node
            
        total_margin += weights[node]

    # Convert margin to probability via logistic sigmoid
    prob = 1.0 / (1.0 + math.exp(-total_margin))
    risk_score = int(round(prob * 100))
    risk_score = max(0, min(100, risk_score))

    # Risk grading
    if risk_score <= 30:
        risk_grade = "Low"
        risk_class = "risk-low"
        risk_emoji = "🟢"
        risk_desc = "Highly manageable credit profile. Recommended for standard fast-tracked disbursement."
    elif risk_score <= 60:
        risk_grade = "Medium"
        risk_class = "risk-medium"
        risk_emoji = "🟡"
        risk_desc = "Elevated risk factors detected. Recommended for manual verification or secondary vetting."
    else:
        risk_grade = "High"
        risk_class = "risk-high"
        risk_emoji = "🔴"
        risk_desc = "High probability of default. Recommend rejecting application or requiring collateral/co-signers."

    # Sort feature impacts by magnitude
    sorted_impacts = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Generate Top 3 Reasons
    top3_reasons = []
    for fname, impact in sorted_impacts[:3]:
        reason_text = feature_to_english(fname, impact)
        is_pos = impact > 0
        top3_reasons.append({
            "feature": fname,
            "impact": round(impact, 4),
            "is_risk_aggregator": is_pos,
            "emoji": "⚠️" if is_pos else "🔮",
            "reason": reason_text
        })

    # Prepare top 7 impact breakdown for charts
    chart_impacts = [
        {"feature": f.replace("_", " ").title(), "impact": round(val, 4)}
        for f, val in sorted_impacts[:7]
    ]

    return {
        "status": "success",
        "borrower_id": payload.get("borrower_id", "IW-PREDICT"),
        "risk_score": risk_score,
        "probability": round(prob, 4),
        "risk_grade": risk_grade,
        "risk_class": risk_class,
        "risk_emoji": risk_emoji,
        "risk_desc": risk_desc,
        "top3_reasons": top3_reasons,
        "feature_impacts": chart_impacts
    }

class handler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        try:
            sample = {
                "loan_amount": 50000,
                "years_in_business": 4.5,
                "mobile_money_activity_score": 75,
                "business_type": "retail_shop",
                "business_location": "Nairobi"
            }
            res = run_prediction(sample)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as e:
            self._set_headers(500)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))

    def do_POST(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8")) if body else {}
            
            res = run_prediction(payload)
            self._set_headers(200)
            self.wfile.write(json.dumps(res).encode("utf-8"))
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({"status": "error", "message": str(e)}).encode("utf-8"))
