# 🔮 IWAJU — AI Credit Default Risk Engine

An AI-powered credit scoring, default prediction, and explainable underwriting system built for African digital lending platforms and MSMEs. 

This repository is **100% optimized and pre-configured for instant deployment on Vercel** as well as local development with Streamlit.

---

## ⚡ 1-Minute Live Deployment on Vercel

This repository is ready to deploy directly to [Vercel](https://vercel.com/):

1. Push this repository to your GitHub account (or fork it).
2. Log in to [vercel.com](https://vercel.com/) and click **"Add New..."** ➔ **"Project"**.
3. Select your repository (`Iwaju-credit-engine`).
4. Keep the default settings:
   - **Framework Preset**: *Other*
   - **Root Directory**: `./`
5. Click **"Deploy"**!

Vercel will immediately deploy:
- 🌐 **Live Dashboard**: Interactive borrower credit risk assessment interface at your Vercel URL.
- 🔮 **Visual Pitch Deck**: Accessible directly at `https://<your-project>.vercel.app/pitch`.
- ⚡ **Serverless API**: Real-time evaluation endpoint at `https://<your-project>.vercel.app/api/predict`.

---

## 🏗️ Project Architecture

```text
Iwaju-credit-engine/
│
├── index.html                  # Responsive modern web dashboard (Vercel frontend)
├── pitch.html                  # Interactive visual pitch deck (black & purple)
├── vercel.json                 # Vercel serverless routing configuration
├── requirements.txt            # Zero-overhead serverless Python dependencies
├── requirements-dev.txt        # Local data science dependencies (XGBoost, SHAP, Streamlit)
│
├── api/
│   └── predict.py              # Vercel Serverless Function (pure Python inference engine)
│
├── models/
│   ├── xgb_model.json          # Exported XGBoost decision tree structure
│   ├── xgb_model.pkl           # Pickled XGBClassifier model
│   ├── fill_medians.json       # Feature medians learned at training time
│   ├── feature_columns.json    # Exact feature vector schema (25 columns)
│   └── shap_explainer.pkl      # TreeExplainer artifact
│
├── data/
│   └── historical_loans.csv    # 2,000 African MSME historical loan records
│
└── src/
    ├── app.py                  # Streamlit web dashboard (for local use)
    ├── train_model.py          # Model training and evaluation script
    ├── preprocess.py           # Feature engineering and cleaning pipeline
    ├── explain.py              # SHAP plain-English explanation generator
    └── generate_sample_data.py # Sample loan generator
```

---

## 🔍 How the Vercel Serverless Inference Works

Standard machine learning packages (`xgboost`, `scipy`, `scikit-learn`, `shap`) often exceed Vercel's **250MB Serverless Function size limit** and suffer from slow 5–10s cold starts.

**IWAJU solves this with an ultra-lean tree inference engine (`api/predict.py`)**:
- Uses the standard library (`json`, `math`, `http.server`) with **zero heavy binary dependencies**.
- Loads pre-compiled decision trees from `models/xgb_model.json`.
- Evaluates tree paths and calculates exact feature contributions in **< 1 millisecond**.
- Translates top 3 contributing factors into natural, human-readable plain English reasons (e.g., *"Strong mobile money activity suggests reliable cash flow"*).
- Instant cold start (< 10ms) on Vercel Edge/Serverless.

---

## 💻 Running Locally

### Option A: Local Web Dashboard (HTML/API)
Simply open `index.html` in your browser, or run a local web server:
```bash
python -m http.server 3000
```
Visit `http://localhost:3000` for the dashboard and `http://localhost:3000/pitch.html` for the pitch deck.

### Option B: Local Streamlit Dashboard
If you prefer running the original Streamlit interface locally:
```bash
# Install local dev dependencies
pip install -r requirements-dev.txt

# Launch Streamlit
streamlit run src/app.py
```

### Option C: Retrain the XGBoost Model
```bash
python src/train_model.py
```
