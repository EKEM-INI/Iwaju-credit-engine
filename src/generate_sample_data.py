"""
generate_sample_data.py
-----------------------
Creates a realistic sample CSV of historical loan data for Pezesha.
Run this ONCE to create your training dataset.
Usage: python src/generate_sample_data.py
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

random.seed(42)

# Resolve data directory relative to this script
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)
np.random.seed(42)

N = 2000  # number of borrowers to generate

BUSINESS_TYPES = [
    "retail_shop", "food_vendor", "salon_barber", "transport",
    "tailoring", "electronics_repair", "agro_dealer", "pharmacy",
    "hardware_store", "mobile_money_agent"
]

LOCATIONS = [
    "Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret",
    "Thika", "Malindi", "Kitale", "Garissa", "Nyeri"
]

def random_date(start_year=2020, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

rows = []
for i in range(N):
    years_in_biz = round(np.random.exponential(3) + 0.5, 1)
    years_in_biz = min(years_in_biz, 20)

    mobile_score = round(np.random.beta(2, 2) * 100, 1)
    loan_amount = round(np.random.lognormal(mean=9, sigma=1), 2)  # KES
    loan_amount = max(500, min(loan_amount, 500000))

    biz_type = random.choice(BUSINESS_TYPES)
    location = random.choice(LOCATIONS)

    # Inject some missing values (5% chance each field)
    if random.random() < 0.05:
        years_in_biz = np.nan
    if random.random() < 0.05:
        mobile_score = np.nan
    if random.random() < 0.03:
        biz_type = np.nan

    # Default probability — influenced by risk factors
    base_p = 0.20
    if not np.isnan(mobile_score if not isinstance(mobile_score, float) else mobile_score):
        base_p -= (mobile_score / 100) * 0.12
    if not np.isnan(years_in_biz if not isinstance(years_in_biz, float) else years_in_biz):
        base_p -= min(years_in_biz / 10, 0.08)
    if loan_amount > 50000:
        base_p += 0.05
    base_p = max(0.02, min(base_p, 0.70))

    outcome_roll = random.random()
    if outcome_roll < base_p:
        repayment = "defaulted"
    elif outcome_roll < base_p + 0.25:
        repayment = "late"
    else:
        repayment = "on_time"

    rows.append({
        "borrower_id": f"PEZ{10000 + i}",
        "loan_amount": loan_amount,
        "loan_date": random_date().strftime("%Y-%m-%d"),
        "business_type": biz_type,
        "business_location": location,
        "years_in_business": years_in_biz,
        "repayment_history": repayment,
        "mobile_money_activity_score": mobile_score,
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(DATA_DIR, "historical_loans.csv"), index=False)
print(f"✅  Sample data saved → data/historical_loans.csv  ({N} rows)")
print(df["repayment_history"].value_counts())
