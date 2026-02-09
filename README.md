# Olist Subscription Uplift Engine

An interactive Streamlit dashboard that helps identify **persuadable customers** (customers who are more likely to make a repeat purchase *because* they received a voucher), so marketing spend is focused where it actually creates incremental impact.

## Live Demo
[[Olist Uplift Engine Dashboard](https://olist-uplift-engine.streamlit.app/)]

## What this does
This project uses uplift modeling to compare multiple meta-learners and supports a simple business simulation.

## Tech Stack
Python, Streamlit, Plotly, CausalML, XGBoost, pandas, scikit-learn

**Included learners**
- S-Learner
- T-Learner
- X-Learner

**Dashboard features**
- Business inputs: voucher cost and average revenue per repeat purchase
- Targeting slider: choose what % of customers to target
- Metrics: customers targeted, voucher budget, waste avoided
- Gain chart: compares cumulative incremental purchases vs a random baseline
- Calculates an “optimal” targeting % using maximum lift vs random baseline

## Repository contents
- `app.py`  
  Streamlit dashboard that loads a packaged model + test sample (`uplift_model.pkl`) and displays interactive results.

- `main.py`  
  Training / packaging script. Produces:
  - `uplift_model.pkl` (model + feature list + sampled test results)
  - `uplift_comparison_results.png` (saved comparison plot)

- `requirements.txt`  
  Python dependencies for running the app.

## Run locally

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

