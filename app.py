import os
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from causalml.inference.meta import BaseSRegressor

st.set_page_config(page_title="Olist Uplift Engine", layout="wide")

# Header section
st.title("🚀 Olist Subscription Uplift Engine")
st.markdown("This interactive dashboard identifies **Persuadable** customers to maximize repeat purchases while minimizing marketing waste.")

def calculate_optimal_pct(df, model_col):
    # 1. Sort by model score
    temp_df = df.sort_values(model_col, ascending=False).reset_index(drop=True)
    
    # 2. Calculate Cumulative Gain
    temp_df['cum_gain'] = temp_df['outcome'].cumsum()
    
    # 3. Calculate Random Baseline Gain for each point
    total_outcomes = temp_df['outcome'].sum()
    temp_df['random_gain'] = (temp_df.index + 1) / len(temp_df) * total_outcomes
    
    # 4. Find the point of maximum "Lift" (Difference between model and random)
    temp_df['lift'] = temp_df['cum_gain'] - temp_df['random_gain']
    max_lift_idx = temp_df['lift'].idxmax()
    
    # 5. Return the percentage of the population at that point
    optimal_pct = (max_lift_idx / len(temp_df)) * 100
    return round(optimal_pct, 1)

def calculate_profit_optimal_pct(df, model_col, voucher_cost, avg_revenue, total_customers):
    """
    Returns the targeting percentage that maximizes profit, based on:
    profit(k) = incremental_purchases(k) * avg_revenue - targeted(k) * voucher_cost

    df is a *sample* (e.g., 2000 rows). We scale results to total_customers assuming the sample is representative.
    """
    temp_df = df.sort_values(model_col, ascending=False).reset_index(drop=True)

    # cumulative actual outcomes (on sample)
    temp_df["cum_gain"] = temp_df["outcome"].cumsum()

    # random baseline on sample
    total_outcomes = temp_df["outcome"].sum()
    temp_df["random_gain"] = (temp_df.index + 1) / len(temp_df) * total_outcomes

    # incremental purchases vs random (on sample)
    temp_df["inc_purchases_sample"] = temp_df["cum_gain"] - temp_df["random_gain"]

    # scale from sample size -> total customers
    scale = total_customers / len(temp_df)
    temp_df["inc_purchases_total"] = temp_df["inc_purchases_sample"] * scale

    # targeted customers at each cutoff (total scale)
    temp_df["targeted_total"] = (temp_df.index + 1) * scale

    # profit at each cutoff
    temp_df["profit"] = (temp_df["inc_purchases_total"] * avg_revenue) - (temp_df["targeted_total"] * voucher_cost)

    best_idx = temp_df["profit"].idxmax()

    profit_opt_pct = ((best_idx + 1) / len(temp_df)) * 100
    best_profit = temp_df.loc[best_idx, "profit"]

    return round(profit_opt_pct, 1), round(best_profit, 2)

# Load the saved model and features
@st.cache_resource
def load_data():
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, 'uplift_model.pkl')
    package = joblib.load(model_path)
    df = package['test_results']
    
    return package, df

# Update the assignment to include the dataframe
model, df_results = load_data()
population_default = int(model.get("population_customers", 96096))

# Business Logic Inputs
st.sidebar.header("Business Parameters")
voucher_cost = st.sidebar.number_input("Cost per Voucher (R$)", value=15.0)
avg_revenue = st.sidebar.number_input("Avg. Revenue per Repeat Purchase (R$)", value=120.0)

# The Targeting Slider
st.sidebar.markdown("---")
population_size = st.sidebar.number_input(
    "Population size (customers)", value=population_default, step=1000
)
target_pct = st.sidebar.slider("Percentage of Customers to Target", 0, 100, 30)

# Simulation Results (Logic)
total_customers = int(population_size)
targeted = int(total_customers * (target_pct / 100))
untargeted = total_customers - targeted

# Metrics Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Customers Targeted", f"{targeted:,}")
with col2:
    st.metric("Voucher Budget", f"R${targeted * voucher_cost:,.2f}")
with col3:
    st.metric("Waste Avoided", f"R${untargeted * voucher_cost:,.2f}", delta="Cost Saved")

st.subheader("Uplift Performance Comparison")

def plot_interactive_gain(df):
    fig = go.Figure()
    
    df.columns = [str(c) for c in df.columns]
    learners_to_plot = ['S-Learner', 'T-Learner', 'X-Learner']

    # Add the Random Baseline
    fig.add_trace(go.Scatter(
        x=[0, len(df)], y=[0, df['outcome'].sum()],
        mode='lines', name='Random Baseline',
        line=dict(color='white', dash='dash', width=2)
    ))

    # Loop through each model and add a curve
    for model_name in learners_to_plot:
        # Sort data specifically for THIS model
        temp_df = df.sort_values(model_name, ascending=False).reset_index(drop=True)
        cum_gain = temp_df['outcome'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df))), 
            y=cum_gain,
            mode='lines', 
            name=str(model_name).replace('_', '-').title(),
            line=dict(width=3)
        ))

    # Max-lift cutoff reference line (69.3% of population)
    cutoff_x = int(round(0.693 * len(df)))
    fig.add_vline(
        x=cutoff_x,
        line_width=2,
        line_dash="dot",
        line_color="white",
        annotation_text="Max-lift cutoff",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis_title="Number of Customers Targeted",
        yaxis_title="Cumulative Incremental Purchases",
        hovermode="x unified",
        template="plotly_dark",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99)
    )
    return fig

model, df_results = load_data()
st.markdown(
    "The model shows modest uplift overall, with the strongest signal concentrated in a very small "
    "customer segment. As a result, profit-maximizing targeting focuses on the top-ranked customers, "
    "even though the model remains statistically better than random across much of the population."
)
st.plotly_chart(plot_interactive_gain(df_results), width='stretch')

lift_opt_pct = calculate_optimal_pct(df_results, "S-Learner")

profit_opt_pct, best_profit = calculate_profit_optimal_pct(
    df_results, "S-Learner", voucher_cost, avg_revenue, total_customers
)

st.caption(
    f"Note: Profit optimization is estimated using a sample of {len(df_results):,} records "
    f"and scaled to a population of {int(population_size):,} records. "
    "This is a proxy for expected impact at scale."
)

st.info(f"Max-lift cutoff (vs random): {lift_opt_pct}%.")

profit_targeted = int(int(population_size) * (profit_opt_pct / 100))

st.success(
    f"At voucher cost R\\${voucher_cost:,.2f} and revenue R\\${avg_revenue:,.2f}, "
    f"profit is maximized by targeting {profit_opt_pct}% "
    f"(~{profit_targeted:,} customers). "
    f"Estimated profit: R\\${best_profit:,.2f}."
)

# Glossary
st.markdown("---")
st.subheader("Glossary")
st.markdown(
    """
**Business Concepts**
- **Persuadable customers**: People likely to make an additional purchase only if they receive a voucher.
- **Marketing waste**: Voucher spend on customers who would purchase anyway or would not purchase even with a voucher.
- **Treatment**: Receiving a voucher (the marketing action).
- **Outcome**: Whether a customer makes a repeat purchase (the target behavior).
- **Incremental purchase**: A purchase that happened because of the voucher, beyond what would have happened without it.
- **Voucher cost**: The cost to issue a single voucher.
- **Average revenue per repeat purchase**: The typical revenue from a repeat purchase.
- **Profit (estimated)**: (Incremental purchases × average revenue) − (targeted customers × voucher cost), scaled to the chosen population size.

**Model Concepts**
- **S-learner**: One model that uses features plus treatment to predict outcomes.
- **T-learner**: Two models, one for treated and one for untreated, then compares their predictions.
- **X-learner**: Uses separate treated/untreated models and reweights to better handle treatment imbalance.
- **Uplift**: The predicted increase in outcome probability caused by treatment.

**Charts & Metrics**
- **Gain curve**: Cumulative incremental outcomes when targeting customers from highest to lowest uplift.
- **Random baseline**: Expected cumulative outcomes if customers are targeted at random.
- **Targeting percentage / Top X%**: The share of customers chosen for vouchers, ranked by uplift.
- **Max-lift cutoff**: The targeting % where the uplift model beats the random baseline by the largest margin.
- **Profit-optimal cutoff**: The targeting % that maximizes estimated profit; this can differ from max-lift because costs and revenue scale differently with targeting.

**Assumptions & Limitations**
- **Sample-based estimation**: Metrics are computed on a sample of customers, not the full population.
- **Scaling to full potential**: We extrapolate sample results to the full population size, assuming the sample is representative.
"""
)

