import pandas as pd
from pathlib import Path
from causalml.inference.meta import BaseSRegressor, BaseTRegressor, BaseXRegressor
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from causalml.metrics import plot_gain
import joblib
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# Set the data folder
data_folder = Path(r"C:\Users\hanna\.cache\kagglehub\datasets\olistbr\brazilian-ecommerce\versions\2")

# Load the data
orders = pd.read_csv(data_folder / "olist_orders_dataset.csv")
payments = pd.read_csv(data_folder / "olist_order_payments_dataset.csv")
products = pd.read_csv(data_folder / "olist_products_dataset.csv")
customers = pd.read_csv(data_folder / "olist_customers_dataset.csv")
order_items = pd.read_csv(data_folder / "olist_order_items_dataset.csv")

# Merge the data
merged_df = orders.merge(payments, on="order_id", how="left")
merged_df = merged_df.merge(order_items, on="order_id", how="left")
merged_df = merged_df.merge(products, on="product_id", how="left")
merged_df = merged_df.merge(customers, on="customer_id", how="left")

# Fill missing values
merged_df["product_category_name"] = merged_df["product_category_name"].fillna("Unknown")
merged_df["customer_state"] = merged_df["customer_state"].fillna("Unknown")
merged_df["product_weight_g"] = merged_df["product_weight_g"].fillna(0)

# Create the treatment variable
merged_df['is_treated'] = 0
merged_df.loc[merged_df["payment_type"] == "voucher", 'is_treated'] = 1

# Check for missing values
# merged_df.isnull().sum()

# Customer-level columns
merged_df["total_orders"] = merged_df.groupby("customer_unique_id")["order_id"].transform("nunique")

# Define Y
merged_df['y'] = (merged_df['total_orders'] > 1).astype(int)

# Build customer-level table
customer_df = (
    merged_df.groupby("customer_unique_id", as_index=False)
    .agg(
        is_treated=("is_treated", "max"), #if voucher was ever used - treated
        y=("y", "max"), #if a repeat purchase was ever made
        customer_state=("customer_state", "first"), #state doesn't change
        product_weight_g=("product_weight_g", "mean"), #to summarize weights
        product_category_name=("product_category_name", lambda s: s.mode().iloc[0] if not s.mode().empty else "Unknown"),
    )
)
""" print(f"Customer table shape: {customer_df.shape}")
print(f"Unique customers: {customer_df['customer_unique_id'].nunique()}") """

# Encode the data (customer-level)
product_category_name = pd.get_dummies(customer_df["product_category_name"])
location = pd.get_dummies(customer_df["customer_state"])

# Combine the dummies and numerical variables - Define X (customer-level)
X = pd.concat([customer_df[["product_weight_g"]], product_category_name, location], axis=1)
X = X.astype(float)
assert X.isnull().sum().sum() == 0

# Define T (customer-level)
T = customer_df['is_treated']
T = T.fillna(0).astype(int)

# Define Y (customer-level)
Y = customer_df['y']
Y = Y.fillna(0).astype(int)

# Temporary sampling to make the code run faster
# X, T, Y = X.sample(5000, random_state=42), T.sample(5000, random_state=42), Y.sample(5000, random_state=42)

# Define the XGBoost learner
base_algo = XGBRegressor(n_estimators=100, max_depth=5, tree_method = 'gpu_hist', random_state=42, device='cuda')

# Define learners
learners = {
    's_learner': BaseSRegressor(learner=base_algo),
    't_learner': BaseTRegressor(learner=base_algo),
    'x_learner': BaseXRegressor(learner=base_algo)
}

# Split the data into training and testing sets
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=42, stratify=T)

# Get the uplift scores
results = {}
for name, learner in learners.items():
    learner.fit(X=X_train, treatment=T_train, y=Y_train)
    results[name] = learner.predict(X=X_test).flatten()

# Create a dataframe for the plot gain function
df_preds = pd.DataFrame(results)
df_preds['treatment'] = T_test.values
df_preds['outcome'] = Y_test.values

plot_gain(df_preds, outcome_col='outcome', treatment_col='treatment')

plt.title('Comparison of Uplift Score Distributions')
plt.xlabel('Predicted Uplift (Probability Increase)')
plt.ylabel('Density')
plt.legend(title='Learners')
plt.savefig('uplift_comparison_results.png', dpi=300)
plt.show()

# Dashboard 
dashboard_data = pd.DataFrame({
    'S-Learner': results['s_learner'],
    'T-Learner': results['t_learner'],
    'X-Learner': results['x_learner'],
    'treatment': T_test.values,
    'outcome': Y_test.values
}).sample(2000, random_state=42) 

export_package = {
    'model': learners['s_learner'],
    'features': X.columns.tolist(),
    'test_results': dashboard_data,
    "population_customers": int(customer_df.shape[0])
}

joblib.dump(export_package, 'uplift_model.pkl')
print("Model and test results successfully packaged into uplift_model.pkl")

""" print(f"Treatment ratio in Train: {T_train.mean():.2%}")
print(f"Treatment ratio in Test:  {T_test.mean():.2%}") """
