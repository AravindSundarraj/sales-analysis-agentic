import pandas as pd
import numpy as np
import duckdb
import pyarrow as pa
np.random.seed(42)
assets = ["AAPL", "MSFT", "GOOGL", "AMZN",
    "TSLA", "XOM", "JPM", "BND",
    "GLD", "VNQ"]
weights = np.random.dirichlet(np.ones(len(assets)), size=1)[0]
print("weights:", weights)
portfolio_df = pd.DataFrame({
    "asset": assets,
    "weight": weights
})
parquet_file = "data/user_portfolio.parquet"
portfolio_df.to_parquet(parquet_file, index=False)
con = duckdb.connect(database=':memory:')
con.execute("""
    CREATE TABLE portfolio AS
    SELECT * FROM read_parquet('data/user_portfolio.parquet')
""")
result = con.execute("SELECT * FROM portfolio").fetchdf()
print('result:', result)
# Step 2 — Add Asset Classification Table
asset_metadata_df = pd.DataFrame({
    "asset": [
        "AAPL", "MSFT", "GOOGL", "AMZN",
        "TSLA", "XOM", "JPM", "BND",
        "GLD", "VNQ"
    ],
    "sector": [
        "Technology", "Technology", "Technology", "Consumer Discretionary",
        "Consumer Discretionary", "Energy", "Financials", "Fixed Income",
        "Commodity", "Real Estate"
    ],
    "asset_class": [
        "Equity", "Equity", "Equity", "Equity",
        "Equity", "Equity", "Equity", "Bond",
        "Commodity", "Equity"
    ]
})
# Ensure all string columns are explicitly converted to str
asset_metadata_df = asset_metadata_df.astype({
    "asset": str,
    "sector": str,
    "asset_class": str
})
table = pa.Table.from_pandas(asset_metadata_df)


# Register metadata into DuckDB
con.register("asset_metadata_df", table)

con.execute("""
    CREATE TABLE asset_metadata AS
    SELECT * FROM asset_metadata_df
""")

# ----------------------------
# Join Portfolio with Metadata
# ----------------------------

# | Stage             | Type of Validation | Judge Type |
# | ----------------- | ------------------ | ---------- |
# | Ticker existence  | Deterministic      | Rule-based |
# | Weights sum to 1  | Deterministic      | Rule-based |
# | Risk alignment    | Reasoning          | LLM Judge  |
# | Rebalancing logic | Reasoning          | LLM Judge  |
# | Financial safety  | Reasoning          | LLM Judge  |


joined_df = con.execute("""
    SELECT 
        p.asset,
        p.weight,
        m.sector,
        m.asset_class
    FROM portfolio p
    JOIN asset_metadata m
    ON p.asset = m.asset
""").fetchdf()

print("joined_df:", joined_df)
# Step 3 — Mathematical Portfolio Risk Calculation (Variance Model)
# Find out how risky your portfolio is — basically, “how much it could go up or down in value” based on your asset mix.


# ----------------------------
# Step 3A: Fetch Portfolio Weights
# ----------------------------

# Get the portfolio weights

# Look at your portfolio and see how much money is in each asset.

# Example: 40% AAPL, 30% MSFT, 30% bonds.

# This tells us how much each asset affects total risk.

# Get historical or simulated returns

# We imagine how each asset’s price moves every day.

# Since we don’t have real data, we simulate it (random numbers that behave like stock returns).

# This helps us see how the assets fluctuate individually.

# Compute correlations between assets

# Some assets move together (e.g., tech stocks), some move independently (e.g., bonds).

# We calculate how each asset’s ups and downs affect others.

# This is called the covariance matrix — basically a map of relationships between assets.

# Calculate total portfolio risk

# Combine weights and correlations to get one number: the portfolio’s overall volatility (risk).

# Bigger number = more risk, smaller number = less risk.

# We can also convert it to “yearly risk” so it’s easier to interpret.

# 🔹 1. Formula Used in Step 3

# The formula we use is called portfolio variance, which measures total portfolio risk:

# 𝜎
# 𝑝
# 2
# =
# 𝑤
# 𝑇
# Σ
# 𝑤
# σ
# p
# 2
# 	​

# =w
# T
# Σw

# Where:

# Symbol	Meaning

# 𝑤
# w	Weight vector of assets (how much money is in each asset)

# Σ
# Σ	Covariance matrix of asset returns (how assets move together)

# 𝜎
# 𝑝
# 2
# σ
# p
# 2
# 	​

# 	Portfolio variance (total risk squared)

# 𝜎
# 𝑝
# σ
# p
# 	​

# 	Portfolio volatility = √variance

# Step 3A: Get 
# 𝑤
# w (weights) from portfolio

# Step 3B: Simulate or fetch returns → calculate covariance matrix 
# Σ
# Σ

# Step 3C: Apply 
# 𝜎
# 𝑝
# 2
# =
# 𝑤
# 𝑇
# Σ
# 𝑤
# σ
# p
# 2
# 	​

# =w
# T
# Σw → total variance

# Step 3D: Take √variance → volatility (risk in same units as returns)

# Optional: Annualize volatility: 
# 𝜎
# annual
# =
# 𝜎
# daily
# ×
# 252
# σ
# annual
# 	​

# =σ
# daily
# 	​

# ×
# 252
# 	​


# 🔹 Why This Formula Works

# Each asset has its own variance (how volatile it is individually).

# Assets are correlated — some move together, some move oppositely.

# Total portfolio variance is weighted sum of all individual variances + covariances.

# In short:
# It’s the mathematically correct way to combine all asset risks into a single number for the portfolio.

weights_df = con.execute("""
    SELECT asset, weight
    FROM portfolio
    ORDER BY asset
""").fetchdf()

assets = weights_df["asset"].values   # list of tickers
weights = weights_df["weight"].values # numeric weight vector

print("Assets:", assets)
print("Weights:", weights)

# ----------------------------
# Step 3B: Simulate Historical Returns
# ----------------------------

np.random.seed(42)          # for reproducibility
num_assets = len(assets)
num_days = 252               # simulate 1 trading year

# simulate daily returns: mean=0.05% daily, std=2% daily
returns = np.random.normal(
    loc=0.0005, 
    scale=0.02, 
    size=(num_days, num_assets)
)

returns_df = pd.DataFrame(returns, columns=assets)
print("Simulated Returns (first 5 rows):")
print(returns_df.head())

# ----------------------------
# Step 3C: Compute Covariance Matrix
# ----------------------------

cov_matrix = returns_df.cov()
print("Covariance Matrix:")
print(cov_matrix)

# ----------------------------
# Step 3D: Compute Portfolio Variance & Volatility
# ----------------------------

portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
portfolio_volatility = np.sqrt(portfolio_variance)

# Optional: annualize volatility
annual_volatility = portfolio_volatility * np.sqrt(252)

print("Portfolio Variance:", portfolio_variance)
print("Portfolio Volatility (daily):", portfolio_volatility)
print("Portfolio Volatility (annualized):", annual_volatility)

# Step 4 — Compare Portfolio Risk vs User Risk Profile

# Goal of Step 4

# We now have:

# ✅ Portfolio annualized volatility (from Step 3)

# ❓ User risk tolerance (e.g., Conservative / Moderate / Aggressive)

# Step 4 answers:

# “Is this portfolio too risky, too safe, or aligned with the user?”

# ----------------------------
# Step 4 — Compare Risk vs User Profile
# ----------------------------
# Step 4 — System Determines Risk Profile (Not User)

# Instead of asking the user “Are you Moderate?”,
# we simulate user questionnaire inputs and let system classify.
# Example user inputs
investment_horizon_years = 12
income_stability = "High"
max_drawdown_tolerance = 20  # %

#Step 4B — Deterministic Risk Scoring
risk_score = 0

# Horizon scoring
if investment_horizon_years > 15:
    risk_score += 2
elif investment_horizon_years > 7:
    risk_score += 1

# Income stability
if income_stability == "High":
    risk_score += 2
elif income_stability == "Medium":
    risk_score += 1

# Drawdown tolerance
if max_drawdown_tolerance >= 30:
    risk_score += 2
elif max_drawdown_tolerance >= 15:
    risk_score += 1

# Step 4C — Map Score to Category

if risk_score <= 2:
    user_risk_profile = "Conservative"
elif risk_score <= 4:
    user_risk_profile = "Moderate"
else:
    user_risk_profile = "Aggressive"

print("System Assigned Risk Profile:", user_risk_profile)



# Risk thresholds (annualized volatility)
risk_ranges = {
    "Conservative": (0.00, 0.10),
    "Moderate": (0.10, 0.20),
    "Aggressive": (0.20, 0.35)
}

portfolio_vol = annual_volatility

low, high = risk_ranges[user_risk_profile]

print("User Risk Profile:", user_risk_profile)
print("Portfolio Annual Volatility:", portfolio_vol)

if portfolio_vol < low:
    risk_status = "Portfolio is TOO SAFE for the selected risk profile."
elif portfolio_vol > high:
    risk_status = "Portfolio is TOO RISKY for the selected risk profile."
else:
    risk_status = "Portfolio risk is ALIGNED with the selected profile."

print("Risk Evaluation:", risk_status)
