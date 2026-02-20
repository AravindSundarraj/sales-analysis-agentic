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
