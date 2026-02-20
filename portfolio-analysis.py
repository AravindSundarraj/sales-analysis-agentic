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
