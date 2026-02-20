import pandas as pd
import numpy as np
import duckdb
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
print(result)