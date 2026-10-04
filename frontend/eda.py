"""Online data access for the EDA dashboard: downloads the SQLite DB from Google Drive on first use."""
import os
import sqlite3
from pathlib import Path

import pandas as pd

DRIVE_ID = os.getenv("DRIVE_FILE_ID", "1M2sonY7serOCzWYDCKEdxZ5fJ7quOoKd")
DB_PATH = Path(os.getenv("DB_PATH", Path(__file__).resolve().parent / "data" / "superstore.db"))

QUERY = """
SELECT oi.sales, oi.profit, oi.discount, o.order_date, c.segment, p.category, l.market
FROM order_items oi
JOIN orders o        ON oi.order_key  = o.order_key
JOIN dim_customers c ON o.customer_id = c.customer_id
JOIN dim_products p  ON oi.product_id = p.product_id
JOIN dim_locations l ON o.location_id = l.location_id
"""
_cache = None


def ensure_db() -> Path:
    if not DB_PATH.exists():
        import gdown
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=DRIVE_ID, output=str(DB_PATH), quiet=True)
    return DB_PATH


def overview() -> dict:
    global _cache
    if _cache is None:
        with sqlite3.connect(ensure_db()) as con:
            df = pd.read_sql(QUERY, con)
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
        df["discount_tier"] = pd.cut(df["discount"], [-0.01, 0, 0.2, 0.4, 1.0],
                                     labels=["0%", "1-20%", "21-40%", ">40%"]).astype(str)
        month = df.groupby(df["order_date"].dt.to_period("M").astype(str))["sales"].sum()
        _cache = {
            "n_rows": int(len(df)),
            "profit_by_category": df.groupby("category")["profit"].sum().round(2).to_dict(),
            "profit_by_market": df.groupby("market")["profit"].sum().round(2).to_dict(),
            "profit_by_segment": df.groupby("segment")["profit"].sum().round(2).to_dict(),
            "avg_profit_by_discount_tier": df.groupby("discount_tier")["profit"].mean().round(2).to_dict(),
            "monthly_sales": month.round(2).to_dict(),
        }
    return _cache
