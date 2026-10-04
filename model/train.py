"""Training script (same logic as the notebook). Run from repo root:
    python model/train.py notebook/superstore.db
Outputs: model/classifier.joblib, model/clusterer.joblib, model/meta.json
"""
import json, sqlite3, sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

OUT = Path(__file__).parent
DB = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT.parent / "notebook" / "superstore.db"

QUERY = """
SELECT oi.row_id, oi.order_key, oi.sales, oi.quantity, oi.discount, oi.profit, oi.shipping_cost,
       o.customer_id, o.order_date, o.ship_date, o.order_priority, o.ship_mode,
       c.segment, p.category, p.sub_category, l.region, l.market, l.country
FROM order_items oi
JOIN orders o         ON oi.order_key = o.order_key
JOIN dim_customers c  ON o.customer_id = c.customer_id
JOIN dim_products p   ON oi.product_id = p.product_id
JOIN dim_locations l  ON o.location_id = l.location_id
"""
NUM = ["sales", "quantity", "discount", "shipping_cost", "delivery_days", "sales_per_unit"]
CAT = ["ship_mode", "order_priority", "segment", "category", "sub_category", "region", "market"]
CLUSTER_FEATS = ["total_sales", "n_orders", "avg_discount", "profit_margin"]


DRIVE_ID = "1M2sonY7serOCzWYDCKEdxZ5fJ7quOoKd"


def load():
    if not DB.exists():  # fetch the published database from Google Drive
        import gdown
        DB.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=DRIVE_ID, output=str(DB), quiet=False)
    with sqlite3.connect(DB) as con:
        df = pd.read_sql(QUERY, con)
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")
    df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce")
    df["delivery_days"] = (df["ship_date"] - df["order_date"]).dt.days
    df["sales_per_unit"] = df["sales"] / df["quantity"].replace(0, np.nan)
    df = df.dropna(subset=NUM + CAT).drop_duplicates("row_id")
    df["is_profitable"] = (df["profit"] > 0).astype(int)
    return df


def train_classifier(df):
    X, y = df[NUM + CAT], df["is_profitable"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=42))])
    pipe.fit(Xtr, ytr)
    pred, proba = pipe.predict(Xte), pipe.predict_proba(Xte)[:, 1]
    metrics = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                   recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                   roc_auc=roc_auc_score(yte, proba))
    return pipe, {k: round(float(v), 4) for k, v in metrics.items()}


def build_customers(df):
    g = df.groupby("customer_id").agg(total_sales=("sales", "sum"), total_profit=("profit", "sum"),
                                      n_orders=("order_key", "nunique"), avg_discount=("discount", "mean"))
    g["profit_margin"] = g["total_profit"] / g["total_sales"]
    return g.replace([np.inf, -np.inf], np.nan).dropna()


def name_clusters(cust, labels):
    cust = cust.assign(cluster=labels)
    prof, mean = cust.groupby("cluster")[CLUSTER_FEATS].mean(), cust[CLUSTER_FEATS].mean()
    names = {}
    for c, r in prof.iterrows():
        n = f"C{c}: {'High' if r.total_sales > mean.total_sales else 'Low'}-Spend, "
        n += "Discount-Driven" if r.avg_discount > mean.avg_discount else "Full-Price"
        if r.profit_margin < 0:
            n += " (Loss-Making)"
        names[int(c)] = n
    return names, prof


def train_clusterer(cust):
    X = cust[CLUSTER_FEATS]
    scores = {}
    for k in range(2, 9):
        pipe = Pipeline([("scale", PowerTransformer()), ("km", KMeans(k, n_init=10, random_state=42))])
        lab = pipe.fit_predict(X)
        scores[k] = silhouette_score(pipe[:-1].transform(X), lab)
    best_k = max(scores, key=scores.get)
    pipe = Pipeline([("scale", PowerTransformer()), ("km", KMeans(best_k, n_init=10, random_state=42))])
    labels = pipe.fit_predict(X)
    names, prof = name_clusters(cust, labels)
    return pipe, names, prof, best_k, {k: round(float(v), 4) for k, v in scores.items()}


if __name__ == "__main__":
    df = load()
    print("Rows:", len(df))
    clf, clf_metrics = train_classifier(df)
    print("Classification:", clf_metrics)
    cust = build_customers(df)
    km, names, prof, best_k, sil = train_clusterer(cust)
    print("Silhouette per k:", sil, "-> best k =", best_k)
    print(prof.round(3))
    joblib.dump(clf, OUT / "classifier.joblib")
    joblib.dump({"pipeline": km, "names": names, "features": CLUSTER_FEATS}, OUT / "clusterer.joblib")
    meta = dict(num_features=NUM, cat_features=CAT, cluster_features=CLUSTER_FEATS,
                categories={c: sorted(df[c].astype(str).unique().tolist()) for c in CAT},
                classification_metrics=clf_metrics, cluster_names=names, best_k=best_k,
                silhouette_by_k=sil)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Saved models + meta.json")
