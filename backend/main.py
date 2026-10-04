import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import eda

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
if not (MODEL_DIR / "classifier.joblib").exists():
    MODEL_DIR = Path(__file__).resolve().parent / "model"

_clf = None
_clu = None
_meta = None

def get_models():
    global _clf, _clu, _meta
    if _clf is None:
        clf_path = MODEL_DIR / "classifier.joblib"
        if not clf_path.exists():
            db_path = eda.ensure_db()
            subprocess.run([sys.executable, str(Path(__file__).resolve().parent.parent / "model" / "train.py"), str(db_path)], check=True)
        _clf = joblib.load(MODEL_DIR / "classifier.joblib")
        _clu = joblib.load(MODEL_DIR / "clusterer.joblib")
        _meta = json.loads((MODEL_DIR / "meta.json").read_text())
    return _clf, _clu, _meta

app = FastAPI(title="LnT Camp 2026 - Superstore ML API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])



class ProfitInput(BaseModel):
    sales: float = Field(gt=0, examples=[250.0])
    quantity: int = Field(gt=0, examples=[3])
    discount: float = Field(ge=0, le=1, examples=[0.1])
    shipping_cost: float = Field(ge=0, examples=[20.0])
    delivery_days: int = Field(ge=0, examples=[4])
    ship_mode: str
    order_priority: str
    segment: str
    category: str
    sub_category: str
    region: str
    market: str


class CustomerInput(BaseModel):
    total_sales: float = Field(gt=0, examples=[5000.0])
    n_orders: int = Field(gt=0, examples=[8])
    avg_discount: float = Field(ge=0, le=1, examples=[0.12])
    profit_margin: float = Field(examples=[0.1], description="total_profit / total_sales")


@app.get("/")
def root():
    return {"message": "LnT Camp 2026 - Superstore ML API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}



@app.get("/meta")
def get_meta():
    _, _, meta = get_models()
    return {"categories": meta["categories"], "classification_metrics": meta["classification_metrics"],
            "cluster_names": meta["cluster_names"]}


@app.post("/predict/classification")
def predict_profit(x: ProfitInput):
    clf, _, meta = get_models()
    row = x.model_dump()
    row["sales_per_unit"] = row["sales"] / row["quantity"]
    df = pd.DataFrame([row])[meta["num_features"] + meta["cat_features"]]
    proba = float(clf.predict_proba(df)[0, 1])
    return {"is_profitable": proba >= 0.5, "probability_profitable": round(proba, 4)}


@app.post("/predict/clustering")
def predict_cluster(x: CustomerInput):
    _, clu, _ = get_models()
    df = pd.DataFrame([x.model_dump()])[clu["features"]]
    c = int(clu["pipeline"].predict(df)[0])
    return {"cluster": c, "cluster_label": clu["names"][c]}



@app.get("/eda/overview")
def eda_overview():
    try:
        return eda.overview()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"EDA data unavailable: {e}")
