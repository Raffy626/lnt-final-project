import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from pydantic import BaseModel, Field

import eda

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
clf = joblib.load(MODEL_DIR / "classifier.joblib")
clu = joblib.load(MODEL_DIR / "clusterer.joblib")
meta = json.loads((MODEL_DIR / "meta.json").read_text())

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
    return {"categories": meta["categories"], "classification_metrics": meta["classification_metrics"],
            "cluster_names": meta["cluster_names"]}


@app.post("/predict/classification")
def predict_profit(x: ProfitInput):
    row = x.model_dump()
    row["sales_per_unit"] = row["sales"] / row["quantity"]
    df = pd.DataFrame([row])[meta["num_features"] + meta["cat_features"]]
    proba = float(clf.predict_proba(df)[0, 1])
    return {"is_profitable": proba >= 0.5, "probability_profitable": round(proba, 4)}


@app.post("/predict/clustering")
def predict_cluster(x: CustomerInput):
    df = pd.DataFrame([x.model_dump()])[clu["features"]]
    c = int(clu["pipeline"].predict(df)[0])
    return {"cluster": c, "cluster_label": clu["names"][c]}


@app.get("/eda/overview")
def eda_overview():
    try:
        return eda.overview()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"EDA data unavailable: {e}")
