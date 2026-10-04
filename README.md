# LnT Camp 2026 — Final Project
**Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation**

Dataset: Global Superstore (SQLite, normalised). Tim: *(isi nama anggota)*

## Chosen modelling tasks
1. **Classification** — predict whether an order item is profitable (`profit > 0`). Random Forest.
2. **Clustering** — customer segmentation (K-Means on total sales, #orders, avg discount, profit margin).

## Folder structure
```
notebook/   Jupyter notebook (EDA -> preprocessing -> 2 models). Put superstore.db here (not committed)
model/      train.py + saved classifier.joblib, clusterer.joblib, meta.json
backend/    FastAPI app (main.py) + requirements.txt
frontend/   Streamlit simulator (app.py) + requirements.txt
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt -r frontend/requirements.txt jupyter matplotlib seaborn gdown
```
1. The SQLite DB is published online (Google Drive) and is downloaded automatically (via `gdown`) by the notebook, `model/train.py`, and the backend's EDA endpoint. Manual copy at `notebook/superstore.db` also works.
2. Train: run the notebook top-to-bottom (it writes `model/*`), **or** `python model/train.py notebook/superstore.db`.

### Run backend
```bash
cd backend && uvicorn main:app --reload --port 8000
```
Docs: http://localhost:8000/docs

| Method | Endpoint | Input | Output |
|---|---|---|---|
| GET | `/health` | - | `{"status":"ok"}` |
| GET | `/meta` | - | category options, classifier metrics, cluster names |
| GET | `/eda/overview` | - | aggregated stats for the EDA dashboard (downloads the DB on first call) |
| POST | `/predict/classification` | `sales, quantity, discount, shipping_cost, delivery_days, ship_mode, order_priority, segment, category, sub_category, region, market` | `{"is_profitable": bool, "probability_profitable": float}` |
| POST | `/predict/clustering` | `total_sales, n_orders, avg_discount, profit_margin` | `{"cluster": int, "cluster_label": str}` |

### Run frontend
```bash
cd frontend && BACKEND_URL=http://localhost:8000 streamlit run app.py
```

## Deployment
- **Backend** (Render/Railway): root dir = repo root; build `pip install -r backend/requirements.txt`; start `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`. Make sure `model/*.joblib` and `model/meta.json` are committed.
- **Frontend** (Streamlit Community Cloud): main file `frontend/app.py`; set secret `BACKEND_URL = "https://<your-backend-url>"`.
