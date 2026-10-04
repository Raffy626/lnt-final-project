import os

import pandas as pd
import requests
import streamlit as st

try:
    BACKEND = os.getenv("BACKEND_URL") or st.secrets["BACKEND_URL"]
except Exception:
    BACKEND = "http://localhost:8000"

st.set_page_config(page_title="LnT Camp 2026 - Superstore ML", layout="centered")
st.title("Superstore ML Simulator")
st.caption("Bridging the Gap: Empowering Future Talent through Machine Learning for Industry Innovation")


@st.cache_data(ttl=300)
def get_meta():
    return requests.get(f"{BACKEND}/meta", timeout=60).json()


try:
    meta = get_meta()
except Exception as e:
    st.error(f"Backend tidak bisa dihubungi di {BACKEND}: {e}")
    st.stop()

cats = meta["categories"]
page = st.sidebar.radio("Pilih model", ["Klasifikasi: Order Profitable?", "Klustering: Segmen Customer", "EDA Dashboard"])

if page.startswith("Klasifikasi"):
    st.subheader("Prediksi apakah item order akan menguntungkan")
    with st.form("clf"):
        c1, c2 = st.columns(2)
        sales = c1.number_input("Sales ($)", min_value=0.01, value=250.0)
        quantity = c2.number_input("Quantity", min_value=1, value=3, step=1)
        discount = c1.slider("Discount", 0.0, 1.0, 0.1, 0.05)
        shipping_cost = c2.number_input("Shipping cost ($)", min_value=0.0, value=20.0)
        delivery_days = c1.number_input("Delivery days", min_value=0, value=4, step=1)
        ship_mode = c2.selectbox("Ship mode", cats["ship_mode"])
        order_priority = c1.selectbox("Order priority", cats["order_priority"])
        segment = c2.selectbox("Customer segment", cats["segment"])
        category = c1.selectbox("Category", cats["category"])
        sub_category = c2.selectbox("Sub-category", cats["sub_category"])
        region = c1.selectbox("Region", cats["region"])
        market = c2.selectbox("Market", cats["market"])
        go = st.form_submit_button("Prediksi")
    if go:
        payload = dict(sales=sales, quantity=int(quantity), discount=discount, shipping_cost=shipping_cost,
                       delivery_days=int(delivery_days), ship_mode=ship_mode, order_priority=order_priority,
                       segment=segment, category=category, sub_category=sub_category, region=region, market=market)
        r = requests.post(f"{BACKEND}/predict/classification", json=payload, timeout=60)
        if r.ok:
            res = r.json()
            (st.success if res["is_profitable"] else st.error)(
                "Diprediksi PROFITABLE" if res["is_profitable"] else "Diprediksi TIDAK profitable")
            st.metric("Probabilitas profitable", f"{res['probability_profitable']:.1%}")
        else:
            st.error(r.text)
elif page == "EDA Dashboard":
    st.subheader("Ringkasan data Global Superstore")
    r = requests.get(f"{BACKEND}/eda/overview", timeout=180)
    if not r.ok:
        st.error(r.text)
    else:
        d = r.json()
        st.metric("Jumlah order item", f"{d['n_rows']:,}")
        st.write("**Total profit per kategori**"); st.bar_chart(pd.Series(d["profit_by_category"]))
        st.write("**Total profit per market**"); st.bar_chart(pd.Series(d["profit_by_market"]))
        st.write("**Rata-rata profit per tier diskon**")
        st.bar_chart(pd.Series(d["avg_profit_by_discount_tier"]).reindex(["0%", "1-20%", "21-40%", ">40%"]))
        st.write("**Sales bulanan**"); st.line_chart(pd.Series(d["monthly_sales"]))
else:
    st.subheader("Tentukan segmen customer")
    with st.form("clu"):
        total_sales = st.number_input("Total sales ($)", min_value=0.01, value=5000.0)
        n_orders = st.number_input("Jumlah order", min_value=1, value=8, step=1)
        avg_discount = st.slider("Rata-rata discount", 0.0, 1.0, 0.12, 0.01)
        profit_margin = st.number_input("Profit margin (total profit / total sales)", value=0.10, format="%.3f")
        go = st.form_submit_button("Tentukan segmen")
    if go:
        r = requests.post(f"{BACKEND}/predict/clustering", json=dict(
            total_sales=total_sales, n_orders=int(n_orders), avg_discount=avg_discount,
            profit_margin=profit_margin), timeout=60)
        if r.ok:
            st.success(f"Segmen: {r.json()['cluster_label']}")
        else:
            st.error(r.text)
