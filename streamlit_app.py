"""Student A - Asset Performance Index (Streamlit page).
Only calls my FastAPI service on Render. No model is loaded here."""
import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = "https://three6120-26sp-at3-group33-studenta-api-1.onrender.com"
TOKENS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "XLM", "TRX", "XMR"]
TIMEOUT = 70   # free Render services can take ~50 s to wake up


@st.cache_data(ttl=600, show_spinner=False)
def _get(method, endpoint, payload=None):
    if method == "GET":
        resp = requests.get(API_URL + endpoint, timeout=TIMEOUT)
    else:
        resp = requests.post(API_URL + endpoint, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def call_api(method, endpoint, payload=None):
    """Returns the API's answer, or None if it can't be reached (the page doesn't crash)."""
    try:
        return _get(method, endpoint, payload)
    except Exception:
        return None


def show_chart(chart):
    try:
        st.altair_chart(chart, width="stretch")
    except TypeError:
        st.altair_chart(chart, use_container_width=True)


def pct(value):
    return "n/a" if value is None else f"{value:.2%}"


st.set_page_config(page_title="Asset Performance Index", page_icon="📈", layout="wide")
st.title("📈 Asset Performance Index")
st.caption("Student A · Group 33 · XGBoost model · score predicted 7 days ahead")

token = st.selectbox("Choose a cryptocurrency", TOKENS)

# ---------------- prediction ----------------
with st.spinner("Getting the prediction (can take up to a minute if the API is waking up)..."):
    pred = call_api("POST", "/predict", {"token": token, "forecast_horizon": 7})

st.subheader("Performance Score")
if pred is None:
    st.error("The prediction service is unavailable right now. Please try again in a minute.")
else:
    score = pred["prediction"]
    level = "High" if score >= 70 else "Medium" if score >= 40 else "Low"
    c1, c2, c3 = st.columns(3)
    c1.metric("Performance Score (0-100)", f"{score:.1f}")
    c2.metric("Level", level)
    c3.metric("Forecast for", pred["prediction_date"])

    bands = pd.DataFrame({"start": [0, 40, 70], "end": [40, 70, 100], "band": ["Low", "Medium", "High"]})
    base = alt.Chart(bands).mark_bar(height=26).encode(
        x=alt.X("start:Q", scale=alt.Scale(domain=[0, 100]), title=None), x2="end:Q",
        color=alt.Color("band:N", scale=alt.Scale(domain=["Low", "Medium", "High"],
                                                    range=["#f3c9b8", "#e6e6e6", "#bcd7f5"]), legend=None))
    marker = alt.Chart(pd.DataFrame({"score": [score]})).mark_tick(color="#2a78d6", thickness=4, size=34).encode(x="score:Q")
    show_chart((base + marker).properties(height=60))
    st.caption(f"Based on market data up to {pred['as_of_date']} · source: {pred.get('data_source', '-')}")

    if pred.get("missing_features"):
        st.warning("Missing inputs from teammate services: " + ", ".join(pred["missing_features"])
                   + ". The model still predicts, but with lower accuracy.")

# ---------------- my engineered features ----------------
st.subheader("Student A features (latest completed day)")
feats = call_api("POST", "/features", {"token": token})
if feats is None:
    st.info("Features are unavailable right now.")
else:
    f = feats["features"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Daily Return", pct(f.get("DailyReturn")))
    c2.metric("5-Day Return", pct(f.get("Return5")))
    c3.metric("20-Day Return", pct(f.get("Return20")))
    momentum = f.get("Momentum")
    c4.metric("Momentum (USD)", "n/a" if momentum is None else f"{momentum:,.4g}")
    st.caption(f"As of {feats['date']} · source: {feats.get('source', '-')}")

# ---------------- feature importance ----------------
st.subheader("Feature importance")
meta = call_api("GET", "/model-metadata")
if meta is None:
    st.info("Feature importance is unavailable right now.")
else:
    fi = meta.get("feature_importance") or {}
    importance = pd.DataFrame({"feature": [str(k) for k in fi],
                               "importance": [float(v) for v in fi.values()]})
    chart = alt.Chart(importance).mark_bar(color="#2a78d6").encode(
        x=alt.X("importance:Q", title="share of importance"),
        y=alt.Y("feature:N", sort="-x", title=None),
        tooltip=["feature", alt.Tooltip("importance:Q", format=".2f")],
    )
    show_chart(chart)

st.divider()
st.caption("The score combines 20-day return (40%), trend (30%), trading volume (20%) and price stability (10%). "
           "It is a guide, not financial advice.")
