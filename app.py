import os
import joblib
import pandas as pd
from train import train_model
import streamlit as st

st.set_page_config(
    page_title="AutoPrice AI",
    page_icon="🚗",
    layout="wide"
)

DATA_PATH = "data/Automobile_data.csv"
MODEL_PATH = "model/car_price_pipeline.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("First run: training the AI model automatically..."):
            result = train_model()
        st.success(f"Model trained automatically — R²: {result['r2']:.4f}")
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).replace("?", pd.NA)
    return df

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2rem;}
.hero {padding: 1.5rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.25);}
.price {font-size: 2.4rem; font-weight: 800;}
</style>
""", unsafe_allow_html=True)

st.title("🚗 AutoPrice AI")
st.caption("Machine Learning web app for automobile price prediction")

# The model is trained automatically on first launch if it does not exist.
model = load_model()
df = load_data()

st.markdown("### Enter automobile specifications")

# Keep the exact feature names from the user's dataset.
features = [c for c in df.columns if c != "price"]
input_data = {}

# Numeric-looking fields
numeric_candidates = [
    "symboling","normalized-losses","wheel-base","length","width","height",
    "curb-weight","engine-size","bore","stroke","compression-ratio",
    "horsepower","peak-rpm","city-mpg","highway-mpg"
]

left, right = st.columns(2)

for i, col in enumerate(features):
    container = left if i % 2 == 0 else right
    with container:
        series = pd.to_numeric(df[col], errors="coerce") if col in numeric_candidates else None
        if series is not None and series.notna().any():
            default = float(series.median())
            input_data[col] = st.number_input(
                col.replace("-", " ").title(),
                value=default
            )
        else:
            options = df[col].dropna().astype(str).unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(
                col.replace("-", " ").title(),
                options
            )

if st.button("🔮 Predict Price", type="primary", use_container_width=True):
    X_new = pd.DataFrame([input_data])
    prediction = float(model.predict(X_new)[0])

    st.markdown("---")
    st.markdown("### Estimated Price")
    st.markdown(
        f'<div class="hero"><div class="price">${prediction:,.0f}</div>'
        f'<div>Predicted automobile price</div></div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Records", len(df))
c2.metric("Features", len(features))
c3.metric("Target", "Price")

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)

st.caption("Educational machine-learning project built from the provided Automobile dataset.")
