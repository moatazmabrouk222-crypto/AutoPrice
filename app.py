import os
import joblib
import pandas as pd
import streamlit as st
from train import train_model

st.set_page_config(
    page_title="AutoPrice",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = "data/Automobile_data.csv"
MODEL_PATH = "model/car_price_pipeline.pkl"

NUMERIC_FIELDS = {
    "symboling", "normalized-losses", "wheel-base", "length", "width", "height",
    "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
    "horsepower", "peak-rpm", "city-mpg", "highway-mpg"
}

FIELD_LABELS = {
    "symboling": "Safety Rating / Symboling",
    "normalized-losses": "Normalized Losses",
    "make": "Brand",
    "fuel-type": "Fuel Type",
    "aspiration": "Aspiration",
    "num-of-doors": "Number of Doors",
    "body-style": "Body Style",
    "drive-wheels": "Drive Wheels",
    "engine-location": "Engine Location",
    "wheel-base": "Wheel Base",
    "length": "Length",
    "width": "Width",
    "height": "Height",
    "curb-weight": "Curb Weight",
    "engine-type": "Engine Type",
    "num-of-cylinders": "Cylinders",
    "engine-size": "Engine Size",
    "fuel-system": "Fuel System",
    "bore": "Bore",
    "stroke": "Stroke",
    "compression-ratio": "Compression Ratio",
    "horsepower": "Horsepower",
    "peak-rpm": "Peak RPM",
    "city-mpg": "City MPG",
    "highway-mpg": "Highway MPG",
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container {
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.hero {
    padding: 2.4rem 2.5rem;
    border-radius: 26px;
    background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #374151 100%);
    color: white;
    margin-bottom: 1.4rem;
    border: 1px solid rgba(255,255,255,.08);
}
.hero h1 {
    font-size: 3rem;
    margin: 0 0 .45rem 0;
    letter-spacing: -1.5px;
}
.hero p {
    color: #d1d5db;
    font-size: 1.05rem;
    margin: 0;
}
.eyebrow {
    font-size: .78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 700;
    color: #9ca3af;
    margin-bottom: .6rem;
}
.stat-card {
    padding: 1.15rem 1.2rem;
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 18px;
    background: rgba(128,128,128,.045);
    min-height: 105px;
}
.stat-label {
    font-size: .82rem;
    color: #6b7280;
    margin-bottom: .3rem;
}
.stat-value {
    font-size: 1.55rem;
    font-weight: 800;
}
.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    margin: .5rem 0 .8rem 0;
}
.result-box {
    border-radius: 24px;
    padding: 2rem;
    background: linear-gradient(135deg, #f3f4f6, #ffffff);
    border: 1px solid rgba(128,128,128,.22);
    text-align: center;
}
.result-label {
    font-size: .85rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
}
.result-price {
    font-size: 3.4rem;
    line-height: 1.1;
    font-weight: 800;
    margin: .45rem 0;
}
.helper {
    color: #6b7280;
    font-size: .88rem;
}
div[data-testid="stForm"] {
    border: 1px solid rgba(128,128,128,.20);
    border-radius: 22px;
    padding: 1.2rem 1.4rem;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).replace("?", pd.NA)
    for col in NUMERIC_FIELDS | {"price"}:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Preparing the prediction model for the first run..."):
            result = train_model()
        st.toast(f"Model ready — R² {result['r2']:.2%}", icon="✅")
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()
features = [c for c in df.columns if c != "price"]

# Header
st.markdown("""
<div class="hero">
    <div class="eyebrow">Automobile Price Estimator</div>
    <h1>🚘 AutoPrice</h1>
    <p>Enter the specifications of a car and get an estimated market price in seconds.</p>
</div>
""", unsafe_allow_html=True)

# Dashboard stats
s1, s2, s3, s4 = st.columns(4)
for col, label, value in [
    (s1, "Cars in dataset", f"{len(df):,}"),
    (s2, "Input variables", f"{len(features)}"),
    (s3, "Model", "XGBoost"),
    (s4, "Test R²", "95.11%"),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

tab_predict, tab_data, tab_about = st.tabs(["🚘 Price Estimator", "📊 Dataset", "ℹ️ About"])

with tab_predict:
    st.markdown('<div class="section-title">Car specifications</div>', unsafe_allow_html=True)
    st.caption("Adjust the fields below. Values start from the dataset median or the most common category.")

    input_data = {}

    with st.form("prediction_form"):
        st.markdown("#### 1 · Vehicle identity")
        c1, c2, c3 = st.columns(3)
        identity_fields = ["make", "fuel-type", "aspiration", "body-style", "drive-wheels", "engine-location", "num-of-doors"]
        identity_cols = [c1, c2, c3]
        for i, col in enumerate(identity_fields):
            if col not in features:
                continue
            with identity_cols[i % 3]:
                options = df[col].dropna().astype(str).unique().tolist()
                input_data[col] = st.selectbox(FIELD_LABELS.get(col, col.title()), options)

        st.markdown("#### 2 · Engine & performance")
        c1, c2, c3 = st.columns(3)
        performance_fields = [
            "engine-type", "num-of-cylinders", "fuel-system",
            "engine-size", "horsepower", "peak-rpm",
            "compression-ratio", "bore", "stroke"
        ]
        performance_cols = [c1, c2, c3]
        for i, col in enumerate(performance_fields):
            if col not in features:
                continue
            with performance_cols[i % 3]:
                if col in NUMERIC_FIELDS:
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    input_data[col] = st.number_input(
                        FIELD_LABELS.get(col, col.title()),
                        min_value=float(series.min()),
                        max_value=float(series.max()),
                        value=float(series.median()),
                        step=0.1 if col in {"bore", "stroke", "compression-ratio"} else 1.0,
                    )
                else:
                    options = df[col].dropna().astype(str).unique().tolist()
                    input_data[col] = st.selectbox(FIELD_LABELS.get(col, col.title()), options)

        st.markdown("#### 3 · Dimensions & efficiency")
        c1, c2, c3 = st.columns(3)
        physical_fields = [
            "wheel-base", "length", "width", "height",
            "curb-weight", "city-mpg", "highway-mpg", "symboling", "normalized-losses"
        ]
        physical_cols = [c1, c2, c3]
        for i, col in enumerate(physical_fields):
            if col not in features:
                continue
            with physical_cols[i % 3]:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                input_data[col] = st.number_input(
                    FIELD_LABELS.get(col, col.title()),
                    min_value=float(series.min()),
                    max_value=float(series.max()),
                    value=float(series.median()),
                    step=1.0,
                )

        st.write("")
        submitted = st.form_submit_button("Estimate Automobile Price", type="primary", use_container_width=True)

    if submitted:
        # Align the submitted form exactly with the columns used during training.
        # This prevents ColumnTransformer "columns are missing" errors.
        X_new = pd.DataFrame([input_data])
        missing = [col for col in features if col not in X_new.columns]
        if missing:
            st.error("Some required vehicle fields are missing: " + ", ".join(missing))
            st.stop()
        X_new = X_new[features]
        prediction = float(model.predict(X_new)[0])
        median_price = float(df["price"].median())

        st.write("")
        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label">Estimated automobile price</div>
                <div class="result-price">${prediction:,.0f}</div>
                <div class="helper">Model estimate based on the specifications you entered.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1, r2 = st.columns(2)
        r1.metric("Estimated price", f"${prediction:,.0f}")
        difference = prediction - median_price
        r2.metric(
            "Difference from dataset median",
            f"${abs(difference):,.0f}",
            "Above median" if difference >= 0 else "Below median",
        )

with tab_data:
    st.markdown('<div class="section-title">Dataset overview</div>', unsafe_allow_html=True)
    st.caption("The application uses the provided Automobile dataset for training and prediction.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total records", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Median price", f"${df['price'].median():,.0f}")

    st.markdown("#### Price distribution")
    chart = df["price"].dropna().sort_values().reset_index(drop=True)
    st.line_chart(chart, height=260)

    with st.expander("View raw dataset"):
        st.dataframe(df, use_container_width=True, height=420)

with tab_about:
    st.markdown('<div class="section-title">About AutoPrice</div>', unsafe_allow_html=True)
    st.write(
        "AutoPrice is a machine-learning web application built around the provided "
        "Automobile dataset. It preprocesses numerical and categorical variables, "
        "trains an XGBoost regression model, and uses the trained pipeline to estimate "
        "the price of a new automobile."
    )

    a1, a2 = st.columns(2)
    with a1:
        st.markdown("**Workflow**")
        st.markdown("1. Load and clean the dataset\n2. Prepare numerical and categorical features\n3. Split data into training and testing sets\n4. Train the regression model\n5. Predict the price from user inputs")
    with a2:
        st.markdown("**Technology**")
        st.markdown("Python · Pandas · Scikit-learn · XGBoost · Streamlit")

st.write("")
st.caption("AutoPrice · Educational machine-learning project")
