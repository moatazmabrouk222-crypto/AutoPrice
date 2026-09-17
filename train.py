import os
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

DATA_PATH = "data/Automobile_data.csv"
MODEL_PATH = "model/car_price_pipeline.pkl"


def train_model():
    df = pd.read_csv(DATA_PATH).replace("?", pd.NA)

    numeric_cols = [
        "symboling", "normalized-losses", "wheel-base", "length", "width", "height",
        "curb-weight", "engine-size", "bore", "stroke", "compression-ratio",
        "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["price"])
    X = df.drop(columns=["price"])
    y = df["price"]

    num_features = X.select_dtypes(include=["number"]).columns.tolist()
    cat_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), num_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_features)
    ])

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return {
        "rows": len(df),
        "mse": mse,
        "r2": r2,
        "model_path": MODEL_PATH,
    }


if __name__ == "__main__":
    result = train_model()
    print(f"Rows used: {result['rows']}")
    print(f"Test MSE: {result['mse']:.2f}")
    print(f"Test R2: {result['r2']:.4f}")
    print(f"Model saved to: {result['model_path']}")
