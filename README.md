# AutoPrice

A Streamlit web application for automobile price estimation using the provided Automobile dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The model is trained automatically on the first run if `model/car_price_pipeline.pkl` does not exist.

## Deploy

Use `app.py` as the Main file on Streamlit Community Cloud.
