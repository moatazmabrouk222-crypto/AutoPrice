# AutoPrice AI 🚗

A Streamlit machine-learning web app that predicts automobile prices using XGBoost and the provided Automobile dataset.

## Features
- Automatic model training on first app launch
- Robust numeric/categorical preprocessing
- XGBoost regression
- Interactive price prediction form
- Dataset overview

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

You do **not** need to run `train.py` before starting the app. If the saved model is missing, `app.py` trains it automatically.

## Optional manual training

```bash
python train.py
```

## Streamlit Cloud

Push the project to GitHub and deploy `app.py`. Streamlit will install the dependencies and the app will automatically train the model on its first run.
