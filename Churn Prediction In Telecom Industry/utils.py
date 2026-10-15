"""
utils.py — shared, cached helper functions for the Streamlit app.
"""
from pathlib import Path
import io

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH   = ROOT / "data" / "customer_churn.xlsx"
MODEL_PATH  = ROOT / "models" / "churn_pipeline.joblib"
OUTPUT_DIR  = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SHEET_NAME  = "Databel - Customer"
TARGET      = "Churn Label"

DROP_COLUMNS = [
    "Customer ID",
    "Phone Number",
    "Churn Label",
    "Churn Category",
    "Churn Reason",
]

# ---------------------------------------------------------------------------
# Data loaders (cached so we don't re-read on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading training data…")
def load_training_data() -> pd.DataFrame:
    """Return the raw customer_churn DataFrame (Databel - Customer sheet)."""
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()
    return df


@st.cache_resource(show_spinner="Loading trained model…")
def load_model():
    """
    Return (pipeline, feature_columns) from the saved .joblib file.
    Returns (None, None) if the model has not been trained yet.
    """
    if not MODEL_PATH.exists():
        return None, None
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["feature_columns"]


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def run_prediction(
    df: pd.DataFrame,
    pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Given a raw DataFrame of new customers, return the same DataFrame with
    two extra columns appended: 'Predicted Churn' and 'Churn Probability (%)'.
    """
    X = df.drop(
        columns=[c for c in DROP_COLUMNS if c in df.columns],
        errors="ignore",
    )
    X = X.reindex(columns=feature_columns)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)[:, 1]

    results = df.copy()
    results["Predicted Churn"]      = ["Yes" if p == 1 else "No" for p in preds]
    results["Churn Probability (%)"] = (probs * 100).round(2)
    return results


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------

def get_numeric_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    cols = df.select_dtypes(include="number").columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


def get_categorical_cols(df: pd.DataFrame, exclude: list[str] | None = None) -> list[str]:
    cols = df.select_dtypes(exclude="number").columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


# ---------------------------------------------------------------------------
# Excel download helper
# ---------------------------------------------------------------------------

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to an in-memory Excel (.xlsx) byte string."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()
