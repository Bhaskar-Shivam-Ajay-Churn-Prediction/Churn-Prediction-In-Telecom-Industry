"""
utils.py — shared, cached helper functions and data sanitization for the Streamlit app.
"""
from pathlib import Path
import io
import re
import joblib
import numpy as np
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

# Canonical features in Databel churn model
NUMERIC_FEATURES = [
    "Account Length (in months)",
    "Local Calls",
    "Local Mins",
    "Intl Calls",
    "Intl Mins",
    "Extra International Charges",
    "Customer Service Calls",
    "Avg Monthly GB Download",
    "Extra Data Charges",
    "Age",
    "Number of Customers in Group",
    "Monthly Charge",
    "Total Charges",
]

CATEGORICAL_FEATURES = [
    "Intl Active",
    "Intl Plan",
    "Unlimited Data Plan",
    "State",
    "Gender",
    "Under 30",
    "Senior",
    "Group",
    "Device Protection & Online Backup",
    "Contract Type",
    "Payment Method",
]

# Common raw strings representing missing or invalid values
MISSING_VALUES_REPRESENTATIONS = {
    "?", "NA", "N/A", "n/a", "na", "null", "NULL", "None", "none",
    "nan", "NaN", "-", "--", "", " ", "#N/A", "#VALUE!", "#REF!",
    "#DIV/0!", "#NUM!", "#NAME?", "#NULL!", "missing", "MISSING",
    "unknown", "UNKNOWN",
}

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
# Feature extraction & data cleaning
# ---------------------------------------------------------------------------

def get_feature_types(pipeline, feature_columns: list[str] | None = None) -> tuple[list[str], list[str]]:
    """
    Extract (numeric_features, categorical_features) from the fitted pipeline.
    Falls back to canonical lists if preprocessor inspection fails.
    """
    num_cols = []
    cat_cols = []
    try:
        if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            prep = pipeline.named_steps["preprocessor"]
            for name, _, cols in prep.transformers:
                if name == "numeric":
                    num_cols = list(cols)
                elif name == "categorical":
                    cat_cols = list(cols)
    except Exception:
        pass

    if not num_cols:
        num_cols = [c for c in NUMERIC_FEATURES if feature_columns is None or c in feature_columns]
    if not cat_cols:
        cat_cols = [c for c in CATEGORICAL_FEATURES if feature_columns is None or c in feature_columns]

    return num_cols, cat_cols


def _normalize_col_key(col_name: str) -> str:
    """Helper to produce a normalized key for flexible column matching."""
    return re.sub(r"[^a-z0-9]", "", str(col_name).strip().lower())


def clean_raw_data(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Sanitize raw customer data to make it completely safe for model prediction:
    1. Normalizes column names (strips whitespace, case-insensitive match against feature_columns).
    2. Safely removes identifiers and target leakage columns (Customer ID, Churn Label, etc.).
    3. Replaces dirty missing-value placeholders ('?', 'NA', '-', whitespace, etc.) with np.nan.
    4. Robustly coerces numeric columns to numeric (float), converting non-numeric text to NaN.
    5. Cleans categorical columns (strips whitespace, preserves NaN for SimpleImputer).
    6. Ensures all expected feature_columns exist, reindexing to the exact model order.
    Returns (cleaned_X, audit_info) where audit_info records what was cleaned.
    """
    audit = {
        "coerced_numeric_values": 0,
        "coerced_columns": [],
        "missing_columns_added": [],
        "dropped_columns": [],
    }

    X = df.copy()

    # 1. Clean column names
    X.columns = [str(c).strip() for c in X.columns]

    # 2. Case-insensitive and whitespace-flexible column alignment
    col_lookup = {_normalize_col_key(c): c for c in feature_columns}
    drop_lookup = {_normalize_col_key(c): c for c in DROP_COLUMNS}

    rename_map = {}
    cols_to_drop = []

    for c in X.columns:
        norm_key = _normalize_col_key(c)
        if norm_key in drop_lookup:
            cols_to_drop.append(c)
        elif norm_key in col_lookup and c != col_lookup[norm_key]:
            rename_map[c] = col_lookup[norm_key]

    if rename_map:
        X = X.rename(columns=rename_map)

    # 3. Drop identifier/target columns
    if cols_to_drop:
        audit["dropped_columns"] = cols_to_drop
        X = X.drop(columns=cols_to_drop, errors="ignore")

    # 4. Check for missing feature columns and create them with NaN
    missing_cols = [c for c in feature_columns if c not in X.columns]
    if missing_cols:
        audit["missing_columns_added"] = missing_cols
        for c in missing_cols:
            X[c] = np.nan

    # 5. Reindex to exact feature_columns expected by pipeline
    X = X.reindex(columns=feature_columns)

    # 6. Global placeholder replacement for raw strings (strip whitespace first)
    def _replace_placeholders(v):
        if isinstance(v, str):
            stripped = v.strip()
            return np.nan if stripped in MISSING_VALUES_REPRESENTATIONS else stripped
        return v

    X = X.apply(lambda col: col.map(_replace_placeholders) if col.dtype == object else col)

    # 7. Clean numeric features — always coerce to numeric regardless of current dtype
    if numeric_cols is None:
        numeric_cols = [c for c in NUMERIC_FEATURES if c in feature_columns]

    for c in numeric_cols:
        if c in X.columns:
            # Strip strings and replace all placeholder representations first
            X[c] = X[c].apply(lambda v: v.strip() if isinstance(v, str) else v)
            X[c] = X[c].replace(list(MISSING_VALUES_REPRESENTATIONS), np.nan)
            before_non_null = int(X[c].notna().sum())
            X[c] = pd.to_numeric(X[c], errors="coerce")
            after_non_null = int(X[c].notna().sum())
            coerced = before_non_null - after_non_null
            if coerced > 0:
                audit["coerced_numeric_values"] += coerced
                audit["coerced_columns"].append(c)

    # 8. Clean categorical features
    if categorical_cols is None:
        categorical_cols = [c for c in CATEGORICAL_FEATURES if c in feature_columns]

    for c in categorical_cols:
        if c in X.columns:
            X[c] = X[c].apply(
                lambda v: str(v).strip()
                if (pd.notna(v) and str(v).strip() not in MISSING_VALUES_REPRESENTATIONS)
                else np.nan
            )

    return X, audit


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def run_prediction(
    df: pd.DataFrame,
    pipeline,
    feature_columns: list[str],
    return_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Given a raw DataFrame of new customers, sanitize raw data,
    predict churn risk using the pipeline, and return the DataFrame with
    'Predicted Churn' and 'Churn Probability (%)' appended.
    """
    numeric_cols, categorical_cols = get_feature_types(pipeline, feature_columns)
    X_clean, audit = clean_raw_data(df, feature_columns, numeric_cols, categorical_cols)

    preds = pipeline.predict(X_clean)
    probs = pipeline.predict_proba(X_clean)[:, 1]

    results = df.copy()
    results["Predicted Churn"]       = ["Yes" if p == 1 else "No" for p in preds]
    results["Churn Probability (%)"] = (probs * 100).round(2)

    if return_audit:
        return results, audit
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
