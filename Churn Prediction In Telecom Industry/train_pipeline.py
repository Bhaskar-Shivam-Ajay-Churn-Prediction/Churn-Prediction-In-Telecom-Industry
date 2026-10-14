from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.xlsx"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SHEET_NAME = "Databel - Customer"
TARGET = "Churn Label"

# These columns should not be used to predict churn:
# identifiers can encourage memorization; churn category/reason are only known
# after a customer has churned and would cause target leakage.
DROP_COLUMNS = [
    "Customer ID",
    "Phone Number",
    "Churn Category",
    "Churn Reason",
]

def main():
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    df.columns = df.columns.str.strip()

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found. Columns: {list(df.columns)}")

    # Remove exact duplicate rows and rows with no target label.
    df = df.drop_duplicates().dropna(subset=[TARGET]).copy()

    # Target is Yes/No in this dataset.
    y = df[TARGET].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    valid = y.notna()
    df = df.loc[valid].copy()
    y = y.loc[valid].astype(int)

    cols_to_drop = [c for c in [TARGET, *DROP_COLUMNS] if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Keep a record of the exact feature names for prediction.
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\\n--- Test-set evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.3f}")
    print("\\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Stayed (No)", "Churned (Yes)"]))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and its feature schema together.
    joblib.dump(
        {"pipeline": pipeline, "feature_columns": feature_columns},
        MODEL_DIR / "churn_pipeline.joblib",
    )

    # Save test predictions for review.
    test_results = X_test.copy()
    test_results["Actual Churn"] = y_test.map({0: "No", 1: "Yes"})
    test_results["Predicted Churn"] = pd.Series(y_pred, index=X_test.index).map({0: "No", 1: "Yes"})
    test_results["Churn Probability"] = y_prob
    test_results.to_excel(OUTPUT_DIR / "test_predictions.xlsx", index=False)

    print(f"\\nSaved model to: {MODEL_DIR / 'churn_pipeline.joblib'}")
    print(f"Saved test predictions to: {OUTPUT_DIR / 'test_predictions.xlsx'}")

if __name__ == "__main__":
    main()
