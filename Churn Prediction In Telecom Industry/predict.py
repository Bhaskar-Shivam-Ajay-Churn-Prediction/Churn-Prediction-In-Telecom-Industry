from pathlib import Path
import argparse
import joblib
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
DEFAULT_INPUT = ROOT / "data" / "new_customers.xlsx"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Columns excluded from model predictions
DROP_COLUMNS = [
    "Customer ID",
    "Phone Number",
    "Churn Label",
    "Churn Category",
    "Churn Reason"
]


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer churn from an Excel file."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the input Excel file"
    )

    parser.add_argument(
        "--sheet",
        default="0",
        help="Excel sheet name or index"
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "churn_predictions.xlsx"),
        help="Path to save the prediction results"
    )

    args = parser.parse_args()

    # Check whether the model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # Check whether the input Excel file exists
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input Excel file not found: {input_path}"
        )

    # Load the trained model
    saved = joblib.load(MODEL_PATH)

    pipeline = saved["pipeline"]
    feature_columns = saved["feature_columns"]

    # Convert numeric sheet arguments into integer indexes
    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    # Read all customer records
    new_df = pd.read_excel(
        input_path,
        sheet_name=sheet
    )

    # Clean column names
    new_df.columns = new_df.columns.str.strip()

    if new_df.empty:
        print("The input Excel file contains no customer records.")
        return

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    print(f"Total customer records loaded: {len(new_df)}")

    # Remove identifiers and target-related columns
    X_new = new_df.drop(
        columns=[
            col for col in DROP_COLUMNS
            if col in new_df.columns
        ],
        errors="ignore"
    )

    # Match the feature columns used during training
    X_new = X_new.reindex(columns=feature_columns)

    # Predict churn for every customer
    predictions = pipeline.predict(X_new)
    probabilities = pipeline.predict_proba(X_new)[:, 1]

    # Prepare the complete output
    results = new_df.copy()

    results["Predicted Churn"] = [
        "Yes" if prediction == 1 else "No"
        for prediction in predictions
    ]

    results["Churn Probability"] = probabilities

    results["Churn Probability"] = (
        results["Churn Probability"] * 100
    ).round(2)

    # Save every prediction to Excel
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.to_excel(
        output_path,
        index=False
    )

    # Display ALL predictions in the terminal
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    print("\nCOMPLETE CUSTOMER PREDICTIONS")
    print("=" * 60)

    print(
        results[
            ["Predicted Churn", "Churn Probability"]
        ].to_string(index=True)
    )

    # Display summary
    churn_count = (predictions == 1).sum()
    no_churn_count = (predictions == 0).sum()

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)

    print(f"Total customers processed : {len(results)}")
    print(f"Predicted churn            : {churn_count}")
    print(f"Predicted no churn         : {no_churn_count}")
    print(f"Output Excel file          : {output_path.resolve()}")

    print("\nAll predictions saved successfully!")


if __name__ == "__main__":
    main()