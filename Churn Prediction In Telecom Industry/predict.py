from pathlib import Path
import argparse
import pandas as pd
from utils import load_model, run_prediction, MODEL_PATH, ROOT

DEFAULT_INPUT = ROOT / "data" / "new_customers.xlsx"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Predict customer churn from an Excel or CSV file."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to the input Excel (.xlsx, .xls) or CSV file"
    )

    parser.add_argument(
        "--sheet",
        default="0",
        help="Excel sheet name or integer index (default: 0)"
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "churn_predictions.xlsx"),
        help="Path to save the prediction results"
    )

    args = parser.parse_args()

    # Check whether the input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load the trained model
    pipeline, feature_columns = load_model()
    if pipeline is None:
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run train_pipeline.py first."
        )

    # Convert numeric sheet arguments into integer indexes
    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    # Read records (support CSV or Excel)
    if input_path.suffix.lower() == ".csv":
        new_df = pd.read_csv(input_path)
    else:
        new_df = pd.read_excel(input_path, sheet_name=sheet)

    # Clean column names
    new_df.columns = new_df.columns.str.strip()

    if new_df.empty:
        print("The input file contains no customer records.")
        return

    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 60)
    print(f"Total customer records loaded: {len(new_df)}")

    # Run predictions with automated raw data sanitization
    results, audit = run_prediction(new_df, pipeline, feature_columns, return_audit=True)

    if audit.get("coerced_numeric_values", 0) > 0 or audit.get("dropped_columns") or audit.get("missing_columns_added"):
        print("\nData Sanitization Audit:")
        if audit.get("coerced_numeric_values", 0) > 0:
            print(f" - Cleaned non-numeric/placeholder values in: {', '.join(audit['coerced_columns'])}")
        if audit.get("missing_columns_added"):
            print(f" - Missing feature columns imputed with defaults: {', '.join(audit['missing_columns_added'])}")
        if audit.get("dropped_columns"):
            print(f" - Excluded identifier/target columns: {', '.join(audit['dropped_columns'])}")

    # Save predictions to output file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".csv":
        results.to_csv(output_path, index=False)
    else:
        results.to_excel(output_path, index=False)

    # Display summary
    churn_count = (results["Predicted Churn"] == "Yes").sum()
    no_churn_count = (results["Predicted Churn"] == "No").sum()
    avg_prob = results["Churn Probability (%)"].mean()

    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Total customers processed : {len(results)}")
    print(f"Predicted churn (Yes)      : {churn_count} ({(churn_count / len(results) * 100):.1f}%)")
    print(f"Predicted no churn (No)    : {no_churn_count} ({(no_churn_count / len(results) * 100):.1f}%)")
    print(f"Average churn probability  : {avg_prob:.2f}%")
    print(f"Output saved to            : {output_path.resolve()}")
    print("\nAll predictions generated and saved successfully!")


if __name__ == "__main__":
    main()
