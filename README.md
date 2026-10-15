# Customer Churn Prediction — ML Pipeline

An end-to-end machine learning project using the Databel customer Excel dataset to predict whether a customer is likely to churn.

## Dataset

The workbook `data/customer_churn.xlsx` contains two sheets:

- **Databel - Customer**: customer-level data. Use this sheet for supervised ML because each row represents an individual customer and `Churn Label` is the target.
- **Databel - Aggregate**: aggregated/grouped customer data for analysis and dashboards. It is not used by the training script.

## Project structure

```text
repository/
├── README.md
└── Churn Prediction In Telecom Industry/
    ├── app.py
    ├── utils.py
    ├── eda.py
    ├── train_pipeline.py
    ├── predict.py
    ├── requirements.txt
    ├── logo.jfif
    └── data/
        ├── customer_churn.xlsx
        └── new_customers.xlsx
```

The `models/` and `outputs/` directories are created by the training and EDA scripts as needed; generated files are not committed.

## Setup

Python 3.10+ is recommended.

```bash
cd "Churn Prediction In Telecom Industry"
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS/Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the project

1. Explore the data and generate charts:

   ```bash
   python eda.py
   ```

2. Train and evaluate the model:

   ```bash
   python train_pipeline.py
   ```

3. Make predictions from a new Excel file. Place it at `data/new_customers.xlsx`, with the same input feature columns used during training, and run:

   ```bash
   python predict.py
   ```

   Or specify input and output paths:

   ```bash
   python predict.py --input "C:/path/to/new_customers.xlsx" --output "outputs/predictions.xlsx"
   ```

4. Launch the Streamlit dashboard:

   ```bash
   streamlit run app.py
   ```

   The dashboard includes customer churn metrics, an interactive EDA view, model training, and predictions from an uploaded Excel file or the bundled sample data.

## Modeling notes

- Target: `Churn Label` (`Yes` = 1, `No` = 0).
- Excludes `Customer ID` and `Phone Number` because they are identifiers.
- Excludes `Churn Category` and `Churn Reason` because these are outcome-related fields only available after churn and would leak the answer.
- Uses a train/test split, imputers, scaling, one-hot encoding, and Logistic Regression in a single scikit-learn pipeline.
- Reports accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.

Use the test-set metrics to assess performance; do not assume a good score until you run the script. For a real deployment, also validate on fresh, representative data and choose the classification threshold based on business costs.
