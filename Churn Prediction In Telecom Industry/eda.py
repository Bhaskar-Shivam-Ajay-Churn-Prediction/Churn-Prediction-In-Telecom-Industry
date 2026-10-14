from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer_churn.xlsx"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_excel(DATA_PATH, sheet_name="Databel - Customer")
df.columns = df.columns.str.strip()

print("Dataset shape:", df.shape)
print("\\nColumns:")
print(df.columns.tolist())
print("\\nData types and non-null counts:")
df.info()
print("\\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\\nDuplicate rows:", df.duplicated().sum())
print("\\nChurn distribution:")
print(df["Churn Label"].value_counts(dropna=False))
print("\\nChurn rate (%):")
print((df["Churn Label"].value_counts(normalize=True) * 100).round(2))

sns.countplot(data=df, x="Churn Label")
plt.title("Customer Churn Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=150)
plt.show()

sns.countplot(data=df, x="Contract Type", hue="Churn Label")
plt.title("Churn by Contract Type")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "churn_by_contract.png", dpi=150)
plt.show()

sns.boxplot(data=df, x="Churn Label", y="Account Length (in months)")
plt.title("Account Length by Churn Status")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "account_length_by_churn.png", dpi=150)
plt.show()
