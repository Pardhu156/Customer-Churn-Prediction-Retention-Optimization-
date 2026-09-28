from pathlib import Path

RANDOM_SEED = 42
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
DEFAULT_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges", "service_count"]
BASE_NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
ENGINEERED_COLUMNS = [
    "tenure_group",
    "service_count",
    "automatic_payment",
    "month_to_month_contract",
    "has_internet",
    "has_streaming",
    "has_support",
]

