import pandas as pd
import pytest


@pytest.fixture
def telco_frame() -> pd.DataFrame:
    rows = []
    for i in range(30):
        churn = "Yes" if i % 3 == 0 else "No"
        rows.append(
            {
                "customerID": f"TEST-{i:04d}",
                "gender": "Female" if i % 2 else "Male",
                "SeniorCitizen": 1 if i % 5 == 0 else 0,
                "Partner": "Yes" if i % 2 else "No",
                "Dependents": "Yes" if i % 4 == 0 else "No",
                "tenure": i + 1,
                "PhoneService": "Yes",
                "MultipleLines": "Yes" if i % 2 else "No",
                "InternetService": ["DSL", "Fiber optic", "No"][i % 3],
                "OnlineSecurity": "Yes" if i % 4 == 0 else "No",
                "OnlineBackup": "Yes" if i % 3 == 0 else "No",
                "DeviceProtection": "Yes" if i % 5 == 0 else "No",
                "TechSupport": "Yes" if i % 6 == 0 else "No",
                "StreamingTV": "Yes" if i % 2 else "No",
                "StreamingMovies": "Yes" if i % 4 else "No",
                "Contract": ["Month-to-month", "One year", "Two year"][i % 3],
                "PaperlessBilling": "Yes" if i % 2 else "No",
                "PaymentMethod": [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ][i % 4],
                "MonthlyCharges": 25.0 + i * 2,
                "TotalCharges": None if i == 0 else (25.0 + i * 2) * (i + 1),
                "Churn": churn,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def customer_record(telco_frame: pd.DataFrame) -> dict:
    return telco_frame.drop(columns=["customerID", "Churn"]).iloc[2].to_dict()

