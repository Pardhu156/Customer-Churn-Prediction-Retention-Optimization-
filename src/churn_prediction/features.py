from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import ENGINEERED_COLUMNS, ID_COLUMN, TARGET_COLUMN

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
AUTOMATIC_PAYMENT_METHODS = {
    "Bank transfer (automatic)",
    "Credit card (automatic)",
}


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-supported features without learning from the target."""

    def fit(self, X: pd.DataFrame, y: object = None) -> "TelcoFeatureEngineer":
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoFeatureEngineer expects a pandas DataFrame")
        frame = X.copy()
        missing = sorted(set(SERVICE_COLUMNS + [
            "tenure", "PaymentMethod", "Contract", "InternetService",
            "StreamingTV", "StreamingMovies", "OnlineSecurity", "TechSupport",
        ]) - set(frame.columns))
        if missing:
            raise ValueError(f"Cannot engineer features; missing columns: {missing}")

        for column in ("tenure", "MonthlyCharges", "TotalCharges"):
            if column in frame:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if "SeniorCitizen" in frame:
            frame["SeniorCitizen"] = frame["SeniorCitizen"].astype("string")

        frame["tenure_group"] = pd.cut(
            frame["tenure"],
            bins=[-np.inf, 12, 24, 48, 60, np.inf],
            labels=["0-12", "13-24", "25-48", "49-60", "61+"],
        ).astype("object")
        frame["service_count"] = frame[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
        frame["automatic_payment"] = np.where(
            frame["PaymentMethod"].isin(AUTOMATIC_PAYMENT_METHODS), "Yes", "No"
        )
        frame["month_to_month_contract"] = np.where(
            frame["Contract"].eq("Month-to-month"), "Yes", "No"
        )
        frame["has_internet"] = np.where(
            frame["InternetService"].eq("No"), "No", "Yes"
        )
        frame["has_streaming"] = np.where(
            frame[["StreamingTV", "StreamingMovies"]].eq("Yes").any(axis=1),
            "Yes",
            "No",
        )
        frame["has_support"] = np.where(
            frame[["OnlineSecurity", "TechSupport"]].eq("Yes").any(axis=1),
            "Yes",
            "No",
        )
        return frame.drop(columns=[ID_COLUMN, TARGET_COLUMN], errors="ignore")

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        original = [
            c for c in self.feature_names_in_ if c not in {ID_COLUMN, TARGET_COLUMN}
        ]
        return np.asarray(original + ENGINEERED_COLUMNS, dtype=object)


FEATURE_DOCUMENTATION = {
    "tenure_group": "Tenure bucket: 0-12, 13-24, 25-48, 49-60, or 61+ months.",
    "service_count": "Count of affirmative subscribed phone/internet service fields.",
    "automatic_payment": "Yes for automatic bank transfer or automatic credit card.",
    "month_to_month_contract": "Yes when Contract is Month-to-month.",
    "has_internet": "Yes when InternetService is not No.",
    "has_streaming": "Yes when StreamingTV or StreamingMovies is Yes.",
    "has_support": "Yes when OnlineSecurity or TechSupport is Yes.",
}

