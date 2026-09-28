import numpy as np

from churn_prediction.preprocessing import build_preprocessor


def test_preprocessor_fits_and_handles_missing_numeric(telco_frame):
    X = telco_frame.drop(columns="Churn")
    transformed = build_preprocessor().fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > X.shape[1]
    assert np.isfinite(transformed.toarray() if hasattr(transformed, "toarray") else transformed).all()


def test_preprocessor_handles_unseen_category(telco_frame):
    X = telco_frame.drop(columns="Churn")
    preprocessor = build_preprocessor().fit(X.iloc[:-1])
    unseen = X.iloc[[-1]].copy()
    unseen["PaymentMethod"] = "A future payment method"
    assert preprocessor.transform(unseen).shape[0] == 1

