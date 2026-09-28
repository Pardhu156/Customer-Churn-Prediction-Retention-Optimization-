import pandas as pd

from churn_prediction.features import TelcoFeatureEngineer


def test_feature_engineering_has_expected_values(telco_frame):
    original = telco_frame.copy(deep=True)
    result = TelcoFeatureEngineer().fit_transform(telco_frame)

    assert "customerID" not in result
    assert "Churn" not in result
    assert result.loc[0, "tenure_group"] == "0-12"
    assert result.loc[2, "automatic_payment"] == "Yes"
    assert result.loc[0, "month_to_month_contract"] == "Yes"
    assert result.loc[2, "has_internet"] == "No"
    assert result.loc[0, "service_count"] == 5
    pd.testing.assert_frame_equal(telco_frame, original)
