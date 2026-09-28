import pandas as pd

from churn_prediction.data import inspect_dataset, split_dataset


def test_inspection_and_deterministic_stratified_split(telco_frame):
    report = inspect_dataset(telco_frame)
    first = split_dataset(telco_frame, random_seed=42)
    second = split_dataset(telco_frame, random_seed=42)

    assert report["rows"] == 30
    assert report["target_column"] == "Churn"
    assert report["class_distribution_count"] == {"No": 20, "Yes": 10}
    for left, right in zip(first, second):
        pd.testing.assert_frame_equal(left, right)
        assert set(left["Churn"]) == {"No", "Yes"}

