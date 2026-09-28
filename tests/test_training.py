from churn_prediction.data import split_dataset
from churn_prediction.training import create_comparison_report, train_candidate


def test_candidate_training_and_comparison(tmp_path, telco_frame):
    processed = tmp_path / "processed"
    artifacts = tmp_path / "artifacts"
    reports = tmp_path / "reports"
    processed.mkdir()
    for name, frame in zip(
        ("train", "validation", "test"), split_dataset(telco_frame, random_seed=42)
    ):
        frame.to_csv(processed / f"{name}.csv", index=False)

    result = train_candidate(
        "logistic_regression", processed, artifacts, reports, tune=False
    )
    comparison = create_comparison_report(reports)

    assert (artifacts / "candidates/logistic_regression/bundle.joblib").is_file()
    assert (reports / "metrics/logistic_regression.json").is_file()
    assert (reports / "thresholds/logistic_regression.csv").is_file()
    assert result["metrics"]["validation"]["roc_auc"] >= 0
    assert comparison["model"].tolist() == ["logistic_regression"]
