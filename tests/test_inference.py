import joblib
import pytest

from churn_prediction.evaluation import encode_target
from churn_prediction.inference import (
    predict_customer,
    predict_customers,
    resolve_production_artifact,
    validate_customer_record,
)
from churn_prediction.models import build_model_pipeline
from churn_prediction.persistence import load_bundle, save_artifacts


@pytest.fixture
def fitted_bundle(tmp_path, telco_frame):
    X = telco_frame.drop(columns="Churn")
    y = encode_target(telco_frame["Churn"])
    pipeline = build_model_pipeline("logistic_regression").fit(X, y)
    return save_artifacts(
        tmp_path / "model",
        pipeline,
        threshold=0.4,
        feature_metadata={"test": True},
        metrics={"validation": {"roc_auc": 0.5}},
        model_metadata={"model_name": "logistic_regression"},
    )


def test_model_loading_and_deterministic_inference(fitted_bundle, customer_record):
    assert load_bundle(fitted_bundle)["threshold"] == 0.4
    first = predict_customer(customer_record, fitted_bundle)
    second = predict_customer(customer_record, fitted_bundle)

    assert first == second
    assert 0 <= first["churn_probability"] <= 1
    assert first["predicted_class"] in {"Yes", "No"}
    assert first["threshold_used"] == 0.4


def test_registry_and_batch_inference(tmp_path, fitted_bundle, customer_record):
    registry = tmp_path / "config" / "production_model.json"
    registry.parent.mkdir()
    registry.write_text(
        '{"status":"production","artifact_path":"../model/bundle.joblib"}'
    )

    assert resolve_production_artifact(registry) == fitted_bundle.resolve()
    single = predict_customer(customer_record, registry_path=registry)
    batch = predict_customers([customer_record, customer_record], registry_path=registry)

    assert len(batch) == 2
    assert batch[0] == batch[1] == single


def test_invalid_customer_input(customer_record):
    missing = dict(customer_record)
    missing.pop("Contract")
    with pytest.raises(ValueError, match="missing required fields"):
        validate_customer_record(missing)

    invalid = dict(customer_record, MonthlyCharges="not-a-number")
    with pytest.raises(ValueError, match="MonthlyCharges must be numeric"):
        validate_customer_record(invalid)


def test_invalid_bundle_is_rejected(tmp_path):
    path = tmp_path / "invalid.joblib"
    joblib.dump({"pipeline": "not enough"}, path)
    with pytest.raises(ValueError, match="Invalid artifact bundle"):
        load_bundle(path)
