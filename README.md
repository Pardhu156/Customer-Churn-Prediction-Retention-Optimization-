# Customer Churn Prediction & Retention Optimization — Stage 1

Stage 1 implements reproducible data preparation, exploratory analysis, churn
modeling, validation-based comparison, threshold analysis, persistence, and
single/batch inference for the IBM Telco Customer Churn dataset. It does not
implement Stage 2 work (SHAP, segmentation, CLV, ROI, hypothesis testing, or a
dashboard).

## Final Stage 1 model

The finalized production model is tuned Logistic Regression (`C=0.1`, no class
weighting) at threshold `0.35`. It achieved validation F1 `0.6556`, ROC-AUC
`0.8598`, and PR-AUC `0.6740`; its one-time untouched-test results are accuracy
`0.7722`, precision `0.5558`, recall `0.7059`, F1 `0.6219`, ROC-AUC `0.8329`,
and PR-AUC `0.6222`. The production registry is
`config/production_model.json`. See `reports/STAGE1_RESULTS.md` for the complete
candidate comparison and selection rationale.

## Dataset

The dataset is intentionally not bundled. Download the real IBM Telco Customer
Churn CSV and place it at exactly:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The target is `Churn` (`Yes`/`No`). `customerID` is retained for data-quality
checks and split traceability, but removed inside the modeling pipeline. The
required schema is:

```text
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract,
PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Churn
```

The loader fails clearly on an absent file, empty dataset, missing columns, or
invalid target labels. `tenure`, `MonthlyCharges`, and `TotalCharges` are coerced
to numeric; canonical blank `TotalCharges` values become missing and are median
imputed using training data only.

## Setup

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

XGBoost uses a native OpenMP runtime. If macOS reports that
`libxgboost.dylib` cannot be loaded, install it once with `brew install libomp`
(or use a Conda environment with XGBoost and OpenMP from the same channel).

All commands below assume the repository root and the activated environment.

## Data inspection, preparation, and EDA

```bash
churn-stage1 inspect
churn-stage1 preprocess
churn-stage1 eda
```

`preprocess` creates deterministic, stratified 60%/20%/20% train, validation,
and test CSVs under `data/processed/`. The test set is not used by candidate
training, tuning, threshold selection, or comparison. Numeric imputers,
scalers, and categorical encoders are fitted only on training folds/data inside
scikit-learn pipelines, which prevents preprocessing leakage.

EDA outputs are written under `reports/eda/`, including categorical churn-rate
plots for contract, payment method, internet service, support/security,
paperless billing, and senior-citizen status; numeric comparisons for tenure,
monthly charges, and total charges; and JSON/CSV summaries. Associations are
descriptive, not causal. The EDA command defaults to the prepared training split
so validation and test outcomes are not inspected.

## Features and preprocessing

Raw categorical fields are most-frequent imputed and one-hot encoded with
unknown-category handling. Numeric fields are median imputed; they are scaled
for Logistic Regression and left unscaled for tree models.

The leakage-safe transformer creates only row-level features available at
prediction time:

- `tenure_group`: 0–12, 13–24, 25–48, 49–60, or 61+ months.
- `service_count`: count of affirmative subscribed service fields.
- `automatic_payment`: automatic bank transfer or credit card indicator.
- `month_to_month_contract`: month-to-month contract indicator.
- `has_internet`: any internet service indicator.
- `has_streaming`: TV or movie streaming indicator.
- `has_support`: online security or tech support indicator.

## Candidate training and evaluation

Train the untuned baseline/candidates separately:

```bash
churn-stage1 train --model logistic_regression
churn-stage1 train --model random_forest
churn-stage1 train --model xgboost
```

Each command fits only the training split and reports train/validation Accuracy,
Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix, and train-minus-
validation gaps. Thresholds from 0.05 through 0.95 are evaluated on validation
data. The recommended threshold maximizes validation F1, with ties resolved by
proximity to 0.5. No business-cost optimization is performed.

Run bounded randomized tuning and regenerate candidate artifacts/evaluation:

```bash
churn-stage1 tune --model logistic_regression --n-iter 10 --cv-folds 5
churn-stage1 tune --model random_forest --n-iter 10 --cv-folds 5
churn-stage1 tune --model xgboost --n-iter 10 --cv-folds 5
churn-stage1 compare
```

Or run the same small search for all three sequentially:

```bash
churn-stage1 tune --model all --n-iter 10 --cv-folds 5
churn-stage1 compare
```

The comparison report does not automatically declare a winner. Use validation
metrics, overfitting gaps, interpretability, and operational requirements. After
choosing a model, and only then, refit on train+validation and evaluate the
untouched test set once:

```bash
churn-stage1 finalize --model logistic_regression
```

Replace the model name with the explicitly selected candidate.

## Reports and artifacts

- Dataset/preprocessing: `reports/dataset_inspection.json` and
  `reports/preprocessing_manifest.json`
- EDA: `reports/eda/`
- Candidate/test metrics and CV results: `reports/metrics/`
- Threshold tables and recommendations: `reports/thresholds/`
- Candidate bundles: `artifacts/candidates/<model>/bundle.joblib`
- Final selected bundle: `artifacts/final/<model>/bundle.joblib`

Each artifact directory contains the complete pipeline plus separately saved
preprocessor, estimator, threshold, feature metadata, model metadata, and
metrics. The complete bundle is the supported loading/inference interface.

## Inference

Use the Python API:

```python
from churn_prediction.inference import predict_customer

result = predict_customer(customer_record, "artifacts/final/logistic_regression/bundle.joblib")
# churn_probability, predicted_class, threshold_used, model_name
```

Or use the production registry to score a JSON file containing one customer
(all predictor fields except `customerID` are required):

```bash
churn-stage1 predict \
  --json-file examples/sample_customer.json
```

Batch inference is also supported:

```bash
churn-stage1 predict-batch \
  --input-csv examples/sample_customers.csv \
  --output-csv predictions.csv
```

## Tests

```bash
pytest
```

Tests cover deterministic stratified splitting, feature engineering,
preprocessing/missing values/unseen categories, model bundle loading, inference,
invalid inputs, and deterministic predictions. Test fixtures are unit-test-only
records and are never written or presented as the IBM dataset.

## Limitations and next stages

This stage optimizes no retention cost function and makes no causal claim. Its
threshold recommendation is based only on validation F1. Performance and
calibration can drift when customer behavior or product definitions change.
Fairness, probability calibration, monitoring, and external validation remain
future considerations. Stage 2 concepts—SHAP, segmentation, CLV, ROI,
hypothesis tests, and dashboards—are deliberately out of scope.
