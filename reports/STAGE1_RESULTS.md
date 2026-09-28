# Stage 1 final results

## Candidate comparison

All candidates used the same stratified train/validation/test split, random seed
42, bounded `RandomizedSearchCV`, and five-fold stratified cross-validation with
ROC-AUC scoring. Thresholds were selected on validation data by maximum F1.

| Model | Threshold | Val accuracy | Val precision | Val recall | Val F1 | Val ROC-AUC | Val PR-AUC | Train–val F1 gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.35 | 0.7935 | 0.5881 | 0.7406 | **0.6556** | **0.8598** | 0.6740 | -0.0122 |
| Random Forest | 0.50 | 0.7679 | 0.5438 | **0.7807** | 0.6411 | 0.8546 | 0.6687 | 0.1044 |
| XGBoost | 0.35 | 0.7906 | 0.5821 | 0.7487 | 0.6550 | 0.8568 | **0.6774** | 0.0077 |

Logistic Regression was selected because it has the best validation F1 and
ROC-AUC, near-best PR-AUC, stable train/validation performance, straightforward
probabilistic inference, and lower operational complexity than the tree
ensembles. Random Forest's higher recall comes with lower precision/F1 and
material overfitting. XGBoost is competitive but does not provide a meaningful
validation improvement to justify its added native dependency and complexity.

At threshold 0.35, Logistic Regression validation recall improves from 0.5535
at threshold 0.50 to 0.7406, while F1 improves from 0.6170 to 0.6556. This is a
validation-F1 choice, not a business-cost-optimized threshold.

## One-time untouched test evaluation

The selected tuned Logistic Regression (`C=0.1`, no class weighting) was refit
on the combined train+validation development set. The untouched 1,409-row test
set was then evaluated once at the locked threshold of 0.35.

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---:|---:|---:|---:|---:|---:|
| 0.7722 | 0.5558 | 0.7059 | 0.6219 | 0.8329 | 0.6222 |

Confusion matrix (`[[TN, FP], [FN, TP]]`): `[[824, 211], [110, 264]]`.

The lower test metrics versus validation indicate modest generalization
degradation, but not the severe train/validation overfitting observed for Random
Forest. Further external validation, calibration analysis, fairness review, and
business-cost threshold selection remain future work.

## Production registration

- Registry: `config/production_model.json`
- Bundle: `artifacts/final/logistic_regression/bundle.joblib`
- Locked threshold: `0.35`
- Candidate artifacts remain under `artifacts/candidates/` for experiments and
  are not used by default production inference.
