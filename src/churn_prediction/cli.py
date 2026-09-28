from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import DEFAULT_DATA_PATH, RANDOM_SEED
from .data import inspect_dataset, load_dataset, prepare_splits
from .eda import create_eda_report
from .inference import predict_customer, predict_customers
from .models import MODEL_NAMES
from .training import create_comparison_report, finalize_selected_model, train_candidate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IBM Telco churn Stage 1 pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect dataset quality and schema")
    _data_argument(inspect)
    inspect.add_argument("--output", default="reports/dataset_inspection.json")

    preprocess = subparsers.add_parser(
        "preprocess", help="Validate data and create deterministic stratified raw splits"
    )
    _data_argument(preprocess)
    preprocess.add_argument("--output-dir", default="data/processed")
    preprocess.add_argument("--report", default="reports/preprocessing_manifest.json")
    preprocess.add_argument("--seed", type=int, default=RANDOM_SEED)

    eda = subparsers.add_parser("eda", help="Create EDA plots and summaries")
    eda.add_argument(
        "--data",
        default="data/processed/train.csv",
        help="Training split used for leakage-safe EDA (default: %(default)s)",
    )
    eda.add_argument("--output-dir", default="reports/eda")

    train = subparsers.add_parser("train", help="Fit a candidate without tuning")
    _model_argument(train)
    _workflow_arguments(train)

    tune = subparsers.add_parser("tune", help="Run a bounded randomized search")
    tune.add_argument("--model", choices=[*MODEL_NAMES, "all"], required=True)
    _workflow_arguments(tune)
    tune.add_argument("--n-iter", type=int, default=10)
    tune.add_argument("--cv-folds", type=int, default=5)

    compare = subparsers.add_parser(
        "compare", help="Create a candidate comparison without declaring a winner"
    )
    compare.add_argument("--reports-dir", default="reports")

    finalize = subparsers.add_parser(
        "finalize", help="Refit an explicitly selected candidate and evaluate test once"
    )
    _model_argument(finalize)
    _workflow_arguments(finalize)

    predict = subparsers.add_parser("predict", help="Score one customer JSON record")
    predict.add_argument("--artifact", help="Optional experimental artifact; production registry is default")
    predict.add_argument("--registry", default="config/production_model.json")
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", help="JSON object as a string")
    source.add_argument("--json-file", help="Path to a JSON object")

    batch = subparsers.add_parser("predict-batch", help="Score customer rows from a CSV")
    batch.add_argument("--input-csv", required=True)
    batch.add_argument("--output-csv", required=True)
    batch.add_argument("--artifact", help="Optional experimental artifact; production registry is default")
    batch.add_argument("--registry", default="config/production_model.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        report = inspect_dataset(load_dataset(args.data))
        _write_json(args.output, report)
        print(json.dumps(report, indent=2))
    elif args.command == "preprocess":
        result = prepare_splits(args.data, args.output_dir, args.report, args.seed)
        print(json.dumps(result, indent=2))
    elif args.command == "eda":
        result = create_eda_report(load_dataset(args.data), args.output_dir)
        print(f"EDA created in {args.output_dir}; rows={result['dataset_inspection']['rows']}")
    elif args.command == "train":
        _print_result(
            train_candidate(
                args.model, args.processed_dir, args.artifacts_dir, args.reports_dir,
                tune=False, random_seed=args.seed,
            )
        )
    elif args.command == "tune":
        names = MODEL_NAMES if args.model == "all" else (args.model,)
        for name in names:
            _print_result(
                train_candidate(
                    name, args.processed_dir, args.artifacts_dir, args.reports_dir,
                    tune=True, n_iter=args.n_iter, cv_folds=args.cv_folds,
                    random_seed=args.seed,
                )
            )
    elif args.command == "compare":
        comparison = create_comparison_report(args.reports_dir)
        print(comparison.to_string(index=False))
    elif args.command == "finalize":
        _print_result(
            finalize_selected_model(
                args.model, args.processed_dir, args.artifacts_dir, args.reports_dir
            )
        )
    elif args.command == "predict":
        record = json.loads(args.json if args.json else Path(args.json_file).read_text())
        _print_result(predict_customer(record, args.artifact, args.registry))
    elif args.command == "predict-batch":
        import pandas as pd

        source = pd.read_csv(args.input_csv)
        predictions = predict_customers(
            source.to_dict(orient="records"), args.artifact, args.registry
        )
        output = pd.concat([source.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(output_path, index=False)
        print(f"Scored {len(output)} customers to {output_path}")
    return 0


def _data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))


def _model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)


def _workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)


def _write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _print_result(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
