#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
import inspect
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TIME_BIN_CLASSES = [0, 1, 2, 3]
EPS = 1.0e-6


def log(message: str) -> None:
    print(message, flush=True)


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def parse_feature_columns(raw: str) -> list[str]:
    columns = [item.strip() for item in raw.split(",") if item.strip()]
    if not columns:
        raise ValueError("--feature-columns must contain at least one column.")
    if len(set(columns)) != len(columns):
        raise ValueError("--feature-columns contains duplicates.")
    return columns


def parse_mrdt_values(raw: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        sex, value = item.split("=", maxsplit=1)
        sex = sex.strip()
        mrdt = float(value)
        if not sex or not math.isfinite(mrdt) or mrdt <= 0:
            raise ValueError(f"Invalid MRDT item: {item!r}")
        values[sex] = mrdt
    if not values:
        raise ValueError("--mrdt-values must be provided, for example F=8,M=8.")
    return values


@dataclass(frozen=True)
class PublicSchema:
    # Public, generic analysis schema. Private source-column mapping stays outside this release.
    subject_id: str
    chronological_age: str
    sex: str
    followup_years: str
    event_death: str
    index_year: str


def project_public_schema(df: pd.DataFrame, schema: PublicSchema, feature_columns: list[str] | None = None) -> pd.DataFrame:
    required = [
        schema.subject_id,
        schema.chronological_age,
        schema.sex,
        schema.followup_years,
        schema.event_death,
        schema.index_year,
    ]
    if feature_columns is not None:
        required.extend(feature_columns)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required column(s): {missing}")

    out = pd.DataFrame(
        {
            "subject_id": df[schema.subject_id].astype("string").str.strip(),
            "chronological_age": pd.to_numeric(df[schema.chronological_age], errors="coerce"),
            "sex": df[schema.sex].astype("string").str.strip(),
            "followup_years": pd.to_numeric(df[schema.followup_years], errors="coerce"),
            "event_death": pd.to_numeric(df[schema.event_death], errors="coerce"),
            "index_year": pd.to_numeric(df[schema.index_year], errors="coerce"),
        }
    )
    if feature_columns is not None:
        for column in feature_columns:
            out[column] = pd.to_numeric(df[column], errors="coerce")
    return out


def validate_unique_subjects(df: pd.DataFrame, label: str) -> None:
    if bool(df["subject_id"].duplicated().any()):
        raise ValueError(f"{label} contains duplicated subject_id values.")


def clean_analytic_rows(df: pd.DataFrame, feature_columns: list[str] | None) -> pd.DataFrame:
    # Keep rows with complete covariates and valid right-censored survival labels.
    required = ["subject_id", "chronological_age", "sex", "followup_years", "event_death", "index_year"]
    if feature_columns is not None:
        required.extend(feature_columns)

    out = df.copy()
    out = out[out["subject_id"].notna() & (out["subject_id"].astype(str).str.len() > 0)].copy()
    out["event_death"] = out["event_death"].round()
    finite_columns = ["chronological_age", "followup_years", "event_death", "index_year"]
    if feature_columns is not None:
        finite_columns.extend(feature_columns)
    finite = np.logical_and.reduce([np.isfinite(pd.to_numeric(out[column], errors="coerce")) for column in finite_columns])
    out = out.loc[finite].copy()
    out = out[out["event_death"].isin([0, 1]) & (out["followup_years"] >= 0)].copy()
    out = out.dropna(subset=required).reset_index(drop=True)
    out["event_death"] = out["event_death"].astype(int)
    out["index_year"] = out["index_year"].astype(int)
    return out


def assign_temporal_split(df: pd.DataFrame, train_end_year: int, test_start_year: int) -> pd.DataFrame:
    # Time-based split avoids training on records indexed after the held-out period begins.
    out = df.copy()
    out["split"] = np.where(out["index_year"] <= train_end_year, "train", np.where(out["index_year"] >= test_start_year, "test", "excluded"))
    return out[out["split"] != "excluded"].reset_index(drop=True)


def make_time_bin_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, int]]:
    # Four-class target: event-free through 3 years, or death in year 1, 2, or 3.
    # Non-events censored before 3 years are scored but not used to fit classifiers.
    followup = df["followup_years"].astype(float)
    event = df["event_death"].astype(int)

    class_1 = (event == 1) & (followup <= 1.0)
    class_2 = (event == 1) & (followup > 1.0) & (followup <= 2.0)
    class_3 = (event == 1) & (followup > 2.0) & (followup <= 3.0)
    class_0 = (followup >= 3.0) & ~((event == 1) & (followup <= 3.0))
    eligible = class_0 | class_1 | class_2 | class_3

    labels = pd.Series(np.nan, index=df.index, dtype=float)
    labels.loc[class_0] = 0
    labels.loc[class_1] = 1
    labels.loc[class_2] = 2
    labels.loc[class_3] = 3

    diagnostics = {
        "class_0_event_free_through_3y": int(class_0.sum()),
        "class_1_death_le_1y": int(class_1.sum()),
        "class_2_death_gt_1_le_2y": int(class_2.sum()),
        "class_3_death_gt_2_le_3y": int(class_3.sum()),
        "excluded_censored_before_3y": int(((event == 0) & (followup < 3.0)).sum()),
    }
    return labels, eligible, diagnostics


def class_counts(labels: pd.Series) -> dict[str, int]:
    counts = labels.dropna().astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def assign_oof_folds(train: pd.DataFrame, labels: pd.Series, eligible: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    # Training predictions are out-of-fold, stratified by target class and sex.
    rng = np.random.default_rng(seed)
    folds = np.full(len(train), -1, dtype=int)
    strata = pd.Series("unlabeled", index=train.index, dtype="string")
    strata.loc[eligible] = "class_" + labels.loc[eligible].astype(int).astype(str)
    strata = strata + "|sex_" + train["sex"].astype(str)

    for _, index_values in train.assign(stratum=strata).groupby("stratum", sort=True).groups.items():
        positions = train.index.get_indexer(np.asarray(list(index_values)))
        rng.shuffle(positions)
        folds[positions] = np.arange(len(positions)) % n_folds
    return folds


@dataclass(frozen=True)
class DesignSpec:
    numeric_columns: list[str]
    sex_levels: list[str]
    means: dict[str, float]
    scales: dict[str, float]


def fit_design_spec(df: pd.DataFrame, numeric_columns: list[str]) -> DesignSpec:
    # Minimal preprocessing: z-score numeric predictors and one-hot encode sex.
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for column in numeric_columns:
        values = df[column].astype(float).to_numpy()
        mean = float(np.mean(values))
        scale = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
        if not math.isfinite(scale) or scale <= 1.0e-12:
            scale = 1.0
        means[column] = mean
        scales[column] = scale
    return DesignSpec(
        numeric_columns=numeric_columns,
        sex_levels=sorted(df["sex"].astype(str).unique().tolist()),
        means=means,
        scales=scales,
    )


def transform_design(df: pd.DataFrame, spec: DesignSpec) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in spec.numeric_columns:
        values = df[column].astype(float).to_numpy()
        columns.append((values - spec.means[column]) / spec.scales[column])
    sex = df["sex"].astype(str)
    for level in spec.sex_levels:
        columns.append((sex == level).to_numpy(dtype=float))
    return np.column_stack(columns).astype(np.float32)


def build_tabpfn_classifier(args: argparse.Namespace, seed: int) -> tuple[Any, dict[str, Any]]:
    from tabpfn import TabPFNClassifier  # type: ignore
    from tabpfn.constants import ModelVersion  # type: ignore

    signature = inspect.signature(TabPFNClassifier)
    parameters = signature.parameters
    kwargs: dict[str, Any] = {}
    if args.device != "auto" and "device" in parameters:
        kwargs["device"] = args.device
    if "n_estimators" in parameters:
        kwargs["n_estimators"] = args.n_estimators
    if "ignore_pretraining_limits" in parameters:
        kwargs["ignore_pretraining_limits"] = True
    if "random_state" in parameters:
        kwargs["random_state"] = seed

    model_version = ModelVersion(args.tabpfn_model_version)
    model = TabPFNClassifier.create_default_for_version(model_version, **kwargs)
    return model, {"model_version": model_version.value, "init_kwargs": kwargs}


def predict_class_probabilities(model: Any, design: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(design), dtype=float)
    classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    aligned = np.zeros((design.shape[0], len(TIME_BIN_CLASSES)), dtype=float)
    for output_index, class_value in enumerate(TIME_BIN_CLASSES):
        if class_value in classes:
            aligned[:, output_index] = probabilities[:, classes.index(class_value)]
    return np.clip(aligned, 0.0, 1.0)


def stratified_context(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    # Cap TabPFN context size while preserving class/sex representation.
    if len(df) <= max_rows:
        return df.copy().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_random"] = rng.random(len(work))
    work["_stratum"] = work["y_time_bin"].astype(int).astype(str) + "|" + work["sex"].astype(str)
    strata = list(work.groupby("_stratum", sort=True))
    base_take = max(1, max_rows // len(strata))
    selected = [group.nsmallest(base_take, "_random") for _, group in strata]
    out = pd.concat(selected, ignore_index=True)
    if len(out) < max_rows:
        remaining = work[~work["subject_id"].isin(out["subject_id"])]
        out = pd.concat([out, remaining.nsmallest(max_rows - len(out), "_random")], ignore_index=True)
    return out.nsmallest(max_rows, "_random").drop(columns=["_random", "_stratum"]).reset_index(drop=True)


def fit_predict_pair(
    full_training_rows: pd.DataFrame,
    reference_training_rows: pd.DataFrame,
    prediction_rows: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    # Paired classifiers estimate full clinical risk and age/sex reference risk.
    full_context = stratified_context(full_training_rows, args.max_context_rows, seed)
    reference_context = stratified_context(reference_training_rows, args.max_context_rows, seed + 1)

    full_predictors = ["chronological_age", *feature_columns]
    reference_predictors = ["chronological_age"]
    full_design_spec = fit_design_spec(full_context, full_predictors)
    reference_design_spec = fit_design_spec(reference_context, reference_predictors)

    full_model, full_init = build_tabpfn_classifier(args, seed)
    reference_model, reference_init = build_tabpfn_classifier(args, seed)
    full_model.fit(transform_design(full_context, full_design_spec), full_context["y_time_bin"].astype(int).to_numpy())
    reference_model.fit(
        transform_design(reference_context, reference_design_spec),
        reference_context["y_time_bin"].astype(int).to_numpy(),
    )

    return (
        predict_class_probabilities(full_model, transform_design(prediction_rows, full_design_spec)),
        predict_class_probabilities(reference_model, transform_design(prediction_rows, reference_design_spec)),
        {
            "seed": seed,
            "full_context_rows": int(len(full_context)),
            "reference_context_rows": int(len(reference_context)),
            "full_model": full_init,
            "reference_model": reference_init,
        },
    )


def add_tgage_scores(df: pd.DataFrame, mrdt_values: dict[str, float], eps: float) -> pd.DataFrame:
    # Convert excess mortality risk into age units using sex-specific MRDT.
    out = df.copy()
    age = out["chronological_age"].astype(float).to_numpy()
    mrdt = out["sex"].astype(str).map(mrdt_values).astype(float).to_numpy()
    if not np.isfinite(mrdt).all():
        missing = sorted(set(out.loc[~np.isfinite(mrdt), "sex"].astype(str)))
        raise ValueError(f"Missing MRDT values for sex label(s): {missing}")

    for prefix in ["full", "reference"]:
        out[f"risk_{prefix}_1y"] = out[f"p_{prefix}_class_1"]
        out[f"risk_{prefix}_2y"] = out[f"p_{prefix}_class_1"] + out[f"p_{prefix}_class_2"]
        out[f"risk_{prefix}_3y"] = (
            out[f"p_{prefix}_class_1"] + out[f"p_{prefix}_class_2"] + out[f"p_{prefix}_class_3"]
        )

    for horizon in [1, 2, 3]:
        full_risk = np.clip(out[f"risk_full_{horizon}y"].astype(float).to_numpy(), eps, 1.0 - eps)
        reference_risk = np.clip(out[f"risk_reference_{horizon}y"].astype(float).to_numpy(), eps, 1.0 - eps)
        acceleration = np.log2(full_risk / reference_risk) * mrdt
        out[f"tgage_acceleration_{horizon}y"] = acceleration
        out[f"tgage_{horizon}y"] = age + acceleration
    return out


def score_tgage(
    modeling_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_columns: list[str],
    mrdt_values: dict[str, float],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # OOF models score training rows; one final refit scores held-out test rows.
    train = modeling_df[modeling_df["split"] == "train"].copy().reset_index(drop=True)
    test = modeling_df[modeling_df["split"] == "test"].copy().reset_index(drop=True)
    reference_train = reference_df[reference_df["split"] == "train"].copy().reset_index(drop=True)

    train_labels, train_eligible, train_label_diagnostics = make_time_bin_labels(train)
    reference_labels, reference_eligible, reference_label_diagnostics = make_time_bin_labels(reference_train)
    train["y_time_bin"] = train_labels
    reference_train["y_time_bin"] = reference_labels

    full_training_pool = train.loc[train_eligible].copy().reset_index(drop=True)
    reference_training_pool = reference_train.loc[reference_eligible].copy().reset_index(drop=True)
    observed_counts = class_counts(full_training_pool["y_time_bin"])
    if len(observed_counts) < 2:
        raise ValueError(f"At least two time-bin classes are required for training. Observed: {observed_counts}")

    n_folds = min(args.n_folds, min(observed_counts.values()))
    if n_folds < 2:
        raise ValueError("Out-of-fold scoring requires at least two examples in the smallest observed class.")
    train["_fold"] = assign_oof_folds(train, train_labels, train_eligible, n_folds, args.seed)

    for prefix in ["full", "reference"]:
        for class_value in TIME_BIN_CLASSES:
            train[f"p_{prefix}_class_{class_value}"] = np.nan
            test[f"p_{prefix}_class_{class_value}"] = np.nan

    fold_diagnostics: list[dict[str, Any]] = []
    for fold_id in range(n_folds):
        fit_rows = train[(train["_fold"] != fold_id) & train["y_time_bin"].notna()].copy().reset_index(drop=True)
        heldout_rows = train[train["_fold"] == fold_id].copy()
        heldout_ids = set(heldout_rows["subject_id"].astype(str))
        reference_fit_rows = reference_training_pool[~reference_training_pool["subject_id"].astype(str).isin(heldout_ids)]
        full_prob, reference_prob, diagnostics = fit_predict_pair(
            fit_rows,
            reference_fit_rows.reset_index(drop=True),
            heldout_rows,
            feature_columns,
            args,
            args.seed + fold_id + 1,
        )
        for class_index, class_value in enumerate(TIME_BIN_CLASSES):
            train.loc[heldout_rows.index, f"p_full_class_{class_value}"] = full_prob[:, class_index]
            train.loc[heldout_rows.index, f"p_reference_class_{class_value}"] = reference_prob[:, class_index]
        diagnostics["fold"] = fold_id
        diagnostics["fit_rows"] = int(len(fit_rows))
        diagnostics["scored_rows"] = int(len(heldout_rows))
        fold_diagnostics.append(diagnostics)
        log(f"- Fold {fold_id + 1}/{n_folds}: fit={len(fit_rows)}, scored={len(heldout_rows)}")

    full_prob, reference_prob, final_diagnostics = fit_predict_pair(
        full_training_pool,
        reference_training_pool,
        test,
        feature_columns,
        args,
        args.seed + 100_000,
    )
    for class_index, class_value in enumerate(TIME_BIN_CLASSES):
        test[f"p_full_class_{class_value}"] = full_prob[:, class_index]
        test[f"p_reference_class_{class_value}"] = reference_prob[:, class_index]

    predictions = pd.concat([train, test], ignore_index=True)
    predictions = add_tgage_scores(predictions, mrdt_values, EPS)
    diagnostics = {
        "train_time_bin_labels": train_label_diagnostics,
        "reference_time_bin_labels": reference_label_diagnostics,
        "train_class_counts": observed_counts,
        "n_folds": int(n_folds),
        "folds": fold_diagnostics,
        "final_test_model": final_diagnostics,
    }
    return predictions, diagnostics


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    schema = PublicSchema(
        subject_id=args.subject_id_column,
        chronological_age=args.age_column,
        sex=args.sex_column,
        followup_years=args.followup_column,
        event_death=args.event_column,
        index_year=args.index_year_column,
    )
    feature_columns = parse_feature_columns(args.feature_columns)

    modeling = project_public_schema(read_csv(Path(args.modeling_table)), schema, feature_columns)
    reference_path = Path(args.reference_table) if args.reference_table else Path(args.modeling_table)
    reference = project_public_schema(read_csv(reference_path), schema, None)

    validate_unique_subjects(modeling, "modeling table")
    validate_unique_subjects(reference, "reference table")
    modeling = assign_temporal_split(clean_analytic_rows(modeling, feature_columns), args.train_end_year, args.test_start_year)
    reference = assign_temporal_split(clean_analytic_rows(reference, None), args.train_end_year, args.test_start_year)
    if modeling.empty or reference.empty:
        raise ValueError("No rows remain after required-field filtering and temporal split assignment.")
    if not set(modeling["sex"].astype(str)).issubset(set(parse_mrdt_values(args.mrdt_values))):
        raise ValueError("Every sex label in the modeling table must have a corresponding MRDT value.")

    return modeling, reference, feature_columns, {
        "modeling_table": args.modeling_table,
        "reference_table": str(reference_path),
        "feature_columns": feature_columns,
        "schema": schema.__dict__,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Core TGAge implementation for anonymized tabular survival data.")
    parser.add_argument("--modeling-table", required=True, help="CSV containing anonymized survival columns and feature columns.")
    parser.add_argument("--reference-table", default="", help="Optional CSV for the age/sex reference model. Defaults to modeling table.")
    parser.add_argument("--feature-columns", required=True, help="Comma-separated anonymized feature column names.")
    parser.add_argument("--out-dir", default="outputs/tgage")

    parser.add_argument("--subject-id-column", default="subject_id")
    parser.add_argument("--age-column", default="chronological_age")
    parser.add_argument("--sex-column", default="sex")
    parser.add_argument("--followup-column", default="followup_years")
    parser.add_argument("--event-column", default="event_death")
    parser.add_argument("--index-year-column", default="index_year")

    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2022)
    parser.add_argument("--mrdt-values", required=True, help="Comma-separated sex-specific values, for example F=8,M=8.")

    parser.add_argument("--tabpfn-model-version", choices=["v2", "v2.5", "v2.6", "v3"], default="v3")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--max-context-rows", type=int, default=10000)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_end_year >= args.test_start_year:
        raise ValueError("--train-end-year must be less than --test-start-year.")
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mrdt_values = parse_mrdt_values(args.mrdt_values)

    log("TGAge run started")
    modeling, reference, feature_columns, input_diagnostics = load_inputs(args)
    predictions, score_diagnostics = score_tgage(modeling, reference, feature_columns, mrdt_values, args)

    predictions_path = out_dir / "tgage_predictions.csv"
    diagnostics_path = out_dir / "diagnostics.json"
    config_path = out_dir / "run_config.json"
    predictions.to_csv(predictions_path, index=False)
    write_json(diagnostics_path, score_diagnostics)
    write_json(
        config_path,
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": input_diagnostics,
            "mrdt_values": mrdt_values,
            "tabpfn_version": package_version("tabpfn"),
            "numpy_version": package_version("numpy"),
            "pandas_version": package_version("pandas"),
            "outputs": {
                "predictions": predictions_path,
                "diagnostics": diagnostics_path,
            },
        },
    )
    log(f"- Wrote {predictions_path}")
    log(f"- Wrote {diagnostics_path}")
    log(f"- Wrote {config_path}")
    log("TGAge run completed")


if __name__ == "__main__":
    main()
