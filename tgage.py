#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any

import numpy as np
import pandas as pd


CLIENT_ID_CANDIDATES = ["ClientId", "ClientID", "client_id", "ID"]
AGE_CANDIDATES = ["CA", "ChronologicalAge", "chronological_age", "index_age_years", "Age"]
SEX_CANDIDATES = ["sex", "Sex", "gender", "Gender", "SEX"]
FOLLOWUP_YEARS_CANDIDATES = ["followup_years", "FollowupYears", "follow_up_years", "followup_time_years"]
FOLLOWUP_DAYS_CANDIDATES = ["followup_days", "FollowupDays", "follow_up_days", "followup_time_days"]
EVENT_CANDIDATES = ["event_death", "EventDeath", "event", "death_event"]
INDEX_DATE_CANDIDATES = ["index_date", "IndexDate", "INDEX_DATE"]
INDEX_YEAR_CANDIDATES = ["index_year", "IndexYear", "INDEX_YEAR"]

DEFAULT_OUT_DIR = Path("outputs/tgage")
TIMEBIN_CLASSES = [0, 1, 2, 3]
DEFAULT_EPS = 1.0e-6

NON_FEATURE_COLUMNS = {
    "clientid",
    "client_id",
    "id",
    "ca",
    "chronologicalage",
    "chronological_age",
    "index_age_years",
    "age",
    "sex",
    "gender",
    "followup_years",
    "followupyears",
    "follow_up_years",
    "followup_time_years",
    "followup_days",
    "followupdays",
    "follow_up_days",
    "followup_time_days",
    "event_death",
    "eventdeath",
    "event",
    "death_event",
    "split",
    "split_label",
    "model_split",
    "index_year",
    "indexyear",
    "index_date",
    "indexdate",
    "birthyear",
    "birth_year",
    "deathyear",
    "death_year",
}


def log(message: str) -> None:
    print(message, flush=True)


def env_value(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw if raw is not None and raw.strip() else default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None and raw.strip() else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None and raw.strip() else default


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def pick_first(available: list[str], candidates: list[str], context: str) -> str:
    for candidate in candidates:
        if candidate in available:
            return candidate
    lower_map = {column.lower(): column for column in available}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    raise RuntimeError(f"{context}: none of {candidates} were found in columns {available}")


def pick_optional(available: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in available:
            return candidate
    lower_map = {column.lower(): column for column in available}
    for candidate in candidates:
        found = lower_map.get(candidate.lower())
        if found is not None:
            return found
    return None


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        try:
            return pd.read_parquet(path)
        except Exception:
            import polars as pl  # type: ignore

            polars_df = pl.read_parquet(path)
            return pd.DataFrame({column: polars_df[column].to_list() for column in polars_df.columns})
    raise RuntimeError(f"Unsupported input type for {path}; expected .csv or .parquet")


def normalize_id_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def normalize_sex_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    cleaned = str(value).strip().upper()
    return cleaned if cleaned else None


def parse_date_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.mask(text.isin(["", "nan", "NaN", "<NA>", "None"]))
    compact = text.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if bool(compact.any()):
        parsed.loc[compact] = pd.to_datetime(text.loc[compact], format="%Y%m%d", errors="coerce")
    if bool((~compact).any()):
        parsed.loc[~compact] = pd.to_datetime(text.loc[~compact], errors="coerce")
    return parsed


def parse_allowed_sexes(raw: str) -> list[str]:
    return [token.strip().upper() for token in str(raw).split(",") if token.strip()]


def standardize_survival_input(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    available = list(df.columns)
    column_map = {
        "ClientId": pick_first(available, CLIENT_ID_CANDIDATES, "survival input"),
        "CA": pick_first(available, AGE_CANDIDATES, "survival input"),
        "sex": pick_first(available, SEX_CANDIDATES, "survival input"),
        "followup_years": pick_first(available, FOLLOWUP_YEARS_CANDIDATES, "survival input"),
        "event_death": pick_first(available, EVENT_CANDIDATES, "survival input"),
    }
    optional_map: dict[str, str] = {}
    for target, candidates in {
        "followup_days": FOLLOWUP_DAYS_CANDIDATES,
        "index_date": INDEX_DATE_CANDIDATES,
        "index_year": INDEX_YEAR_CANDIDATES,
    }.items():
        found = pick_optional(available, candidates)
        if found is not None:
            optional_map[target] = found

    out = pd.DataFrame(
        {
            "ClientId": normalize_id_series(df[column_map["ClientId"]]),
            "CA": pd.to_numeric(df[column_map["CA"]], errors="coerce"),
            "sex": df[column_map["sex"]].map(normalize_sex_value),
            "followup_years": pd.to_numeric(df[column_map["followup_years"]], errors="coerce"),
            "event_death": pd.to_numeric(df[column_map["event_death"]], errors="coerce").round().astype("Int64"),
        }
    )
    for target, source in optional_map.items():
        if target in {"followup_days", "index_year"}:
            out[target] = pd.to_numeric(df[source], errors="coerce").round().astype("Int64")
        else:
            out[target] = df[source].astype("string").str.strip()
    if "index_year" not in out.columns and "index_date" in out.columns:
        out["index_year"] = parse_date_series(out["index_date"]).dt.year.astype("Int64")
    if "index_year" not in out.columns:
        raise RuntimeError("Input must contain index_year or index_date so train/test splits can be assigned.")
    return out.sort_values("ClientId").reset_index(drop=True), {**column_map, **optional_map}


def read_feature_columns_from_metadata(path: Path) -> list[str]:
    df = read_table(path)
    for candidate in ["feature", "feature_name", "predictor", "parName"]:
        if candidate in df.columns:
            values = [str(value).strip() for value in df[candidate].tolist() if str(value).strip()]
            if values:
                return values
    raise RuntimeError(f"Feature metadata does not contain a recognized feature column: {path}")


def resolve_feature_columns(df: pd.DataFrame, configured: str, metadata_path: str) -> list[str]:
    if configured.strip():
        return [token.strip() for token in configured.split(",") if token.strip()]
    if metadata_path.strip():
        return read_feature_columns_from_metadata(Path(metadata_path))
    id_col = pick_first(list(df.columns), CLIENT_ID_CANDIDATES, "feature input")
    excluded = set(NON_FEATURE_COLUMNS)
    excluded.add(id_col.lower())
    return [column for column in df.columns if column.lower() not in excluded]


def standardize_feature_input(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    client_col = pick_first(list(df.columns), CLIENT_ID_CANDIDATES, "feature input")
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise RuntimeError(f"Feature input is missing required column(s): {missing}")
    out = pd.DataFrame({"ClientId": normalize_id_series(df[client_col])})
    for column in feature_columns:
        out[column] = df[column]
    out = out[out["ClientId"].notna() & (out["ClientId"].astype(str).str.len() > 0)].copy()
    return out.sort_values("ClientId").reset_index(drop=True)


def ensure_unique_client_ids(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    duplicated = df["ClientId"].duplicated(keep=False)
    if bool(duplicated.any()):
        path = out_dir / f"duplicate_clientids_{label}.csv"
        df.loc[duplicated, ["ClientId"]].groupby("ClientId").size().reset_index(name="n_rows").to_csv(path, index=False)
        raise RuntimeError(f"{label} contains duplicate ClientId rows; wrote {path}")


def required_survival_mask(df: pd.DataFrame) -> pd.Series:
    ca = pd.to_numeric(df["CA"], errors="coerce")
    followup = pd.to_numeric(df["followup_years"], errors="coerce")
    event = pd.to_numeric(df["event_death"], errors="coerce")
    index_year = pd.to_numeric(df["index_year"], errors="coerce")
    client = df["ClientId"].astype("string")
    sex = df["sex"].astype("string")
    return (
        client.notna()
        & (client.str.strip().str.len() > 0)
        & ca.notna()
        & np.isfinite(ca)
        & sex.notna()
        & (sex.str.strip().str.len() > 0)
        & followup.notna()
        & np.isfinite(followup)
        & (followup >= 0)
        & event.notna()
        & event.isin([0, 1])
        & index_year.notna()
    )


def assign_split_labels(df: pd.DataFrame, train_start: int, train_end: int, test_start: int, test_end: int) -> pd.DataFrame:
    out = df.copy()
    index_year = pd.to_numeric(out["index_year"], errors="coerce")
    out["split_label"] = pd.NA
    out.loc[(index_year >= train_start) & (index_year <= train_end), "split_label"] = "train"
    out.loc[(index_year >= test_start) & (index_year <= test_end), "split_label"] = "test"
    return out


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        split_df = df[df["split_label"] == split]
        for sex_value in ["ALL"] + sorted(str(value) for value in split_df["sex"].dropna().unique()):
            subset = split_df if sex_value == "ALL" else split_df[split_df["sex"].astype(str) == sex_value]
            event = pd.to_numeric(subset["event_death"], errors="coerce")
            ca = pd.to_numeric(subset["CA"], errors="coerce")
            rows.append(
                {
                    "split_label": split,
                    "sex": sex_value,
                    "n_rows": int(len(subset)),
                    "n_events": int((event == 1).sum()),
                    "event_rate": float(event.mean()) if len(subset) else np.nan,
                    "ca_mean": float(ca.mean()) if len(subset) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_synthetic_smoke_inputs(out_dir: Path, n_rows: int, seed: int) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    smoke_dir = out_dir / "smoke_input"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = [f"SMOKE_F{i:02d}" for i in range(1, 25)]
    client_ids = [f"smoke_{i:05d}" for i in range(n_rows)]
    sex = np.where(np.arange(n_rows) % 2 == 0, "F", "M")
    index_years = np.resize(np.arange(2015, 2026), n_rows)
    rng.shuffle(index_years)
    ca = rng.normal(78, 9, n_rows).clip(45, 102)
    x = rng.normal(0, 1, (n_rows, len(feature_columns)))
    linear = 0.04 * (ca - 78) + 0.2 * x[:, 0] - 0.15 * x[:, 1] + np.where(sex == "M", 0.08, 0.0)
    event_time = rng.exponential(scale=np.exp(-linear) * 4.0)
    censor_time = rng.uniform(0.8, 5.0, n_rows)
    followup = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    for year_mask in [(index_years >= 2015) & (index_years <= 2021), (index_years >= 2022) & (index_years <= 2025)]:
        idx = np.where(year_mask)[0]
        if len(idx) >= 4 and int(event[idx].sum()) == 0:
            event[idx[:2]] = 1
            followup[idx[:2]] = np.minimum(followup[idx[:2]], 0.8)
        if len(idx) >= 4 and int((event[idx] == 0).sum()) == 0:
            event[idx[2:4]] = 0
            followup[idx[2:4]] = 3.6

    survival_df = pd.DataFrame(
        {
            "ClientId": client_ids,
            "CA": ca,
            "sex": sex,
            "followup_years": followup,
            "event_death": event,
            "index_year": index_years,
            "index_date": [f"{int(year)}-01-01" for year in index_years],
        }
    )
    feature_df = pd.DataFrame({"ClientId": client_ids})
    for i, column in enumerate(feature_columns):
        feature_df[column] = x[:, i]
    metadata_df = pd.DataFrame({"feature_order": np.arange(1, 25), "feature": feature_columns})

    survival_path = smoke_dir / "synthetic_survival.csv"
    feature_path = smoke_dir / "synthetic_features.csv"
    metadata_path = smoke_dir / "synthetic_feature_metadata.csv"
    survival_df.to_csv(survival_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    return {"survival_path": survival_path, "feature_path": feature_path, "metadata_path": metadata_path}


def prepare_modeling_tables(args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    synthetic_paths: dict[str, Path] = {}
    modeling_table_path = Path(args.modeling_table_path) if args.modeling_table_path.strip() else None
    survival_path = Path(args.survival_path) if args.survival_path.strip() else None
    feature_path = Path(args.feature_table_path) if args.feature_table_path.strip() else None
    feature_metadata_path = args.feature_metadata_path

    if args.make_synthetic_smoke_data:
        synthetic_paths = build_synthetic_smoke_inputs(out_dir, args.smoke_n_rows, args.smoke_seed)
        survival_path = synthetic_paths["survival_path"]
        feature_path = synthetic_paths["feature_path"]
        feature_metadata_path = str(synthetic_paths["metadata_path"])
        modeling_table_path = None

    if modeling_table_path is not None:
        log(f"- Modeling table: {modeling_table_path}")
        raw_modeling = read_table(modeling_table_path)
        feature_columns = resolve_feature_columns(raw_modeling, args.feature_columns, feature_metadata_path)
        standardized_survival, source_columns = standardize_survival_input(raw_modeling)
        feature_df = standardize_feature_input(raw_modeling, feature_columns)
        ensure_unique_client_ids(standardized_survival, out_dir, "modeling_survival_columns")
        ensure_unique_client_ids(feature_df, out_dir, "modeling_feature_columns")
        modeling_df = standardized_survival.merge(feature_df, on="ClientId", how="inner", validate="one_to_one")
        reference_df = standardized_survival.copy()
        input_mode = "modeling_table"
    else:
        if survival_path is None or feature_path is None:
            raise RuntimeError("Provide --modeling-table-path, or provide both --survival-path and --feature-table-path.")
        log(f"- Survival table: {survival_path}")
        log(f"- Feature table: {feature_path}")
        raw_survival = read_table(survival_path)
        raw_features = read_table(feature_path)
        reference_df, source_columns = standardize_survival_input(raw_survival)
        feature_columns = resolve_feature_columns(raw_features, args.feature_columns, feature_metadata_path)
        feature_df = standardize_feature_input(raw_features, feature_columns)
        ensure_unique_client_ids(reference_df, out_dir, "survival_input")
        ensure_unique_client_ids(feature_df, out_dir, "feature_input")
        overlap = sorted(set(reference_df.columns).intersection(feature_columns))
        if overlap:
            raise RuntimeError(f"Feature columns overlap with survival columns: {overlap}")
        modeling_df = reference_df.merge(feature_df, on="ClientId", how="inner", validate="one_to_one")
        input_mode = "survival_plus_features"

    if len(feature_columns) != 24 and not args.allow_non_24_feature_count:
        raise RuntimeError(f"Expected 24 clinical feature columns, resolved {len(feature_columns)}.")
    leakage = [column for column in feature_columns if column.lower() in NON_FEATURE_COLUMNS]
    if leakage:
        raise RuntimeError(f"Potential leakage/non-feature columns were selected: {leakage}")

    allowed_sexes = parse_allowed_sexes(args.allowed_sexes)
    for name, df in {"modeling": modeling_df, "reference": reference_df}.items():
        ensure_unique_client_ids(df, out_dir, f"{name}_table")

    def filter_and_split(df: pd.DataFrame) -> pd.DataFrame:
        filtered = df[required_survival_mask(df)].copy()
        filtered["event_death"] = pd.to_numeric(filtered["event_death"], errors="coerce").astype(int)
        filtered["index_year"] = pd.to_numeric(filtered["index_year"], errors="coerce").astype(int)
        filtered = assign_split_labels(filtered, args.train_start, args.train_end, args.test_start, args.test_end)
        filtered = filtered[filtered["split_label"].notna()].copy()
        if allowed_sexes:
            filtered = filtered[filtered["sex"].isin(allowed_sexes)].copy()
        return filtered.sort_values("ClientId").reset_index(drop=True)

    modeling_df = filter_and_split(modeling_df)
    reference_df = filter_and_split(reference_df)
    if modeling_df.empty:
        raise RuntimeError("No feature-complete modeling rows remain after filtering and split assignment.")
    if reference_df.empty:
        raise RuntimeError("No reference rows remain after filtering and split assignment.")

    missing_features = modeling_df[feature_columns].isna().sum()
    if int(missing_features.sum()) > 0:
        path = out_dir / "feature_missingness.csv"
        missing_features.rename("n_missing").reset_index(names="feature").to_csv(path, index=False)
        raise RuntimeError(f"Feature-complete modeling table has missing feature values; wrote {path}")

    build_split_summary(modeling_df).to_csv(out_dir / "modeling_split_summary.csv", index=False)
    build_split_summary(reference_df).to_csv(out_dir / "reference_split_summary.csv", index=False)
    pd.DataFrame({"feature_order": np.arange(1, len(feature_columns) + 1), "feature": feature_columns}).to_csv(
        out_dir / "feature_columns_used.csv", index=False
    )

    diagnostics = {
        "input_mode": input_mode,
        "source_columns": source_columns,
        "synthetic_paths": synthetic_paths,
        "n_modeling_rows": int(len(modeling_df)),
        "n_reference_rows": int(len(reference_df)),
        "feature_columns": feature_columns,
    }
    return modeling_df, reference_df, feature_columns, diagnostics


def is_numeric_without_missing(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.notna().all() and np.isfinite(numeric.to_numpy(dtype=float)).all())


def clean_categorical_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isna() | (cleaned == ""), "__MISSING__")
    return cleaned.astype(str)


@dataclass
class PreprocessorSpec:
    predictor_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    scale_numeric: bool
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_levels: dict[str, list[str]]
    output_columns: list[str]
    zero_variance_columns: list[str]
    unknown_level: str = "__OTHER__"


def fit_preprocessor(
    df: pd.DataFrame,
    predictor_columns: list[str],
    *,
    scale_numeric: bool,
    sd_eps: float,
    max_categorical_levels: int,
) -> PreprocessorSpec:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    numeric_means: dict[str, float] = {}
    numeric_stds: dict[str, float] = {}
    categorical_levels: dict[str, list[str]] = {}
    output_columns: list[str] = []
    zero_variance_columns: list[str] = []
    unknown_level = "__OTHER__"

    for column in predictor_columns:
        values = df[column]
        is_numeric = column != "sex" and is_numeric_without_missing(values)
        if column == "CA":
            is_numeric = True
        if is_numeric:
            numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(numeric).all():
                raise RuntimeError(f"Numeric predictor {column} contains non-finite training values.")
            mean = float(np.mean(numeric))
            std = float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
            if not math.isfinite(std) or std < sd_eps:
                std = 1.0
                zero_variance_columns.append(column)
            numeric_columns.append(column)
            numeric_means[column] = mean if scale_numeric else 0.0
            numeric_stds[column] = std if scale_numeric else 1.0
            output_columns.append(column)
        else:
            levels = sorted(clean_categorical_series(values).unique().tolist())
            if unknown_level not in levels:
                levels.append(unknown_level)
            if len(levels) > max_categorical_levels:
                raise RuntimeError(f"Categorical predictor {column} has {len(levels)} levels.")
            categorical_columns.append(column)
            categorical_levels[column] = levels
            output_columns.extend([f"{column}__{level}" for level in levels])

    return PreprocessorSpec(
        predictor_columns=predictor_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        scale_numeric=scale_numeric,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_levels=categorical_levels,
        output_columns=output_columns,
        zero_variance_columns=zero_variance_columns,
        unknown_level=unknown_level,
    )


def transform_preprocessor(df: pd.DataFrame, spec: PreprocessorSpec) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column in spec.numeric_columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError(f"Numeric predictor {column} contains non-finite values during transform.")
        columns.append((values - spec.numeric_means[column]) / spec.numeric_stds[column])
    for column in spec.categorical_columns:
        levels = spec.categorical_levels[column]
        values = clean_categorical_series(df[column])
        values = values.where(values.isin(set(levels)), spec.unknown_level)
        for level in levels:
            columns.append((values == level).to_numpy(dtype=float))
    if not columns:
        return np.empty((len(df), 0), dtype=np.float32)
    matrix = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Preprocessed design matrix contains non-finite values.")
    return matrix


def qcut_labels(series: pd.Series, q: int, prefix: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series([f"{prefix}_missing"] * len(series), index=series.index, dtype="object")
    finite = numeric.notna() & np.isfinite(numeric)
    unique_n = int(numeric[finite].nunique())
    if unique_n < 2:
        out.loc[finite] = f"{prefix}_all"
        return out
    try:
        labels = pd.qcut(numeric[finite], q=min(q, unique_n), labels=False, duplicates="drop")
    except ValueError:
        out.loc[finite] = f"{prefix}_all"
        return out
    out.loc[finite] = prefix + "_" + labels.astype("Int64").astype(str)
    return out


def allocate_stratified_counts(labels: pd.Series, max_rows: int) -> pd.Series | None:
    counts = labels.value_counts(sort=False)
    if len(counts) > max_rows:
        return None
    raw = counts * (max_rows / counts.sum())
    allocation = np.floor(raw).astype(int)
    allocation[(counts > 0) & (allocation == 0)] = 1
    while int(allocation.sum()) > max_rows:
        removable = allocation[allocation > 1].sort_values(ascending=False)
        if removable.empty:
            return None
        allocation[removable.index[0]] -= 1
    remaining = max_rows - int(allocation.sum())
    if remaining > 0:
        fractional = (raw - np.floor(raw)).sort_values(ascending=False, kind="mergesort")
        for label in fractional.index:
            if remaining <= 0:
                break
            capacity = int(counts[label] - allocation[label])
            take = min(capacity, remaining)
            if take > 0:
                allocation[label] += take
                remaining -= take
    return allocation.astype(int)


def make_classifier_strata(df: pd.DataFrame, label_col: str) -> pd.Series:
    labels = "y_" + pd.to_numeric(df[label_col], errors="coerce").astype("Int64").astype(str)
    labels = labels + "|sex_" + clean_categorical_series(df["sex"])
    labels = labels + "|" + qcut_labels(df["CA"], 5, "ca").astype(str)
    if "index_year" in df.columns:
        years = pd.to_numeric(df["index_year"], errors="coerce").astype("Int64").astype(str)
        labels = labels + "|year_" + years
    return labels.astype(str)


def subsample_classifier_context(
    train_df: pd.DataFrame,
    label_col: str,
    max_context_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if max_context_rows <= 1:
        raise RuntimeError("--max-context-rows must be greater than 1.")
    if len(train_df) <= max_context_rows:
        return train_df.copy(), {
            "used": False,
            "method": "not_needed",
            "n_before": int(len(train_df)),
            "n_after": int(len(train_df)),
            "max_context_rows": int(max_context_rows),
        }

    base = train_df.sort_values("ClientId").reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    base["_random_order"] = rng.random(len(base))
    labels = make_classifier_strata(base, label_col)
    allocation = allocate_stratified_counts(labels, max_context_rows)
    if allocation is None:
        selected = base.nsmallest(max_context_rows, "_random_order").copy()
        method = "deterministic_random_subsample_after_sparse_strata"
    else:
        tmp = base.copy()
        tmp["_stratum"] = labels.to_numpy()
        parts: list[pd.DataFrame] = []
        for stratum, group in tmp.groupby("_stratum", sort=True):
            take_n = int(allocation.get(stratum, 0))
            if take_n > 0:
                parts.append(group.nsmallest(take_n, "_random_order"))
        selected = pd.concat(parts, ignore_index=True) if parts else base.nsmallest(max_context_rows, "_random_order")
        method = "deterministic_stratified_subsample"

    selected = selected.drop_duplicates(subset=["ClientId"], keep="first").copy()
    protected_ids: set[str] = set()
    needed_classes = set(pd.to_numeric(base[label_col], errors="coerce").dropna().astype(int).tolist())
    observed_classes = set(pd.to_numeric(selected[label_col], errors="coerce").dropna().astype(int).tolist())
    for class_value in sorted(needed_classes - observed_classes):
        candidate = base[pd.to_numeric(base[label_col], errors="coerce").astype("Int64") == class_value].nsmallest(1, "_random_order")
        if not candidate.empty:
            selected = pd.concat([selected, candidate], ignore_index=True)
            protected_ids.add(str(candidate.iloc[0]["ClientId"]))

    needed_sexes = set(base["sex"].astype(str).dropna().tolist())
    observed_sexes = set(selected["sex"].astype(str).dropna().tolist())
    for sex_value in sorted(needed_sexes - observed_sexes):
        candidate = base[base["sex"].astype(str) == sex_value].nsmallest(1, "_random_order")
        if not candidate.empty:
            selected = pd.concat([selected, candidate], ignore_index=True)
            protected_ids.add(str(candidate.iloc[0]["ClientId"]))

    if len(selected) > max_context_rows:
        selected["_protected"] = selected["ClientId"].astype(str).isin(protected_ids)
        drop_n = len(selected) - max_context_rows
        drop_idx = selected[~selected["_protected"]].sort_values("_random_order", ascending=False).head(drop_n).index
        selected = selected.drop(index=drop_idx).drop(columns=["_protected"])
    if len(selected) > max_context_rows:
        raise RuntimeError("Could not preserve class/sex coverage within --max-context-rows.")

    return selected.drop(columns=[c for c in ["_random_order", "_stratum"] if c in selected.columns]).reset_index(drop=True), {
        "used": True,
        "method": method,
        "n_before": int(len(train_df)),
        "n_after": int(len(selected)),
        "max_context_rows": int(max_context_rows),
        "seed": int(seed),
        "n_strata": int(labels.nunique()),
    }


def make_timebin_labels(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    followup = pd.to_numeric(df["followup_years"], errors="coerce")
    event = pd.to_numeric(df["event_death"], errors="coerce")
    valid = followup.notna() & np.isfinite(followup.to_numpy(dtype=float)) & event.isin([0, 1])

    class1 = valid & (event == 1) & (followup <= 1.0)
    class2 = valid & (event == 1) & (followup > 1.0) & (followup <= 2.0)
    class3 = valid & (event == 1) & (followup > 2.0) & (followup <= 3.0)
    class0 = valid & (followup >= 3.0) & ~((event == 1) & (followup <= 3.0))
    eligible = class0 | class1 | class2 | class3
    censored_before_3y = valid & (event == 0) & (followup < 3.0)

    label = pd.Series(np.nan, index=df.index, dtype=float)
    label.loc[class0] = 0.0
    label.loc[class1] = 1.0
    label.loc[class2] = 2.0
    label.loc[class3] = 3.0
    return label, eligible, {
        "n_input_rows": int(len(df)),
        "n_eligible_labeled": int(eligible.sum()),
        "n_class_0_event_free_through_3y": int(class0.sum()),
        "n_class_1_death_le_1y": int(class1.sum()),
        "n_class_2_death_gt_1_le_2y": int(class2.sum()),
        "n_class_3_death_gt_2_le_3y": int(class3.sum()),
        "n_excluded_censored_before_3y": int(censored_before_3y.sum()),
        "n_excluded_missing_or_invalid": int((~valid).sum()),
    }


def class_count_dict(values: pd.Series | np.ndarray) -> dict[str, int]:
    counts = pd.Series(values).dropna().astype(int).value_counts().sort_index()
    return {str(int(key)): int(value) for key, value in counts.items()}


def assign_oof_folds(train: pd.DataFrame, labels: pd.Series, eligible: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    folds = np.full(len(train), -1, dtype=int)
    rng = np.random.default_rng(seed)
    categories = pd.Series("unlabeled_censored_before_3y", index=train.index, dtype="string")
    categories.loc[eligible] = "class_" + labels.loc[eligible].astype(int).astype(str)
    categories = categories + "|sex_" + train["sex"].astype(str)
    for _, index_values in train.assign(_fold_category=categories).groupby("_fold_category", sort=True).groups.items():
        positions = train.index.get_indexer(np.asarray(list(index_values)))
        rng.shuffle(positions)
        folds[positions] = np.arange(len(positions), dtype=int) % n_folds
    if bool((folds < 0).any()):
        raise RuntimeError("Out-of-fold assignment failed for one or more rows.")
    return folds


def import_tabpfn_classifier_or_raise() -> Any:
    try:
        from tabpfn import TabPFNClassifier  # type: ignore

        return TabPFNClassifier
    except ImportError as exc:
        raise RuntimeError("TabPFNClassifier is required; install dependencies with `uv sync` or `pip install -e .`.") from exc


def build_tabpfn_classifier(TabPFNClassifier: Any, args: argparse.Namespace, seed: int) -> tuple[Any, dict[str, Any]]:
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(TabPFNClassifier).parameters
    except (TypeError, ValueError):
        parameters = {}

    if args.tabpfn_device != "auto" and "device" in parameters:
        kwargs["device"] = args.tabpfn_device
    if "n_estimators" in parameters:
        kwargs["n_estimators"] = args.tabpfn_n_estimators
    if "auto_scale_n_estimators" in parameters:
        kwargs["auto_scale_n_estimators"] = args.tabpfn_auto_scale_n_estimators
    if "ignore_pretraining_limits" in parameters:
        kwargs["ignore_pretraining_limits"] = args.tabpfn_ignore_pretraining_limits
    if "random_state" in parameters:
        kwargs["random_state"] = seed
    elif "seed" in parameters:
        kwargs["seed"] = seed

    try:
        from tabpfn.constants import ModelVersion  # type: ignore

        model_version = ModelVersion(args.tabpfn_model_version)
        model = TabPFNClassifier.create_default_for_version(model_version, **kwargs)
        return model, {"model_version": model_version.value, "init_kwargs": kwargs, "init_status": "explicit_model_version"}
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Installed tabpfn does not support requested model version {args.tabpfn_model_version!r}; "
            "the release was tested with tabpfn==8.0.7."
        ) from exc


def predict_timebin_proba(model: Any, x: np.ndarray, expected_classes: list[int] = TIMEBIN_CLASSES) -> tuple[np.ndarray, dict[str, Any]]:
    proba = np.asarray(model.predict_proba(x), dtype=float)
    if proba.ndim != 2:
        raise RuntimeError("TabPFNClassifier returned a non-matrix probability output.")
    classes_attr = np.asarray(getattr(model, "classes_", []))
    if classes_attr.size:
        model_classes = [int(float(value)) for value in classes_attr.tolist()]
    elif proba.shape[1] == len(expected_classes):
        model_classes = list(expected_classes)
    else:
        raise RuntimeError(f"Could not infer probability columns from predict_proba shape {proba.shape}.")
    out = np.zeros((proba.shape[0], len(expected_classes)), dtype=float)
    missing: list[int] = []
    for out_idx, class_value in enumerate(expected_classes):
        if class_value in model_classes:
            out[:, out_idx] = proba[:, model_classes.index(class_value)]
        else:
            missing.append(class_value)
    if not np.isfinite(out).all():
        raise RuntimeError("TabPFNClassifier returned non-finite predicted probabilities.")
    return np.clip(out, 0.0, 1.0), {
        "model_classes": model_classes,
        "missing_expected_classes": missing,
        "n_predicted": int(proba.shape[0]),
    }


def fit_predict_tabpfn_pair(
    train_eligible: pd.DataFrame,
    reference_train_eligible: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
    seed: int,
    TabPFNClassifier: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    class_counts = class_count_dict(train_eligible["y_timebin"])
    ref_class_counts = class_count_dict(reference_train_eligible["y_timebin"])
    if len(class_counts) < 2 or len(ref_class_counts) < 2:
        raise RuntimeError(f"TabPFN training requires at least two classes; full={class_counts}, reference={ref_class_counts}")

    context, subsampling = subsample_classifier_context(train_eligible, "y_timebin", args.max_context_rows, seed)
    ref_context, ref_subsampling = subsample_classifier_context(
        reference_train_eligible,
        "y_timebin",
        args.max_context_rows,
        seed + 97_001,
    )

    full_predictors = ["CA", "sex", *feature_columns]
    ref_predictors = ["CA", "sex"]
    full_preprocessor = fit_preprocessor(
        context,
        full_predictors,
        scale_numeric=args.scale_numeric,
        sd_eps=args.sd_eps,
        max_categorical_levels=args.max_categorical_levels,
    )
    ref_preprocessor = fit_preprocessor(
        ref_context,
        ref_predictors,
        scale_numeric=args.scale_numeric,
        sd_eps=args.sd_eps,
        max_categorical_levels=args.max_categorical_levels,
    )
    full_model, full_init = build_tabpfn_classifier(TabPFNClassifier, args, seed)
    ref_model, ref_init = build_tabpfn_classifier(TabPFNClassifier, args, seed)
    full_model.fit(transform_preprocessor(context, full_preprocessor), context["y_timebin"].astype(int).to_numpy())
    ref_model.fit(transform_preprocessor(ref_context, ref_preprocessor), ref_context["y_timebin"].astype(int).to_numpy())

    full_proba, full_proba_diag = predict_timebin_proba(full_model, transform_preprocessor(predict_df, full_preprocessor))
    ref_proba, ref_proba_diag = predict_timebin_proba(ref_model, transform_preprocessor(predict_df, ref_preprocessor))
    return full_proba, ref_proba, {
        "seed": int(seed),
        "n_context": int(len(context)),
        "n_reference_context": int(len(ref_context)),
        "training_class_counts": class_counts,
        "reference_training_class_counts": ref_class_counts,
        "context_class_counts": class_count_dict(context["y_timebin"]),
        "reference_context_class_counts": class_count_dict(ref_context["y_timebin"]),
        "subsampling": subsampling,
        "reference_subsampling": ref_subsampling,
        "full_tabpfn_init": full_init,
        "reference_tabpfn_init": ref_init,
        "full_preprocessor": asdict(full_preprocessor),
        "reference_preprocessor": asdict(ref_preprocessor),
        "full_probability_diagnostics": full_proba_diag,
        "reference_probability_diagnostics": ref_proba_diag,
    }


def resolve_mrdt_values(args: argparse.Namespace, expected_sexes: list[str]) -> tuple[dict[str, float], dict[str, Any]]:
    values: dict[str, float] = {}
    notes: list[str] = []
    if args.mrdt_path.strip():
        table = read_table(Path(args.mrdt_path))
        sex_col = pick_first(list(table.columns), ["sex", "Sex", "SEX"], "MRDT table")
        value_col = pick_first(list(table.columns), ["MRDT", "mrdt", "doubling_time", "risk_doubling_time"], "MRDT table")
        for _, row in table.iterrows():
            sex = normalize_sex_value(row[sex_col])
            value = pd.to_numeric(pd.Series([row[value_col]]), errors="coerce").iloc[0]
            if sex in expected_sexes and pd.notna(value) and math.isfinite(float(value)) and float(value) > 0:
                values[sex] = float(value)
        notes.append(f"Read MRDT values from {args.mrdt_path}")
    if args.mrdt_f.strip():
        values["F"] = float(args.mrdt_f)
    if args.mrdt_m.strip():
        values["M"] = float(args.mrdt_m)
    missing = sorted(set(expected_sexes) - set(values))
    if missing:
        raise RuntimeError(
            "Missing positive MRDT value(s) for sex label(s): "
            + ", ".join(missing)
            + ". Provide --mrdt-path or explicit --mrdt-f/--mrdt-m."
        )
    return values, {"mrdt_values_by_sex": values, "notes": notes}


def add_tgage_columns(df: pd.DataFrame, mrdt_values: dict[str, float], eps: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    ca = pd.to_numeric(out["CA"], errors="coerce").to_numpy(dtype=float)
    mrdt = out["sex"].astype(str).map(mrdt_values).to_numpy(dtype=float)
    if not np.isfinite(mrdt).all():
        raise RuntimeError("MRDT mapping produced missing or non-finite values.")

    clipping: dict[str, int] = {}
    for prefix in ["full", "ref"]:
        p1 = pd.to_numeric(out[f"P_{prefix}_class1"], errors="coerce").to_numpy(dtype=float)
        p2 = pd.to_numeric(out[f"P_{prefix}_class2"], errors="coerce").to_numpy(dtype=float)
        p3 = pd.to_numeric(out[f"P_{prefix}_class3"], errors="coerce").to_numpy(dtype=float)
        out[f"F_{prefix}_1y"] = p1
        out[f"F_{prefix}_2y"] = p1 + p2
        out[f"F_{prefix}_3y"] = p1 + p2 + p3
        for horizon in [1, 2, 3]:
            values = pd.to_numeric(out[f"F_{prefix}_{horizon}y"], errors="coerce").to_numpy(dtype=float)
            clipped = np.isfinite(values) & ((values < eps) | (values > 1.0 - eps))
            clipping[f"n_clipped_F_{prefix}_{horizon}y"] = int(clipped.sum())
            out[f"F_{prefix}_{horizon}y_clipped"] = np.clip(values, eps, 1.0 - eps)
            out[f"F_{prefix}_{horizon}y_clipped_flag"] = clipped

    for horizon in [1, 2, 3]:
        full_risk = out[f"F_full_{horizon}y_clipped"].to_numpy(dtype=float)
        ref_risk = out[f"F_ref_{horizon}y_clipped"].to_numpy(dtype=float)
        ratio = full_risk / ref_risk
        acceleration = np.log(ratio) / math.log(2.0) * mrdt
        out[f"risk_ratio_{horizon}y"] = ratio
        out[f"TGAgeAccel_{horizon}y"] = acceleration
        out[f"TGAge_{horizon}y"] = ca + acceleration
    out["MRDT_sex"] = mrdt
    return out, {
        "eps": float(eps),
        "probability_clipping_counts": clipping,
        "n_any_probability_clipped_rows": int(
            np.logical_or.reduce([out[column].to_numpy(dtype=bool) for column in out.columns if column.endswith("_clipped_flag")]).sum()
        ),
    }


def score_cohort(
    modeling_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_columns: list[str],
    mrdt_values: dict[str, float],
    args: argparse.Namespace,
    TabPFNClassifier: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = modeling_df[modeling_df["split_label"] == "train"].copy().reset_index(drop=True)
    test = modeling_df[modeling_df["split_label"] == "test"].copy().reset_index(drop=True)
    reference_train = reference_df[reference_df["split_label"] == "train"].copy().reset_index(drop=True)

    train_labels, train_eligible_mask, label_diag = make_timebin_labels(train)
    train["y_timebin"] = train_labels
    train["_timebin_eligible"] = train_eligible_mask
    eligible_train = train[train["_timebin_eligible"]].copy().reset_index(drop=True)

    ref_labels, ref_eligible_mask, ref_label_diag = make_timebin_labels(reference_train)
    reference_train["y_timebin"] = ref_labels
    reference_train["_timebin_eligible"] = ref_eligible_mask
    reference_eligible = reference_train[reference_train["_timebin_eligible"]].copy().reset_index(drop=True)

    class_counts = class_count_dict(eligible_train["y_timebin"])
    ref_class_counts = class_count_dict(reference_eligible["y_timebin"])
    if len(class_counts) < 2 or len(ref_class_counts) < 2:
        raise RuntimeError(f"Training requires at least two time-bin classes; full={class_counts}, reference={ref_class_counts}")

    min_class_count = min(int(value) for value in class_counts.values())
    effective_folds = min(args.n_folds, min_class_count, len(train))
    if effective_folds < 2:
        raise RuntimeError(f"Out-of-fold scoring needs at least two folds; class_counts={class_counts}")
    train["_oof_fold"] = assign_oof_folds(train, train_labels, train_eligible_mask, effective_folds, args.seed)

    for prefix in ["full", "ref"]:
        for class_value in TIMEBIN_CLASSES:
            train[f"P_{prefix}_class{class_value}"] = np.nan
            test[f"P_{prefix}_class{class_value}"] = np.nan

    fold_diagnostics: list[dict[str, Any]] = []
    for fold_id in range(effective_folds):
        fit_rows = train[(train["_oof_fold"] != fold_id) & train["_timebin_eligible"]].copy().reset_index(drop=True)
        predict_rows = train[train["_oof_fold"] == fold_id].copy()
        heldout_ids = set(predict_rows["ClientId"].astype(str))
        ref_fit_rows = reference_eligible[~reference_eligible["ClientId"].astype(str).isin(heldout_ids)].copy().reset_index(drop=True)
        full_proba, ref_proba, fold_diag = fit_predict_tabpfn_pair(
            fit_rows,
            ref_fit_rows,
            predict_rows,
            feature_columns,
            args,
            args.seed + fold_id + 1,
            TabPFNClassifier,
        )
        for col_idx, class_value in enumerate(TIMEBIN_CLASSES):
            train.loc[predict_rows.index, f"P_full_class{class_value}"] = full_proba[:, col_idx]
            train.loc[predict_rows.index, f"P_ref_class{class_value}"] = ref_proba[:, col_idx]
        fold_diag["fold_id"] = int(fold_id)
        fold_diag["n_fit_rows"] = int(len(fit_rows))
        fold_diag["n_heldout_rows_scored"] = int(len(predict_rows))
        fold_diagnostics.append(fold_diag)
        log(f"- OOF fold {fold_id + 1}/{effective_folds}: fit={len(fit_rows)}; scored={len(predict_rows)}")

    full_test, ref_test, final_diag = fit_predict_tabpfn_pair(
        eligible_train,
        reference_eligible,
        test,
        feature_columns,
        args,
        args.seed + 100_000,
        TabPFNClassifier,
    )
    for col_idx, class_value in enumerate(TIMEBIN_CLASSES):
        test[f"P_full_class{class_value}"] = full_test[:, col_idx]
        test[f"P_ref_class{class_value}"] = ref_test[:, col_idx]

    predictions = pd.concat([train, test], ignore_index=True)
    predictions, tgage_diag = add_tgage_columns(predictions, mrdt_values, args.eps)
    return predictions, {
        "label_diagnostics_train": label_diag,
        "reference_label_diagnostics_train": ref_label_diag,
        "eligible_train_class_counts": class_counts,
        "reference_eligible_train_class_counts": ref_class_counts,
        "effective_folds": int(effective_folds),
        "folds": fold_diagnostics,
        "final_refit": final_diag,
        "tgage_transform": tgage_diag,
        "leakage_control": {
            "train_scores": "Out-of-fold predictions; each row is scored by models that did not train on that row.",
            "test_scores": "Final models are fit on eligible training rows only and applied to held-out test rows.",
        },
    }


class FenwickTree:
    def __init__(self, size: int) -> None:
        self.size = size
        self.tree = np.zeros(size + 1, dtype=float)

    def add(self, index_zero_based: int, delta: float) -> None:
        i = index_zero_based + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def prefix_sum(self, index_zero_based: int) -> float:
        if index_zero_based < 0:
            return 0.0
        i = min(index_zero_based + 1, self.size)
        out = 0.0
        while i > 0:
            out += self.tree[i]
            i -= i & -i
        return out

    def total(self) -> float:
        return self.prefix_sum(self.size - 1)

    def range_sum(self, left_zero_based: int, right_zero_based: int) -> float:
        if right_zero_based < left_zero_based:
            return 0.0
        return self.prefix_sum(right_zero_based) - self.prefix_sum(left_zero_based - 1)


def compute_c_index(time: np.ndarray, event: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(time) & np.isfinite(score) & np.isin(event, [0, 1])
    if int(valid.sum()) < 2:
        return {"c_index": np.nan, "c_index_status": "skipped_too_few_rows", "c_index_evaluable_pairs": 0}
    time_v = time[valid].astype(float)
    event_v = event[valid].astype(int)
    score_v = score[valid].astype(float)
    if int((event_v == 1).sum()) == 0:
        return {"c_index": np.nan, "c_index_status": "skipped_no_events", "c_index_evaluable_pairs": 0}

    unique_scores = np.unique(score_v)
    ranks = np.searchsorted(unique_scores, score_v)
    tree = FenwickTree(len(unique_scores))
    for rank in ranks:
        tree.add(int(rank), 1.0)

    order = np.argsort(time_v, kind="mergesort")
    concordant = 0.0
    comparable = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        current_time = time_v[order[i]]
        while j < len(order) and time_v[order[j]] == current_time:
            j += 1
        group = order[i:j]
        for idx in group:
            tree.add(int(ranks[idx]), -1.0)
        later_total = tree.total()
        if later_total > 0:
            for idx in group:
                if event_v[idx] != 1:
                    continue
                rank = int(ranks[idx])
                less = tree.prefix_sum(rank - 1)
                equal = tree.range_sum(rank, rank)
                concordant += less + 0.5 * equal
                comparable += later_total
        i = j

    if comparable <= 0:
        return {"c_index": np.nan, "c_index_status": "skipped_no_comparable_pairs", "c_index_evaluable_pairs": 0}
    return {"c_index": concordant / comparable, "c_index_status": "evaluated", "c_index_evaluable_pairs": comparable}


def rank_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(int)
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    sum_pos_ranks = float(ranks[: len(pos)].sum())
    return (sum_pos_ranks - (len(pos) * (len(pos) + 1) / 2.0)) / (len(pos) * len(neg))


def horizon_auc(df: pd.DataFrame, score_column: str, horizon: float) -> dict[str, Any]:
    time = pd.to_numeric(df["followup_years"], errors="coerce").to_numpy(dtype=float)
    event = pd.to_numeric(df["event_death"], errors="coerce").to_numpy(dtype=float)
    score = pd.to_numeric(df[score_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(score) & np.isin(event, [0, 1])
    case = valid & (event == 1) & (time <= horizon)
    control = valid & ~case & (time >= horizon)
    included = case | control
    if int(case.sum()) == 0 or int(control.sum()) == 0:
        return {
            "auc": math.nan,
            "status": "skipped_need_cases_and_controls",
            "eligible_n": int(included.sum()),
            "case_n": int(case.sum()),
            "control_n": int(control.sum()),
        }
    return {
        "auc": rank_auc(case[included].astype(int), score[included].astype(float)),
        "status": "evaluated",
        "eligible_n": int(included.sum()),
        "case_n": int(case.sum()),
        "control_n": int(control.sum()),
    }


def evaluate_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        split_df = predictions[predictions["split_label"] == split].copy()
        for sex in sorted(str(value) for value in split_df["sex"].dropna().unique()):
            subset = split_df[split_df["sex"].astype(str) == sex].copy()
            if subset.empty:
                continue
            for score_column in ["TGAge_3y", "TGAgeAccel_3y", "CA"]:
                time = pd.to_numeric(subset["followup_years"], errors="coerce").to_numpy(dtype=float)
                event = pd.to_numeric(subset["event_death"], errors="coerce").to_numpy(dtype=int)
                score = pd.to_numeric(subset[score_column], errors="coerce").to_numpy(dtype=float)
                cindex = compute_c_index(time, event, score)
                row = {
                    "split_label": split,
                    "sex": sex,
                    "score_column": score_column,
                    "n_rows": int(len(subset)),
                    "n_events": int((event == 1).sum()),
                    **cindex,
                }
                for horizon in [1.0, 2.0, 3.0]:
                    auc = horizon_auc(subset, score_column, horizon)
                    row[f"AUC_{horizon:g}y"] = auc["auc"]
                    row[f"AUC_{horizon:g}y_status"] = auc["status"]
                rows.append(row)
    return pd.DataFrame(rows)


def dependency_status(out_dir: Path) -> dict[str, Any]:
    status = {
        "python": sys.version,
        "numpy": package_version("numpy"),
        "pandas": package_version("pandas"),
        "polars": package_version("polars"),
        "tabpfn_available": module_available("tabpfn"),
        "tabpfn": package_version("tabpfn"),
        "scikit_learn": package_version("scikit-learn"),
    }
    write_json(out_dir / "dependency_check.json", status)
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TGAge multiclass TabPFN biological-age implementation.")
    parser.add_argument("--modeling-table-path", default=env_value("MODELING_TABLE_PATH", ""))
    parser.add_argument("--survival-path", default=env_value("SURVIVAL_PATH", ""))
    parser.add_argument("--feature-table-path", default=env_value("FEATURE_TABLE_PATH", ""))
    parser.add_argument("--feature-metadata-path", default=env_value("FEATURE_METADATA_PATH", ""))
    parser.add_argument("--feature-columns", default=env_value("FEATURE_COLUMNS", ""))
    parser.add_argument("--out-dir", default=env_value("OUT_DIR", str(DEFAULT_OUT_DIR)))
    parser.add_argument("--train-start", type=int, default=env_int("TRAIN_START", 2015))
    parser.add_argument("--train-end", type=int, default=env_int("TRAIN_END", 2021))
    parser.add_argument("--test-start", type=int, default=env_int("TEST_START", 2022))
    parser.add_argument("--test-end", type=int, default=env_int("TEST_END", 2025))
    parser.add_argument("--allowed-sexes", default=env_value("ALLOWED_SEXES", "F,M"))
    parser.add_argument("--max-context-rows", type=int, default=env_int("MAX_CONTEXT_ROWS", 10000))
    parser.add_argument("--n-folds", type=int, default=env_int("N_FOLDS", 5))
    parser.add_argument("--seed", type=int, default=env_int("SEED", 42))
    parser.add_argument("--tabpfn-model-version", choices=["v2", "v2.5", "v2.6", "v3"], default=env_value("TABPFN_MODEL_VERSION", "v3"))
    parser.add_argument("--tabpfn-device", default=env_value("TABPFN_DEVICE", "auto"))
    parser.add_argument("--tabpfn-n-estimators", type=int, default=env_int("TABPFN_N_ESTIMATORS", 8))
    parser.add_argument("--tabpfn-auto-scale-n-estimators", action=argparse.BooleanOptionalAction, default=env_flag("TABPFN_AUTO_SCALE_N_ESTIMATORS", False))
    parser.add_argument("--tabpfn-ignore-pretraining-limits", action=argparse.BooleanOptionalAction, default=env_flag("TABPFN_IGNORE_PRETRAINING_LIMITS", True))
    parser.add_argument("--scale-numeric", action=argparse.BooleanOptionalAction, default=env_flag("SCALE_NUMERIC", True))
    parser.add_argument("--sd-eps", type=float, default=float(env_value("SD_EPS", "1e-12")))
    parser.add_argument("--max-categorical-levels", type=int, default=env_int("MAX_CATEGORICAL_LEVELS", 100))
    parser.add_argument("--eps", type=float, default=env_float("EPS", DEFAULT_EPS))
    parser.add_argument("--mrdt-path", default=env_value("MRDT_PATH", ""))
    parser.add_argument("--mrdt-f", default=env_value("MRDT_F", ""))
    parser.add_argument("--mrdt-m", default=env_value("MRDT_M", ""))
    parser.add_argument("--allow-non-24-feature-count", action="store_true", default=env_flag("ALLOW_NON_24_FEATURE_COUNT", False))
    parser.add_argument("--prepare-only", action="store_true", default=env_flag("PREPARE_ONLY", False))
    parser.add_argument("--make-synthetic-smoke-data", action="store_true", default=env_flag("MAKE_SYNTHETIC_SMOKE_DATA", False))
    parser.add_argument("--smoke-n-rows", type=int, default=env_int("SMOKE_N_ROWS", 160))
    parser.add_argument("--smoke-seed", type=int, default=env_int("SMOKE_SEED", 20260516))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.train_start > args.train_end or args.test_start > args.test_end or args.train_end >= args.test_start:
        raise RuntimeError("Expected train_start <= train_end < test_start <= test_end.")
    if args.n_folds < 2:
        raise RuntimeError("--n-folds must be at least 2.")
    if args.tabpfn_n_estimators <= 0:
        raise RuntimeError("--tabpfn-n-estimators must be positive.")
    if not (0.0 < args.eps < 0.5):
        raise RuntimeError("--eps must be between 0 and 0.5.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log("TGAge multiclass TabPFN run started")
    log(f"- Output directory: {out_dir}")
    log(f"- Train years: {args.train_start}-{args.train_end}; test years: {args.test_start}-{args.test_end}")

    dependencies = dependency_status(out_dir)
    modeling_df, reference_df, feature_columns, prep_diag = prepare_modeling_tables(args, out_dir)
    allowed_sexes = parse_allowed_sexes(args.allowed_sexes)
    mrdt_values, mrdt_diag = resolve_mrdt_values(args, allowed_sexes)

    run_config: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "tgage.py",
        "hostname": socket.gethostname(),
        "output_dir": out_dir,
        "train_years": [args.train_start, args.train_end],
        "test_years": [args.test_start, args.test_end],
        "allowed_sexes": allowed_sexes,
        "features_used": feature_columns,
        "full_model_predictors": ["CA", "sex", *feature_columns],
        "reference_model_predictors": ["CA", "sex"],
        "class_labels": {
            "0": "known event-free through 3 years",
            "1": "death <= 1 year",
            "2": "death > 1 and <= 2 years",
            "3": "death > 2 and <= 3 years",
        },
        "eps": float(args.eps),
        "n_folds_requested": int(args.n_folds),
        "max_context_rows": int(args.max_context_rows),
        "tabpfn_model_version": args.tabpfn_model_version,
        "tabpfn_n_estimators": int(args.tabpfn_n_estimators),
        "scale_numeric": bool(args.scale_numeric),
        "dependency_check": dependencies,
        "preparation": prep_diag,
        "mrdt": mrdt_diag,
    }

    if args.prepare_only:
        run_config["run_status"] = "prepare_only_completed"
        write_json(out_dir / "run_config_used.json", run_config)
        log("- Prepare-only mode completed.")
        return

    if not dependencies["tabpfn_available"]:
        raise RuntimeError("tabpfn is not installed; install dependencies before running model training.")
    TabPFNClassifier = import_tabpfn_classifier_or_raise()
    predictions, score_diagnostics = score_cohort(modeling_df, reference_df, feature_columns, mrdt_values, args, TabPFNClassifier)
    metrics = evaluate_scores(predictions)

    predictions_path = out_dir / "tgage_individual_predictions.csv"
    metrics_path = out_dir / "tgage_score_metrics.csv"
    diagnostics_path = out_dir / "diagnostics.json"
    run_config_path = out_dir / "run_config_used.json"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    write_json(diagnostics_path, score_diagnostics)
    run_config.update(
        {
            "run_status": "completed",
            "n_predictions": int(len(predictions)),
            "score_diagnostics": score_diagnostics,
            "output_paths": {
                "individual_predictions": predictions_path,
                "score_metrics": metrics_path,
                "diagnostics": diagnostics_path,
                "run_config_used": run_config_path,
            },
        }
    )
    write_json(run_config_path, run_config)

    log("")
    log("- Outputs written:")
    for path in [predictions_path, metrics_path, diagnostics_path, run_config_path]:
        log(f"  - {path}")
    log("TGAge multiclass TabPFN run completed")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
