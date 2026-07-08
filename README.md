# TGAge

This repository contains an anonymous core implementation of **TGAge**, a tabular foundation model based biological age score.

The release is intentionally limited to the model construction code. It does not include private data, derived cohorts, fitted model artifacts, logs, or manuscript-identifying material.

## Method Summary

TGAge trains two multiclass TabPFN classifiers on fixed 3-year mortality time bins:

- a full model using chronological age, sex, and selected clinical features;
- a reference model using chronological age and sex only.

For each individual, the cumulative mortality probabilities at 1, 2, and 3 years are converted to a risk ratio between the full and reference models. TGAge acceleration is then:

```text
TGAgeAccel_H = log2(F_full_H / F_ref_H) * MRDT_sex
TGAge_H      = chronological_age + TGAgeAccel_H
```

Training rows are scored out of fold. Held-out test rows are scored by models fit on the eligible training rows only.

## Inputs

Use either:

- `--modeling-table-path`: one CSV/parquet table containing ID, age, sex, survival labels, split date/year, and feature columns; or
- `--survival-path` plus `--feature-table-path`: separate survival/cohort and feature tables joined by `ClientId`.

Required standardized fields, with common aliases accepted:

- `ClientId`
- `CA`
- `sex`
- `followup_years`
- `event_death`
- `index_year` or `index_date`

Feature columns can be supplied with `--feature-columns F1,F2,...` or through `--feature-metadata-path`, whose feature-name column may be `feature`, `feature_name`, `predictor`, or `parName`.

Sex-specific mortality-rate doubling time values are required through `--mrdt-f/--mrdt-m` or `--mrdt-path`.

## Usage

Install dependencies:

```bash
uv sync
```

Run a preparation-only smoke check:

```bash
uv run python tgage.py \
  --make-synthetic-smoke-data \
  --prepare-only \
  --mrdt-f 8 \
  --mrdt-m 8
```

Run on analytic data:

```bash
uv run python tgage.py \
  --survival-path data/survival.csv \
  --feature-table-path data/features.csv \
  --feature-metadata-path data/selected_features.csv \
  --mrdt-f 8.0 \
  --mrdt-m 8.0 \
  --out-dir outputs/tgage
```

Main outputs:

- `tgage_individual_predictions.csv`
- `tgage_score_metrics.csv`
- `diagnostics.json`
- `run_config_used.json`

## Data Availability

The study data are private and are not distributed with this anonymous code release. The `.gitignore` is configured to exclude local data, outputs, logs, checkpoints, and manuscript files.
