# TGAge

This repository contains the core implementation of **TGAge**, a tabular-foundation-model-based second-generation biological aging clock.

The release is intentionally limited to the model construction code. It does not include the private data source, derived cohorts, fitted model artifacts, logs, or manuscript-identifying material.

## Method Summary

TGAge trains two multiclass TabPFN classifiers on a chosen prediction horizon (here we use 3 years as an example), discretized into intervals:

- a full model using chronological age, sex, and selected clinical features;
- a reference model using chronological age and sex only.

For each individual, the cumulative mortality probabilities at 1, 2, and 3 years are converted to a risk ratio between the full and reference models. TGAge acceleration is then:

```text
TGAgeAccel_H = log2(F_full_H / F_ref_H) * MRDT_sex
TGAge_H      = chronological_age + TGAgeAccel_H
```

Training rows are scored out of fold. Held-out test rows are scored by models fit on the eligible training rows only.

## Inputs

The script intentionally does not guess source column names. Before running it, prepare an anonymized CSV with public, generic column names:

- `subject_id`
- `chronological_age`
- `sex`
- `followup_years`
- `event_death`
- `index_year`
- selected feature columns, supplied explicitly with `--feature-columns`

If your local analytic table uses different names, pass the corresponding `--*-column` arguments. Do not publish private data dictionaries or source-system column names with this release.

Sex-specific mortality-rate doubling time values are supplied as `--mrdt-values`, for example `F=8,M=8`.

## Usage

Install dependencies:

```bash
uv sync
```

Run on an anonymized analytic table:

```bash
uv run python tgage.py \
  --modeling-table data/anonymized_modeling_table.csv \
  --feature-columns feature_01,feature_02,feature_03 \
  --mrdt-values F=8,M=8 \
  --out-dir outputs/tgage
```

Main outputs:

- `tgage_predictions.csv`
- `diagnostics.json`
- `run_config.json`

## Data Availability

The study data are private and are not distributed with this anonymous code release. The `.gitignore` is configured to exclude local data, outputs, logs, checkpoints, and manuscript files.
