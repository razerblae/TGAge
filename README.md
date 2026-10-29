# TGAge

This repository contains the core implementation of **TGAge**, a tabular-foundation-model-based second-generation biological aging clock.

This release includes model construction code. Study data and fitted model artifacts are not included as the data are private and regulated under the HIPAA rules.

## Method Summary

TGAge trains two multiclass TabPFN classifiers over a 3-year horizon:

- a full model using chronological age, sex, and selected clinical features;
- a reference model using chronological age and sex only.

Labels depend only on `followup_years`, for both deaths and censored encounters: class 1 is ≤1 year, class 2 is (1, 2], class 3 is (2, 3], and class 0 is >3 years. Code class 0 corresponds to manuscript class `k = 4`.

The cumulative probabilities of follow-up ending (death or censoring) by 1, 2, and 3 years give the full/reference probability ratio. TGAge acceleration is:

```text
TGAgeAccel_H = log2(F_full_H / F_ref_H) * MRDT_sex
TGAge_H      = chronological_age + TGAgeAccel_H
```

All valid training rows are eligible. Training rows are scored out of fold; held-out test rows are scored after a final fit on training rows. Contexts exceeding `--max-context-rows` (default 10,000) are sampled proportionally by class and sex, preserving empirical proportions up to integer rounding.

## Inputs

Prepare de-identified CSV tables with these columns:

- `subject_id`
- `chronological_age`
- `sex`
- `followup_years`
- `event_death`
- `index_year`
- selected feature columns, supplied explicitly with `--feature-columns`

Use `--*-column` arguments for different column names. The reference table needs the same columns except clinical features.

Sex-specific mortality-rate doubling time values are supplied as `--mrdt-values`, for example `F=8,M=8`.

## Usage

Install dependencies:

```bash
uv sync
```

Run on de-identified analytic tables:

```bash
uv run python tgage.py \
  --modeling-table data/anonymized_modeling_table.csv \
  --reference-table data/anonymized_reference_table.csv \
  --feature-columns feature_01,feature_02,feature_03 \
  --mrdt-values F=8,M=8 \
  --out-dir outputs/tgage
```

Main outputs:

- `tgage_predictions.csv`
- `diagnostics.json`
- `run_config.json`

## Data Availability

Study data are private and are not distributed with this code release.
