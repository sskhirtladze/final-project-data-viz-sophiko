# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a data preprocessing pipeline for an Environmental Economics research paper. The `code/preprocessing.py` script is a Python port of three R scripts that merge and clean data from four international datasets:
- **B-Ready indices** (World Bank business regulatory environment, with environmental sub-scores)
- **WDI** (World Bank World Development Indicators)
- **WGI** (Worldwide Governance Indicators)
- **Enterprise Survey** (World Bank firm-level .dta file) + **EPI** (Environmental Performance Index 2024)

> Note: The README and `environment.yml` describe a placeholder "Fire Perimeter Analysis" project from a course template. The actual code is unrelated to that.

## Setup

```bash
conda env create -f environment.yml
conda activate fire_analysis
```

The `environment.yml` is missing several required packages. Install them manually after activating the environment:

```bash
pip install pyreadstat openpyxl scikit-learn
```

## Running the Pipeline

```bash
python code/preprocessing.py
```

This runs the entire pipeline sequentially and writes all outputs to `data/derived-data/`. Paths are resolved relative to the script file, so the script works from any working directory without modification.

## Architecture

`code/preprocessing.py` is a single 562-line sequential ETL script organized in five numbered sections:

**Section 1 — B-Ready Indices (lines 43–123):** Reads all sheets from the B-Ready Excel workbook, merges on `EconomyCode`, filters to environmental variables, sums scores across 8 pillar codes (BE, BL, US, IT, TX, DR, MC, BI), applies PCA (StandardScaler → 3 principal components), and appends regulatory/service/efficiency scores. ISO code corrections are applied via the `ISO_FIX` dict (lines 39, 103).

**Section 2 — Country-Level Controls (lines 127–228):** Pivots WDI and WGI from long to wide format, selects specific indicator codes, merges with EPI scores, log-transforms GDP per capita, and joins B-Ready environmental scores. Output: `country_level_vars`.

**Section 3 — Enterprise Survey Cleaning (lines 231–507):** The most complex section. Loads the Stata `.dta` file, filters to survey years 2022–2025, replaces sentinel missing values (−9 to −5) with `NaN`, recodes binary variables (1/2 → 1/0), applies country-specific logic for B-Ready follow-up/follow-down economies, derives analysis variables (log employment, firm age, exporter status, climate damage share, etc.), harmonizes 11 country names, and renames 19 variables to match expected R labels.

**Section 4 — Missingness Check (lines 511–548):** Prints counts of missing values across 35 analysis variables for data quality diagnostics.

**Section 5 — Individual Outputs (lines 551–565):** Writes three CSV files to `data/derived-data/`:
- `ES_firm_level.csv` — main analysis dataset (firm + country variables merged)
- `country_level_vars.csv`
- `BR_scores_env.csv`

**Section 6 — Combined Dataset (lines 567–590):** Merges `ES_firm_level` with the remaining pillar-level columns from `BR_scores_env` (those not already present) and writes `combined_data.csv` — a single file containing all firm-level, country-level, and B-Ready pillar data.

## Key Implementation Details

- The helper `_col(df, *names)` (line ~240) returns the first column name from `names` that exists in `df`, used to handle variable name differences between B-Ready follow-up/follow-down surveys.
- Country-specific branching logic (Bangladesh, Indonesia, Iraq, Madagascar, Sierra Leone, Central African Republic, Timor-Leste) appears throughout Section 3 because these countries have non-standard variable names or sources in the Enterprise Survey.
- The `ISO_FIX` dict corrects legacy ISO codes (`ROM→ROU`, `TMP→TLS`, `WBG→PSE`) applied after B-Ready merges.
