[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)
# Environmental Economics — Data Pipeline

This project prepares and merges data for an environmental economics research paper analyzing how business regulatory environments affect firm-level environmental behavior.

## Setup

```bash
conda env create -f environment.yml
conda activate fire_analysis
```

## Data

Raw data files go in `data/raw-data/` (excluded from git due to size):

| File / Folder | Source | Description |
|---|---|---|
| `01_Scored economies/` | World Bank B-Ready 2024 | Business regulatory scores with environmental sub-indices |
| `P_Data_Extract_From_World_Development_Indicators/` | World Bank WDI | Country-level economic indicators (2023) |
| `P_Data_Extract_From_Worldwide_Governance_Indicators/` | World Bank WGI | Governance indicators (2023) |
| `epi2024results.csv` | Yale EPI | Environmental Performance Index 2024 |
| `New_Comprehensive_October_6_2025.dta` | World Bank Enterprise Survey | Firm-level survey data (2022–2025) |

## Usage

```bash
python code/preprocessing.py
```

Outputs are written to `data/derived-data/`:

| File | Description |
|---|---|
| `ES_firm_level.csv` | Cleaned firm-level data merged with country controls |
| `country_level_vars.csv` | Country-level controls (WDI + WGI + EPI + B-Ready) |
| `BR_scores_env.csv` | B-Ready environmental sub-scores and PCA components |
| `combined_data.csv` | Single file with all datasets merged |

## Project Structure

```
data/
  raw-data/        # Source data (not tracked in git)
  derived-data/    # Pipeline outputs (not tracked in git)
code/
  preprocessing.py # Full ETL pipeline (B-Ready → WDI/WGI/EPI → Enterprise Survey)
```
