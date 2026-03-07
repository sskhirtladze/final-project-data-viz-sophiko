[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YlfKWlZ5)
# Environmental Economics — Data Pipeline & Dashboard

This project prepares, analyzes, and visualizes data for an environmental economics research paper examining how business regulatory environments affect firm-level environmental behavior (CO₂ monitoring and energy management adoption).

## Streamlit App

**Live dashboard:** <https://sskhirtladze-final-project-data-viz-sophik-codedashboard-b5qxmv.streamlit.app/>

> **Note for reviewers:** Streamlit Community Cloud apps go to sleep after 24 hours of inactivity. If you see a "wake up" prompt when opening the link, click it and wait a few seconds — this is normal behavior and is not a bug.

The dashboard features an interactive choropleth map of green adoption rates and governance indicators across 40+ countries, plus a country-profile panel with EPI sub-indicator breakdowns and adoption comparisons.

## Setup

```bash
conda env create -f environment.yml
conda activate econ_data
```

Or use an existing conda environment and install dependencies:

```bash
pip install -r requirements.txt
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

Run the scripts in order:

### 1. Preprocessing

Merges and cleans all raw data sources into analysis-ready CSVs.

```bash
python code/preprocessing.py
```

### 2. Analysis

Runs PCA on B-Ready indices, logit regressions with clustered SEs, random forest feature importance, and produces all plots.

```bash
python code/analysis.py
```

### 3. Dashboard data (pre-build)

Aggregates the 121 MB firm-level CSV down to a 16 KB country-level file for fast dashboard startup. Re-run whenever `ES_firm_level.csv` is regenerated.

```bash
python code/prepare_dashboard_data.py
```

### 4. Dashboard

Interactive choropleth map + country profile panel. Requires steps 1–3 first.

```bash
streamlit run code/dashboard.py
```

### 5. Quarto writeup

Renders `final_project.qmd` (runs the analysis pipeline internally):

```bash
quarto render final_project.qmd
```

Produces `final_project.html` and `final_project.pdf`, and saves charts to `plots/`.

## Outputs

All outputs are written to `data/derived-data/`:

**Preprocessing (`preprocessing.py`):**

| File | Description |
|---|---|
| `ES_firm_level.csv` | Cleaned firm-level data merged with country controls |
| `country_level_vars.csv` | Country-level controls (WDI + WGI + EPI + B-Ready) |
| `BR_scores_env.csv` | B-Ready environmental sub-scores and PCA components |
| `combined_data.csv` | Single file with all datasets merged |

**Analysis (`analysis.py`):**

| File | Description |
|---|---|
| `rf_importance.csv` | Random forest variable importance scores |
| `rf_variable_importance.png/pdf` | Feature importance bar chart |
| `marginal_effects_co.png/pdf` | Marginal effects forest plot — CO₂ monitoring |
| `marginal_effects_em.png/pdf` | Marginal effects forest plot — energy management |
| `marginal_effects_ctrl_co.png/pdf` | Marginal effects (controls only) — CO₂ monitoring |
| `marginal_effects_ctrl_em.png/pdf` | Marginal effects (controls only) — energy management |
| `epi_vs_country_fe_co.png/pdf` | EPI score vs country fixed effects — CO₂ monitoring |
| `epi_vs_country_fe_em.png/pdf` | EPI score vs country fixed effects — energy management |
| `bready_vs_adoption.png/pdf` | B-Ready score vs adoption rates scatter |
| `adoption_heatmap.png/pdf` | Adoption rates by sector/income heatmap |
| `altair_*.html/png` | Interactive Altair charts (HTML + optional PNG) |
| `marginal_effects_ctrl_co/em.csv` | Marginal effects data for QMD plots |
| `epi_vs_country_fe_co/em.csv` | Country FE vs EPI data for QMD plots |
| `adoption_heatmap.csv` | Aggregated heatmap data for QMD plots |

**Quarto writeup (`final_project.qmd`):**

| File | Description |
|---|---|
| `final_project.html` | Self-contained HTML with 5 interactive Altair plots |
| `final_project.pdf` | PDF version (≤3 pages) |
| `plots/*.html/png` | Saved chart files |

**Dashboard (`prepare_dashboard_data.py`):**

| File | Description |
|---|---|
| `dashboard_country_data.csv` | Pre-aggregated country-level data for dashboard (~16 KB, 183 rows) |

## Project Structure

```
data/
  raw-data/                  # Source data (not tracked in git)
  derived-data/              # Pipeline outputs (not tracked in git)
code/
  preprocessing.py           # ETL pipeline: B-Ready → WDI/WGI/EPI → Enterprise Survey
  analysis.py                # Regressions, random forest, plots, and intermediate CSVs
  prepare_dashboard_data.py  # Pre-builds dashboard_country_data.csv
  dashboard.py               # Streamlit interactive dashboard (deployed from code/)
streamlit-app/
  dashboard.py               # Copy of dashboard for Streamlit Community Cloud
  prepare_dashboard_data.py  # Copy of data prep script
plots/                       # Altair chart outputs (HTML + PNG) from final_project.qmd
final_project.qmd            # Quarto writeup with 5 Altair plots (HTML + PDF output)
requirements.txt             # Full pip dependency list
```
