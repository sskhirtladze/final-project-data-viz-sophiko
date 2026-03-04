"""
prepare_dashboard_data.py
=========================
Pre-builds data/derived-data/dashboard_country_data.csv for dashboard.py.

Run once (or whenever ES_firm_level.csv is regenerated):
    conda run -n DAP python code/prepare_dashboard_data.py

Inputs:
    data/raw-data/epi2024results.csv          (~121 KB, 180 rows)
    data/derived-data/ES_firm_level.csv       (~121 MB, firm-level)
Output:
    data/derived-data/dashboard_country_data.csv  (~15 KB, 180 rows × 17 cols)
"""

from pathlib import Path
import pandas as pd

REPO_DIR = Path(__file__).resolve().parent.parent
RAW      = REPO_DIR / "data" / "raw-data"
DERIVED  = REPO_DIR / "data" / "derived-data"
OUT_PATH = DERIVED / "dashboard_country_data.csv"

EPI_SUB_COLS = ["ECO.new", "HLT.new", "BDH.new", "CCH.new",
                "AIR.new", "H2O.new", "WMG.new", "AGR.new"]

# 1. EPI base (~180 countries)
epi_raw = pd.read_csv(RAW / "epi2024results.csv")
epi_df  = epi_raw[["iso", "country", "EPI.new"] + EPI_SUB_COLS].rename(columns={
    "iso":     "iso_code",
    "country": "country_name",
    "EPI.new": "epi_score",
    "ECO.new": "epi_eco", "HLT.new": "epi_hlt", "BDH.new": "epi_bdh",
    "CCH.new": "epi_cch", "AIR.new": "epi_air", "H2O.new": "epi_h2o",
    "WMG.new": "epi_wmg", "AGR.new": "epi_agr",
})

# 2. Enterprise Survey aggregated to country level
es  = pd.read_csv(DERIVED / "ES_firm_level.csv", low_memory=False)
grp = (
    es.groupby("country_name", observed=True)
    .agg(
        iso_code_es         = ("Country Code",            "first"),
        co2_monitoring_rate = ("monitors_co2_emissions",  "mean"),
        energy_mgmt_rate    = ("adopt_energy_management", "mean"),
        n_firms             = ("monitors_co2_emissions",  "count"),
        br_env              = ("br_env",                  "first"),
    )
    .reset_index()
)
grp["co2_monitoring_pct"] = grp["co2_monitoring_rate"] * 100
grp["energy_mgmt_pct"]    = grp["energy_mgmt_rate"]    * 100
grp = grp.drop(columns=["co2_monitoring_rate", "energy_mgmt_rate"])

# 3. Left-join: EPI is the base (non-ES countries get NaN for ES fields)
merged = epi_df.merge(
    grp.drop(columns=["country_name"]),
    left_on="iso_code", right_on="iso_code_es", how="left",
).drop(columns=["iso_code_es"])

# 4. Write
DERIVED.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT_PATH, index=False)
print(f"Written: {OUT_PATH}")
print(f"  {len(merged)} rows × {len(merged.columns)} cols  |  "
      f"{OUT_PATH.stat().st_size / 1024:.1f} KB")
print(f"  EPI countries: {merged['epi_score'].notna().sum()}")
print(f"  ES-matched countries: {merged['br_env'].notna().sum()}")
