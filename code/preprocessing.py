"""
Data preparation pipeline for the Environmental Economics paper.

Python port of the following R scripts (run in this order):
  1. BReady Indices.R        → BR_scores_env, BR_scores_all
  2. country_level_controls.R → country_level_vars  (WDI + WGI + EPI + B-Ready)
  3. Firm_Level_Survey.R     → ES_firm_level  (cleaned firm data, all regressions)

Outputs (written to Data/Outputs/):
  - ES_firm_level.csv       : main analysis dataset, ready for modelling
  - country_level_vars.csv  : country-level controls
  - BR_scores_env.csv       : B-Ready environmental sub-scores + PCA

Required packages:
  pip install pandas numpy pyreadstat openpyxl scikit-learn
  (pyreadstat is pulled in automatically by pandas.read_stata)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 0.  PATHS — resolved relative to this file so the script runs from anywhere
# ─────────────────────────────────────────────────────────────────────────────
REPO_DIR   = Path(__file__).resolve().parent.parent
RAW_DIR    = REPO_DIR / "data" / "raw-data"

BREADY_DIR = RAW_DIR / "01_Scored economies"
WDI_DIR    = RAW_DIR / "P_Data_Extract_From_World_Development_Indicators"
WGI_DIR    = RAW_DIR / "P_Data_Extract_From_Worldwide_Governance_Indicators"
EPI_PATH   = RAW_DIR / "epi2024results.csv"
ES_PATH    = RAW_DIR / "New_Comprehensive_October_6_2025.dta"
OUTPUT_DIR = REPO_DIR / "data" / "derived-data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ISO code corrections used across B-Ready datasets
ISO_FIX = {"ROM": "ROU", "TMP": "TLS", "WBG": "PSE"}


# ═════════════════════════════════════════════════════════════════════════════
# 1.  B-READY INDICES   (BReady Indices.R)
# ═════════════════════════════════════════════════════════════════════════════
print("=== 1. B-Ready Indices ===")

# 1a. Read every sheet in the scores workbook and merge all on EconomyCode ----
xl_path     = BREADY_DIR / "01_B-READY-2024-PILLAR-TOPIC-SCORES-2024_Final Data.xlsx"
sheet_names = pd.ExcelFile(xl_path).sheet_names

sheets = {}
for s in sheet_names:
    df = pd.read_excel(xl_path, sheet_name=s)
    # Each sheet may spell EconomyName differently — normalise it
    name_cols = [c for c in df.columns if "EconomyName" in c]
    if name_cols:
        df = df.rename(columns={name_cols[0]: "EconomyName"})
    sheets[s] = df

merged_by_id = sheets[sheet_names[0]].copy()
for s in sheet_names[1:]:
    # Drop columns already in merged_by_id (except the join key) so pandas 3+
    # doesn't raise MergeError about duplicate suffix columns.
    already = set(merged_by_id.columns) - {"EconomyCode"}
    right = sheets[s].drop(columns=[c for c in sheets[s].columns if c in already])
    merged_by_id = merged_by_id.merge(right, on="EconomyCode", how="outer")

# 1b. Keep only the variables listed in EnviroVars.xlsx ----------------------
env_codes = pd.read_excel(BREADY_DIR / "EnviroVars.xlsx")["Code"].tolist()
keep_cols = (
    ["EconomyName", "EconomyCode"]
    + [v for v in env_codes if v in merged_by_id.columns]
)
BR_scores_env = merged_by_id[keep_cols].copy()

# 1c. br_env = row sum across all numeric env columns -------------------------
env_num_cols = BR_scores_env.select_dtypes("number").columns.tolist()
BR_scores_env["br_env"] = BR_scores_env[env_num_cols].sum(axis=1, skipna=True)

# 1d. Rename and fix ISO codes ------------------------------------------------
BR_scores_env = BR_scores_env.rename(columns={"EconomyCode": "country_code"})
BR_scores_env["country_code"] = BR_scores_env["country_code"].replace(ISO_FIX)

# 1e. Pillar prefix sums (BE, BL, US, IT, TX, DR, MC, BI) --------------------
PREFIXES = ["BE", "BL", "US", "IT", "TX", "DR", "MC", "BI"]
for p in PREFIXES:
    p_cols = [c for c in BR_scores_env.columns if c.startswith(p) and c != p]
    if p_cols:
        BR_scores_env[p] = BR_scores_env[p_cols].sum(axis=1, skipna=True)

# 1f. PCA on the 8 pillar sums ------------------------------------------------
pca_input  = BR_scores_env[PREFIXES].dropna()
scaler     = StandardScaler()
pca        = PCA()
pca_scores = pca.fit_transform(scaler.fit_transform(pca_input))

pc_names = [f"PC{i + 1}" for i in range(pca_scores.shape[1])]
pca_df   = pd.DataFrame(pca_scores, columns=pc_names, index=pca_input.index)
BR_scores_env = BR_scores_env.join(pca_df)

ev = pca.explained_variance_
explained = pca.explained_variance_ratio_
print(f"  Eigenvalues > 1 : PCs {[i+1 for i, e in enumerate(ev) if e > 1]}")
print(f"  Cumulative var (PC1–3): {np.cumsum(explained)[:3].round(3).tolist()}")

# 1g. Merge BR_AllScores for reg_fr / pub_ser / op_eff -----------------------
BR_AllScores = pd.read_excel(BREADY_DIR / "BR_AllScores.xlsx")
BR_AllScores = BR_AllScores.rename(columns={"Economy Code": "country_code"})
BR_AllScores["country_code"] = BR_AllScores["country_code"].replace(ISO_FIX)

BR_scores_env = BR_scores_env.merge(
    BR_AllScores[["Economy", "reg_fr", "pub_ser", "op_eff"]],
    left_on="EconomyName", right_on="Economy", how="left",
).drop(columns="Economy")

# Standardise Côte d'Ivoire name to match Enterprise Survey
BR_scores_env["EconomyName"] = BR_scores_env["EconomyName"].replace(
    {"Côte d'Ivoire": "Cote d'Ivoire"}
)
BR_scores_env = BR_scores_env.rename(columns={"EconomyName": "country_name"})
print(f"  BR_scores_env shape: {BR_scores_env.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 2.  COUNTRY-LEVEL CONTROLS   (country_level_controls.R)
#
#     Reads from local CSV files (no internet needed).
#     Variables use 2023 values from WDI and WGI exports.
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 2. Country-level controls ===")

# ── 2a. WDI ------------------------------------------------------------------
wdi_raw = pd.read_csv(WDI_DIR / "wdi_data.csv")
wdi_raw["2023 [YR2023]"] = pd.to_numeric(wdi_raw["2023 [YR2023]"], errors="coerce")

wdi_wide = (
    wdi_raw[["Country Name", "Country Code", "Series Code", "2023 [YR2023]"]]
    .pivot_table(
        index=["Country Name", "Country Code"],
        columns="Series Code",
        values="2023 [YR2023]",
        aggfunc="first",
    )
    .reset_index()
)
wdi_wide.columns.name = None
wdi_wide = wdi_wide.rename(columns={
    "NY.GDP.PCAP.PP.CD": "gdp_per_capita_ppp",
    "NE.TRD.GNFS.ZS":    "trade_gdp_share",
    "NV.IND.TOTL.ZS":    "industry_share_gdp",
    "FS.AST.PRVT.GD.ZS": "private_credit_gdp",
    "GB.XPD.RSDV.GD.ZS": "r_d_expenditure_gdp",
    "SE.TER.ENRR":        "tertiary_enrollment_pct",
})

# ── 2b. WGI ------------------------------------------------------------------
wgi_path = WGI_DIR / "d6b0dacf-eee4-491e-845c-1319e9c9909f_Data.csv"
wgi_raw  = pd.read_csv(wgi_path)
wgi_raw["2023 [YR2023]"] = pd.to_numeric(wgi_raw["2023 [YR2023]"], errors="coerce")

wgi_wide = (
    wgi_raw[["Country Name", "Country Code", "Series Code", "2023 [YR2023]"]]
    .pivot_table(
        index=["Country Name", "Country Code"],
        columns="Series Code",
        values="2023 [YR2023]",
        aggfunc="first",
    )
    .reset_index()
)
wgi_wide.columns.name = None
wgi_wide = wgi_wide.rename(columns={
    "CC.EST":     "control_corruption_est",
    "CC.PER.RNK": "control_corruption_percent",
    "GE.EST":     "gov_effectiveness_est",
    "GE.PER.RNK": "gov_effectiveness_percent",
    "PV.EST":     "political_stability_est",
    "PV.PER.RNK": "political_stability_percent",
    "RQ.EST":     "regulatory_quality_est",
    "RQ.PER.RNK": "regulatory_quality_percent",
    "RL.EST":     "rule_of_law_est",
    "RL.PER.RNK": "rule_of_law_percent",
    "VA.EST":     "voice_accountability_est",
    "VA.PER.RNK": "voice_accountability_percent",
})

# ── 2c. Merge WGI + WDI + EPI ------------------------------------------------
country_level_vars = pd.merge(
    wgi_wide, wdi_wide, on="Country Code", how="outer", suffixes=("_wgi", "_wdi")
)

# Reconcile duplicate "Country Name" columns from the two sources
if "Country Name_wgi" in country_level_vars.columns:
    country_level_vars["country_name"] = (
        country_level_vars["Country Name_wgi"]
        .fillna(country_level_vars["Country Name_wdi"])
    )
    country_level_vars = country_level_vars.drop(
        columns=["Country Name_wgi", "Country Name_wdi"], errors="ignore"
    )
else:
    country_level_vars = country_level_vars.rename(
        columns={"Country Name": "country_name"}
    )

epi = (
    pd.read_csv(EPI_PATH)[["iso", "EPI.new"]]
    .rename(columns={"iso": "Country Code"})
)
country_level_vars = country_level_vars.merge(epi, on="Country Code", how="left")

# Log-transform GDP per capita (matches R: mutate(gdp_per_capita_ppp = log(...)))
country_level_vars["gdp_per_capita_ppp"] = np.log(
    country_level_vars["gdp_per_capita_ppp"]
)

# ── 2d. Attach B-Ready environmental scores ----------------------------------
country_level_vars = country_level_vars.merge(
    BR_scores_env[[
        "country_code", "br_env", "reg_fr", "pub_ser", "op_eff",
        "PC1", "PC2", "PC3",
    ]],
    left_on="Country Code", right_on="country_code", how="left",
).drop(columns="country_code", errors="ignore")

print(f"  country_level_vars shape: {country_level_vars.shape}")


# ═════════════════════════════════════════════════════════════════════════════
# 3.  ENTERPRISE SURVEY — FIRM-LEVEL CLEANING   (Firm_Level_Survey.R)
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 3. Enterprise Survey cleaning ===")

# convert_categoricals=False keeps all values as raw numerics
ES = pd.read_stata(ES_PATH, convert_categoricals=False)

# Helper: safely pull a column as float, returning NaN series if absent
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(
        df.get(name, pd.Series(np.nan, index=df.index, name=name)),
        errors="coerce",
    )


# ── 3.1  Filter to survey years 2022–2025 ------------------------------------
ES = ES.dropna(subset=["a14y"])
ES = ES[(ES["a14y"] >= 2022) & (ES["a14y"] <= 2025)].copy()

# ── 3.2  Extract country name and interview year from the `country` string ---
ES["country_name"]   = ES["country"].str.extract(r"^([^0-9]+)")[0].str.strip()
ES["year"]           = ES["country"].str.extract(r"(\d+)")[0].astype("Int64")

# ── 3.3  Recode sentinel missing values (−9 … −5) → NaN ---------------------
SENTINELS = [-9, -8, -7, -6, -5]
num_cols  = ES.select_dtypes(include="number").columns
ES[num_cols] = ES[num_cols].replace(SENTINELS, np.nan)

# ── 3.4  Recode pure 1/2 binary variables → 1/0 (mirrors R's across()) ------
#   Only columns whose entire non-null value set is {1, 2} are recoded.
for col in ES.select_dtypes(include="number").columns:
    uniq = set(ES[col].dropna().unique())
    if uniq <= {1.0, 2.0}:
        ES[col] = ES[col].map({1.0: 1, 2.0: 0}).astype("Int64")

# ── 3.5  BR follow-up / follow-down country flags ----------------------------
BR_FOLLOWUP   = {"Bangladesh", "Indonesia", "Iraq", "Madagascar"}
BR_FOLLOWDOWN = {"Sierra Leone", "Central African Republic"}
# Exception countries always use their original (non-BR) variables even when
# BR_follow_up == 1, unless overridden in each variable's logic below.
EXCEPTIONS    = {"Indonesia", "Timor-Leste", "Madagascar"}

ES["BR_follow_up"]   = ES["country_name"].isin(BR_FOLLOWUP).astype(int)
ES["BR_follow_down"] = ES["country_name"].isin(BR_FOLLOWDOWN).astype(int)

is_exception  = ES["country_name"].isin(EXCEPTIONS)
is_followup   = ES["BR_follow_up"]   == 1
is_followdown = ES["BR_follow_down"] == 1

# ── 3.6  Log employment (l1 / l1_BR) ----------------------------------------
#   Exception countries always use original l1.
#   BR follow-up (non-exception) uses l1_BR.
#   All others use l1.
l1    = _col(ES, "l1")
l1_BR = _col(ES, "l1_BR")

empl = pd.Series(np.nan, index=ES.index)
mask = is_exception  & l1.notna()    & (l1    > 0)
empl[mask] = np.log(l1[mask])
mask = ~is_exception & is_followup   & l1_BR.notna() & (l1_BR > 0)
empl[mask] = np.log(l1_BR[mask])
mask = ~is_exception & ~is_followup  & l1.notna()    & (l1    > 0)
empl[mask] = np.log(l1[mask])
ES["empl"] = empl

# ── 3.7  Firm age and log firm age -------------------------------------------
ES["firm_age"]    = ES["a14y"] - ES["b5"]
ES["ln_firm_age"] = np.where(ES["firm_age"] > 0, np.log(ES["firm_age"]), np.nan)

# ── 3.8  Foreign ownership (b2b: % held by foreigners) -----------------------
b2b = _col(ES, "b2b")
ES["foreign"]      = pd.array(
    np.where(b2b.notna(), (b2b >= 50).astype(int), pd.NA), dtype="Int64"
)
ES["some_foreign"] = pd.array(
    np.where(b2b.notna(), (b2b >= 10).astype(int), pd.NA), dtype="Int64"
)

# ── 3.9  Manager experience squared -----------------------------------------
ES["b7_sq"] = _col(ES, "b7") ** 2

# ── 3.10  Has financing (k82 / k82_BR; 1/2/3 = yes, 4 = no) -----------------
#   After step 3.4, k82 still has values 1/2/3/4 because it is not a pure 1/2
#   binary column, so the general recoding did NOT touch it.
k82    = _col(ES, "k82")
k82_BR = _col(ES, "k82_BR")
src    = pd.Series(
    np.where(is_followup, k82_BR, k82), index=ES.index, dtype="float64"
)
ES["has_financing"] = pd.array(
    np.where(src.isin([1, 2, 3]), 1, np.where(src == 4, 0, pd.NA)),
    dtype="Int64",
)

# ── 3.11  Climate damage (ge3 / ge3_BR) --------------------------------------
ge3    = _col(ES, "ge3")
ge3_BR = _col(ES, "ge3_BR")
ES["ge3"] = pd.array(
    np.where(is_followup & ge3_BR.notna(), ge3_BR,
    np.where(~is_followup & ge3.notna(),   ge3, pd.NA)),
    dtype="Int64",
)

# damage_share: fraction of annual sales lost to extreme weather
d2     = _col(ES, "d2");    d2_BR  = _col(ES, "d2_BR")
ge3a   = _col(ES, "ge3a");  ge3a_BR = _col(ES, "ge3a_BR")

# Use BR variants for follow-up countries, originals otherwise
ge3_src  = np.where(is_followup.values, ge3_BR.values,  ge3.values ).astype(float)
d2_src   = np.where(is_followup.values, d2_BR.values,   d2.values  ).astype(float)
ge3a_src = np.where(is_followup.values, ge3a_BR.values, ge3a.values).astype(float)

damage = np.full(len(ES), np.nan)
damage = np.where(ge3_src == 2, 0.0, damage)
valid  = (ge3_src == 1) & (d2_src > 0) & (ge3a_src >= 0)
damage[valid] = ge3a_src[valid] / d2_src[valid]

ES["damage_share"]     = damage
ES["log_damage_share"] = np.where(damage > 0, np.log(damage), np.nan)

# ── 3.12  Industry sector (d1a1a / d1a1a_BR) ---------------------------------
VALID_INDUSTRY  = {1, 2, 3, 4, 6, 51, 52}
INDUSTRY_LABELS = {
    1: "Manufacturing", 2: "Retail",    3: "Wholesale",
    4: "Construction",  51: "Hotel",    52: "Restaurant", 6: "Services",
}

d1a1a    = _col(ES, "d1a1a")
d1a1a_BR = _col(ES, "d1a1a_BR")

raw_ind = pd.Series(
    np.where(
        ES["country_name"] == "Indonesia", d1a1a.values,
        np.where(is_followup.values, d1a1a_BR.values, d1a1a.values),
    ),
    index=ES.index,
    dtype="float64",
)
raw_ind[~raw_ind.isin(VALID_INDUSTRY)] = np.nan
ES["d1a1a"] = pd.Categorical(
    raw_ind.map(INDUSTRY_LABELS),
    categories=list(INDUSTRY_LABELS.values()),
)

# ── 3.13  Exporter (direct or indirect exports > 10 % of sales) --------------
#   Exception countries always use original d3b/d3c.
#   R logic: exporter=1 if either > 10; exporter=0 if both ≤ 10 (no NaN);
#            exporter=NA if both are NA, or one is NA while the other is ≤ 10.
d3b    = _col(ES, "d3b");    d3c    = _col(ES, "d3c")
d3b_BR = _col(ES, "d3b_BR"); d3c_BR = _col(ES, "d3c_BR")

use_br = ~is_exception & is_followup
src_b  = pd.Series(np.where(use_br, d3b_BR, d3b), index=ES.index, dtype="float64")
src_c  = pd.Series(np.where(use_br, d3c_BR, d3c), index=ES.index, dtype="float64")

# NaN comparisons return False in pandas, so d3b_le10 is False when NaN
d3b_gt10 = (src_b > 10).fillna(False)
d3c_gt10 = (src_c > 10).fillna(False)
d3b_le10 = src_b <= 10          # NaN → False
d3c_le10 = src_c <= 10          # NaN → False

ES["exporter"] = pd.array(
    np.where(d3b_gt10 | d3c_gt10,  1,
    np.where(d3b_le10 & d3c_le10,  0, pd.NA)),
    dtype="Int64",
)

# ── 3.14  Country name harmonisation -----------------------------------------
NAME_FIX = {
    "BurkinaFaso":          "Burkina Faso",
    "Congo":                "Congo, Rep.",
    "CÃ´te d'Ivoire":       "Cote d'Ivoire",
    "DRC":                  "Congo, Dem. Rep.",
    "ElSalvador":           "El Salvador",
    "Gambia":               "Gambia, The",
    "Hong Kong SAR China":  "Hong Kong SAR, China",
    "Korea Republic":       "Korea, Rep.",
    "Taiwan China":         "Taiwan, China",
    "West Bank And Gaza":   "West Bank and Gaza",
}
ES["country_name"] = ES["country_name"].replace(NAME_FIX)

# Recalculate boolean masks after name fix (only EXCEPTIONS set changes in practice)
is_exception  = ES["country_name"].isin(EXCEPTIONS)
is_followup   = ES["BR_follow_up"]   == 1
is_followdown = ES["BR_follow_down"] == 1

# ── 3.15  ge7 — monitors CO2 emissions (ge7 / ge7_BR) -----------------------
ge7    = _col(ES, "ge7")
ge7_BR = _col(ES, "ge7_BR")
ES["ge7"] = pd.array(
    np.where(is_followup & ge7_BR.notna(), ge7_BR,
    np.where(ge7.notna(),                  ge7, pd.NA)),
    dtype="Int64",
)

# ── 3.16  ge8d — adopts energy management (ge8d / ge8d_BR / ge8) ------------
#   BR follow-up           → ge8d_BR
#   BR follow-down         → ge8  (these countries have only ge8, not ge8d)
#   All others             → ge8d
ge8d_BR = _col(ES, "ge8d_BR")
ge8d    = _col(ES, "ge8d")
ge8     = _col(ES, "ge8")

ES["ge8d"] = pd.array(
    np.where(is_followup  & ge8d_BR.notna(),                       ge8d_BR,
    np.where(is_followdown & ge8.notna(),                           ge8,
    np.where(~is_followup & ~is_followdown & ge8d.notna(),         ge8d,
             pd.NA))),
    dtype="Int64",
)

# ── 3.17  j42 — held a government contract (j42 / j42_BR) -------------------
j42    = _col(ES, "j42")
j42_BR = _col(ES, "j42_BR")

use_br_j42 = is_followup | ES["country_name"].isin({"Peru", "Timor-Leste"})
ES["j42"] = pd.array(
    np.where(use_br_j42 & j42_BR.notna(), j42_BR,
    np.where(j42.notna(),                  j42, pd.NA)),
    dtype="Int64",
)

# ── 3.18  e6 — uses foreign technology licence (e6 / e6_BR) -----------------
#   Indonesia always uses original e6 (even when BR_follow_up == 1).
e6    = _col(ES, "e6")
e6_BR = _col(ES, "e6_BR")

use_br_e6 = (is_followup | (ES["country_name"] == "Peru")) & ~(
    ES["country_name"] == "Indonesia"
)
ES["e6"] = np.where(
    ES["country_name"] == "Indonesia",  e6,
    np.where(use_br_e6 & e6_BR.notna(), e6_BR,
    np.where(e6.notna(),                 e6, np.nan))
)

# ── 3.19  Final variable renaming (matches variable labels in R) -------------
RENAME_MAP = {
    "ge7":          "monitors_co2_emissions",
    "ge8d":         "adopt_energy_management",
    "empl":         "num_employees",
    "some_foreign": "has_foreign_ownership",
    "b1":           "legal_status",
    "b4":           "has_female_owner",
    "b3a":          "owner_is_manager",
    "b7":           "manager_experience_years",
    "b7_sq":        "manager_experience_years_sq",
    "b7a":          "manager_is_female",
    "l10":          "has_training_programs",
    "h1":           "introduced_new_products",
    "h5":           "introduced_new_process",
    "h8":           "has_rd_spending",
    "e6":           "uses_foreign_tech_license",
    "c22b":         "has_website_social_media",
    "k6":           "has_bank_account",
    "k7":           "has_overdraft_facility",
    "k21":          "has_external_audit",
    "ge3":          "experienced_climate_damage",
    "d1a1a":        "main_activity_type",
    "b8":           "has_quality_certification",
    "exporter":     "is_exporter",
    "j42":          "has_government_contract",
    "a14y":         "interview_year",
}
ES = ES.rename(columns=RENAME_MAP)

# ── 3.20  Merge country-level controls on country_name ----------------------
ES_firm_level = ES.merge(country_level_vars, on="country_name", how="left")

print(f"  ES_firm_level shape : {ES_firm_level.shape}")
print(f"  Countries           : {ES_firm_level['country_name'].nunique()}")
print(
    f"  Years               : "
    f"{sorted(ES_firm_level['interview_year'].dropna().unique().tolist())}"
)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MISSINGNESS CHECK   (Checks.R)
# ═════════════════════════════════════════════════════════════════════════════
ANALYSIS_VARS = [
    # Outcomes
    "monitors_co2_emissions", "adopt_energy_management",
    # Firm characteristics
    "num_employees", "ln_firm_age", "has_foreign_ownership",
    "legal_status", "has_female_owner", "owner_is_manager",
    "manager_experience_years", "manager_experience_years_sq", "manager_is_female",
    # Firm capabilities
    "has_training_programs", "introduced_new_products", "introduced_new_process",
    "has_rd_spending", "uses_foreign_tech_license",
    # Digital & financial access
    "has_website_social_media", "has_bank_account", "has_overdraft_facility",
    "has_financing", "has_external_audit",
    # Market linkages & sector
    "experienced_climate_damage", "main_activity_type",
    "has_quality_certification", "is_exporter", "has_government_contract",
    # Country-level controls
    "gdp_per_capita_ppp", "trade_gdp_share", "industry_share_gdp",
    "private_credit_gdp", "tertiary_enrollment_pct",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "EPI.new", "br_env", "reg_fr", "pub_ser", "op_eff",
    "PC1", "PC2", "PC3",
    "interview_year",
]

avail = [v for v in ANALYSIS_VARS if v in ES_firm_level.columns]
missing_summary = (
    ES_firm_level[avail]
    .isna()
    .sum()
    .rename("n_missing")
    .sort_values(ascending=False)
)
print("\nMissing value counts (analysis variables):")
print(missing_summary.to_string())


# ═════════════════════════════════════════════════════════════════════════════
# 5.  SAVE INDIVIDUAL OUTPUTS
# ═════════════════════════════════════════════════════════════════════════════
ES_firm_level.to_csv(OUTPUT_DIR / "ES_firm_level.csv",           index=False)
country_level_vars.to_csv(OUTPUT_DIR / "country_level_vars.csv", index=False)
BR_scores_env.to_csv(OUTPUT_DIR / "BR_scores_env.csv",           index=False)

print(f"\nOutputs saved to {OUTPUT_DIR}")
print("  - ES_firm_level.csv")
print("  - country_level_vars.csv")
print("  - BR_scores_env.csv")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  COMBINED DATASET
#
#     Merges ES_firm_level (firm + country controls) with the full set of
#     B-Ready pillar columns from BR_scores_env.  Country-level columns
#     already present in ES_firm_level (br_env, reg_fr, pub_ser, op_eff,
#     PC1-PC3) are kept as-is; the remaining pillar-level columns are added.
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== 6. Combined dataset ===")

# Columns already merged into ES_firm_level via country_level_vars
already_present = {"br_env", "reg_fr", "pub_ser", "op_eff", "PC1", "PC2", "PC3"}

# Extra B-Ready columns to bring in (exclude keys and already-present cols)
br_extra_cols = [
    c for c in BR_scores_env.columns
    if c not in {"country_name", "country_code"} and c not in already_present
]

combined_data = ES_firm_level.merge(
    BR_scores_env[["country_name"] + br_extra_cols],
    on="country_name",
    how="left",
)

combined_data.to_csv(OUTPUT_DIR / "combined_data.csv", index=False)
print(f"  combined_data shape : {combined_data.shape}")
print(f"  Saved to {OUTPUT_DIR / 'combined_data.csv'}")
