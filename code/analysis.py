"""
Environmental Economics Paper — Python Analysis
================================================
Equivalent pipeline (in order):
  BReady Indices.R  →  build_bready()
  country_level_controls.R  →  build_country_level()
  Firm_Level_Survey.R  →  build_firm_level()  +  fit all regression models
  ML.R  →  run_random_forest()

Required packages:
  pip install pandas numpy pyreadstat openpyxl scikit-learn statsmodels
  pip install pymer4          # needs R + lme4 installed for GLMER models
  pip install matplotlib

Run:
  python analysis.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
try:
    from pymer4.models import Lmer
    PYMER4 = True
except ImportError:
    PYMER4 = False
    print("pymer4 not found — GLMER models will be skipped.\n"
          "Install: pip install pymer4  (also requires R + lme4)")


# ══════════════════════════════════════════════════════════════════════════════
# 0.  PATHS
# ══════════════════════════════════════════════════════════════════════════════
REPO_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = REPO_DIR / "data" / "raw-data"
OUT         = REPO_DIR / "data" / "derived-data"
OUT.mkdir(exist_ok=True)

BREADY_XLSX = RAW_DIR / "01_Scored economies" / "01_B-READY-2024-PILLAR-TOPIC-SCORES-2024_Final Data.xlsx"
ENVIRO_XLSX = RAW_DIR / "01_Scored economies" / "EnviroVars.xlsx"
BR_ALL_XLSX = RAW_DIR / "01_Scored economies" / "BR_AllScores.xlsx"
WDI_CSV     = RAW_DIR / "P_Data_Extract_From_World_Development_Indicators" / "wdi_data.csv"
WGI_CSV     = RAW_DIR / "P_Data_Extract_From_Worldwide_Governance_Indicators" / "d6b0dacf-eee4-491e-845c-1319e9c9909f_Data.csv"
EPI_CSV     = RAW_DIR / "epi2024results.csv"
ES_DTA      = RAW_DIR / "New_Comprehensive_October_6_2025.dta"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  B-READY INDICES  (BReady Indices.R)
# ══════════════════════════════════════════════════════════════════════════════
_ISO_RECODE = {"ROM": "ROU", "TMP": "TLS", "WBG": "PSE"}
_BR_PREFIXES = ["BE", "BL", "US", "IT", "TX", "DR", "MC", "BI"]


def build_bready():
    """Return br_scores_env: per-country B-Ready env sub-scores, br_env,
    PC1..PC8, and pillar scores (reg_fr, pub_ser, op_eff).
    """
    # Merge all Excel sheets on EconomyCode — normalise EconomyName and drop
    # duplicate columns before each merge to stay compatible with pandas 3+.
    xl = pd.ExcelFile(BREADY_XLSX)
    sheets = {}
    for s in xl.sheet_names:
        df = xl.parse(s)
        name_cols = [c for c in df.columns if "EconomyName" in c]
        if name_cols:
            df = df.rename(columns={name_cols[0]: "EconomyName"})
        sheets[s] = df

    merged = sheets[xl.sheet_names[0]].copy()
    for s in xl.sheet_names[1:]:
        already = set(merged.columns) - {"EconomyCode"}
        right   = sheets[s].drop(columns=[c for c in sheets[s].columns if c in already])
        merged  = merged.merge(right, on="EconomyCode", how="outer")

    env_codes = pd.read_excel(ENVIRO_XLSX)["Code"].tolist()
    env_cols  = [c for c in env_codes if c in merged.columns]

    br = (merged[["EconomyName", "EconomyCode"] + env_cols]
          .rename(columns={"EconomyName": "country_name", "EconomyCode": "country_code"})
          .copy())
    br["country_code"] = br["country_code"].replace(_ISO_RECODE)

    # br_env = sum across all numeric columns (matches data_preparation.py line 79-80)
    br["br_env"] = br.select_dtypes("number").sum(axis=1, skipna=True)

    # Prefix sub-scores (e.g. BE, BL, …)
    for p in _BR_PREFIXES:
        cols = [c for c in env_cols if c.startswith(p)]
        br[p] = br[cols].sum(axis=1, skipna=True) if cols else 0.0

    # PCA on prefix sub-scores (mirrors prcomp with center=TRUE, scale.=TRUE)
    pca_input = br[_BR_PREFIXES].apply(pd.to_numeric, errors="coerce")
    valid     = pca_input.dropna()
    scaler    = StandardScaler()
    pca       = PCA(n_components=len(_BR_PREFIXES))
    scores    = pca.fit_transform(scaler.fit_transform(valid))
    pc_names  = [f"PC{i+1}" for i in range(len(_BR_PREFIXES))]

    br_pca = br.copy()
    br_pca.loc[valid.index, pc_names] = scores

    # Print PCA summary
    ev = pca.explained_variance_
    print("\nB-Ready PCA Summary:")
    print(pd.DataFrame({
        "PC": pc_names, "Eigenvalue": ev.round(3),
        "PropVar": (ev / ev.sum()).round(4),
        "CumProp": np.cumsum(ev / ev.sum()).round(4),
    }).to_string(index=False))

    # Merge reg_fr / pub_ser / op_eff from BR_AllScores into br_scores_env directly
    # (matches data_preparation.py lines 109-116)
    br_all = (pd.read_excel(BR_ALL_XLSX)
              .rename(columns={"Economy Code": "country_code", "Economy": "country_name"})
              .assign(country_code=lambda d: d["country_code"].replace(_ISO_RECODE)))
    pillar_cols = [c for c in ["reg_fr", "pub_ser", "op_eff"] if c in br_all.columns]
    br_pca = (br_pca
              .merge(br_all[["country_name"] + pillar_cols], on="country_name", how="left")
              .assign(country_name=lambda d:
                      d["country_name"].replace({"Côte d'Ivoire": "Cote d'Ivoire"})))

    return br_pca


br_scores_env = build_bready()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  COUNTRY-LEVEL CONTROLS  (country_level_controls.R)
# ══════════════════════════════════════════════════════════════════════════════
def _pivot_wb_csv(path, year_col="2023 [YR2023]"):
    """Pivot a World Bank long CSV to wide on Series Code for a single year."""
    df = pd.read_csv(path)
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    wide = (df[["Country Name", "Country Code", "Series Code", year_col]]
            .pivot_table(index=["Country Name", "Country Code"],
                         columns="Series Code", values=year_col, aggfunc="first")
            .reset_index())
    wide.columns.name = None
    return wide


def build_country_level():
    """Return country_level_vars (WDI + WGI + EPI + B-Ready br_env)."""
    wdi = _pivot_wb_csv(WDI_CSV).rename(columns={
        "NY.GDP.PCAP.PP.CD": "gdp_per_capita_ppp",
        "NE.TRD.GNFS.ZS":    "trade_gdp_share",
        "NV.IND.TOTL.ZS":    "industry_share_gdp",
        "FS.AST.PRVT.GD.ZS": "private_credit_gdp",
        "GB.XPD.RSDV.GD.ZS": "r_d_expenditure_gdp",
        "SE.TER.ENRR":        "tertiary_enrollment_pct",
    })

    wgi = _pivot_wb_csv(WGI_CSV).rename(columns={
        "CC.EST": "control_corruption_est",    "CC.PER.RNK": "control_corruption_percent",
        "GE.EST": "gov_effectiveness_est",     "GE.PER.RNK": "gov_effectiveness_percent",
        "PV.EST": "political_stability_est",   "PV.PER.RNK": "political_stability_percent",
        "RQ.EST": "regulatory_quality_est",    "RQ.PER.RNK": "regulatory_quality_percent",
        "RL.EST": "rule_of_law_est",           "RL.PER.RNK": "rule_of_law_percent",
        "VA.EST": "voice_accountability_est",  "VA.PER.RNK": "voice_accountability_percent",
    })

    combined = wgi.merge(wdi, on="Country Code", how="outer",
                         suffixes=("_wgi", "_wdi"))
    if "Country Name_wgi" in combined.columns:
        combined["country_name"] = (
            combined["Country Name_wgi"]
            .fillna(combined.get("Country Name_wdi"))
        )
        combined = combined.drop(
            columns=["Country Name_wgi", "Country Name_wdi"], errors="ignore"
        )
    else:
        combined = combined.rename(columns={"Country Name": "country_name"})

    # EPI — keep original column name "EPI.new" (matches data_preparation.py)
    epi = pd.read_csv(EPI_CSV)[["iso", "EPI.new"]]
    combined = combined.merge(epi, left_on="Country Code", right_on="iso", how="left")

    # Log-transform GDP per capita (mirrors R code)
    combined["gdp_per_capita_ppp"] = np.log(combined["gdp_per_capita_ppp"])

    # Merge B-Ready env score + pillar scores + PC1-PC3 (matches data_preparation.py)
    br_merge_cols = ["country_code", "br_env"]
    for extra in ["reg_fr", "pub_ser", "op_eff", "PC1", "PC2", "PC3"]:
        if extra in br_scores_env.columns:
            br_merge_cols.append(extra)
    combined = combined.merge(
        br_scores_env[br_merge_cols],
        left_on="Country Code", right_on="country_code", how="left"
    ).drop(columns="country_code", errors="ignore")

    return combined


country_level_vars = build_country_level()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIRM-LEVEL SURVEY  (Firm_Level_Survey.R)
# ══════════════════════════════════════════════════════════════════════════════
_NEG_NA       = [-9, -8, -7, -6, -5]
_BR_FOLLOW_UP  = {"Bangladesh", "Indonesia", "Iraq", "Madagascar"}
_BR_FOLLOW_DOWN = {"Sierra Leone", "Central African Republic"}
_EXCEPT_ORIG  = {"Indonesia", "Timor-Leste", "Madagascar"}   # always use original vars

_NAME_FIX = {
    "BurkinaFaso":         "Burkina Faso",
    "Congo":               "Congo, Rep.",
    "CÃ´te d'Ivoire":      "Cote d'Ivoire",
    "DRC":                 "Congo, Dem. Rep.",
    "ElSalvador":          "El Salvador",
    "Gambia":              "Gambia, The",
    "Hong Kong SAR China": "Hong Kong SAR, China",
    "Korea Republic":      "Korea, Rep.",
    "Taiwan China":        "Taiwan, China",
    "West Bank And Gaza":  "West Bank and Gaza",
}

_INDUSTRY_MAP = {1: "Manufacturing", 2: "Retail", 3: "Wholesale",
                 4: "Construction", 51: "Hotel", 52: "Restaurant", 6: "Services"}


def _col(df, name, fallback=np.nan):
    """Safely get a column as a Series, returning NaN Series if absent."""
    return pd.to_numeric(df[name], errors="coerce") if name in df.columns \
        else pd.Series(fallback, index=df.index, dtype=float)


def build_firm_level():
    """Clean Enterprise Survey data and join country-level controls."""
    # numpy 2.3+ / pandas 2.3+: reading a 250k-row .dta at once raises
    # _ArrayMemoryError for large per-column allocations (even ~2 MB).
    # Work-around: read in 50 k-row chunks so per-column arrays stay small.
    _chunks = []
    for _chunk in pd.read_stata(ES_DTA, convert_categoricals=False,
                                chunksize=50_000):
        _chunks.append(_chunk)
    df = pd.concat(_chunks, ignore_index=True)

    # Filter 2022–2025
    df = df[df["a14y"].between(2022, 2025, inclusive="both")].copy()

    # Extract country name and year from the 'country' string variable
    df["country_name"] = df["country"].str.extract(r"^([^0-9]+)")[0].str.strip()
    df["year"]         = df["country"].str.extract(r"(\d+)")[0].astype(float)

    # Recode sentinel negative values → NaN
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].replace(_NEG_NA, np.nan)

    # Binary recode: columns whose non-NA values are only {1, 2} → {1, 0}
    for col in num_cols:
        vals = set(df[col].dropna().unique())
        if vals and vals <= {1.0, 2.0}:
            df[col] = df[col].map({1.0: 1.0, 2.0: 0.0})

    # Follow-up / follow-down flags
    df["BR_follow_up"]   = df["country_name"].isin(_BR_FOLLOW_UP).astype(int)
    df["BR_follow_down"] = df["country_name"].isin(_BR_FOLLOW_DOWN).astype(int)

    is_exc = df["country_name"].isin(_EXCEPT_ORIG)
    is_fu  = df["BR_follow_up"] == 1
    is_fd  = df["BR_follow_down"] == 1

    # ── Log employment ─────────────────────────────────────────────────────────
    l1    = _col(df, "l1")
    l1_br = _col(df, "l1_BR")
    raw_l = np.where(is_exc, l1,
            np.where(is_fu & ~is_exc, l1_br, l1))
    df["empl"] = np.where(raw_l > 0, np.log(np.where(raw_l > 0, raw_l, 1.0)), np.nan)

    # ── Firm age ───────────────────────────────────────────────────────────────
    df["firm_age"]    = df["a14y"] - df["b5"]
    df["ln_firm_age"] = np.where(df["firm_age"] > 0, np.log(df["firm_age"]), np.nan)

    # ── Foreign ownership ──────────────────────────────────────────────────────
    b2b = _col(df, "b2b")
    df["foreign"]      = np.where(b2b.isna(), np.nan, (b2b >= 50).astype(float))
    df["some_foreign"] = np.where(b2b.isna(), np.nan, (b2b >= 10).astype(float))

    # ── Manager experience squared ─────────────────────────────────────────────
    df["b7_sq"] = _col(df, "b7") ** 2

    # ── Has financing ──────────────────────────────────────────────────────────
    k82 = np.where(is_fu, _col(df, "k82_BR"), _col(df, "k82"))
    df["has_financing"] = pd.Series(k82, index=df.index).map(
        lambda v: 1.0 if v in (1, 2, 3) else (0.0 if v == 4 else np.nan))

    # ── Experienced climate damage (ge3) ───────────────────────────────────────
    df["ge3"] = np.where(is_fu & _col(df, "ge3_BR").notna(),
                         _col(df, "ge3_BR"), _col(df, "ge3"))

    # ── Damage share ───────────────────────────────────────────────────────────
    ge3_v  = np.where(is_fu, _col(df, "ge3_BR"),  _col(df, "ge3"))
    ge3a_v = np.where(is_fu, _col(df, "ge3a_BR"), _col(df, "ge3a"))
    d2_v   = np.where(is_fu, _col(df, "d2_BR"),   _col(df, "d2"))
    dmg = np.where(ge3_v == 2, 0.0,
          np.where((ge3_v == 1) & (d2_v > 0) & (ge3a_v >= 0),
                   ge3a_v / np.where(d2_v > 0, d2_v, np.nan), np.nan))
    df["damage_share"]     = dmg
    df["log_damage_share"] = np.where(dmg > 0, np.log(np.where(dmg > 0, dmg, np.nan)), np.nan)

    # ── Industry type (d1a1a) ──────────────────────────────────────────────────
    valid_d1 = set(_INDUSTRY_MAP.keys())
    d1_raw   = np.where(df["country_name"] == "Indonesia", _col(df, "d1a1a"),
               np.where(is_fu, _col(df, "d1a1a_BR"), _col(df, "d1a1a")))
    df["d1a1a"] = pd.Series(d1_raw, index=df.index).map(
        lambda v: _INDUSTRY_MAP.get(int(v)) if pd.notna(v) and int(v) in valid_d1 else np.nan)

    # ── Exporter ───────────────────────────────────────────────────────────────
    use_br_exp = ~is_exc & is_fu
    src_b = pd.Series(np.where(use_br_exp, _col(df, "d3b_BR"), _col(df, "d3b")), index=df.index, dtype="float64")
    src_c = pd.Series(np.where(use_br_exp, _col(df, "d3c_BR"), _col(df, "d3c")), index=df.index, dtype="float64")
    # NaN comparisons return False in pandas — exporter=NA when one is NaN and the other ≤10
    d3b_gt10 = (src_b > 10).fillna(False)
    d3c_gt10 = (src_c > 10).fillna(False)
    d3b_le10 = (src_b <= 10).fillna(False)
    d3c_le10 = (src_c <= 10).fillna(False)
    df["exporter"] = np.where(d3b_gt10 | d3c_gt10, 1.0,
                     np.where(d3b_le10 & d3c_le10, 0.0, np.nan))

    # ── Country name harmonisation ─────────────────────────────────────────────
    df["country_name"] = df["country_name"].replace(_NAME_FIX)

    # ── Outcome: monitors CO2 (ge7) ────────────────────────────────────────────
    df["ge7"] = np.where(is_fu, _col(df, "ge7_BR"), _col(df, "ge7"))

    # ── Outcome: energy management (ge8d) ──────────────────────────────────────
    df["ge8d"] = np.where(is_fu,  _col(df, "ge8d_BR"),
                 np.where(is_fd,  _col(df, "ge8"),
                                  _col(df, "ge8d")))

    # ── Government contract (j42) ──────────────────────────────────────────────
    j42_use_br = is_fu | df["country_name"].isin({"Peru", "Timor-Leste"})
    df["has_government_contract"] = np.where(j42_use_br,
                                             _col(df, "j42_BR"), _col(df, "j42"))

    # ── Foreign technology license (e6) ────────────────────────────────────────
    # Only Indonesia always uses original e6 (matches R and data_preparation.py)
    is_indonesia = df["country_name"] == "Indonesia"
    e6_use_br    = (is_fu | (df["country_name"] == "Peru")) & ~is_indonesia
    df["e6_clean"] = np.where(is_indonesia, _col(df, "e6"),
                     np.where(e6_use_br & _col(df, "e6_BR").notna(), _col(df, "e6_BR"),
                                           _col(df, "e6")))

    # ── Rename to final variable names ─────────────────────────────────────────
    df = df.rename(columns={
        "ge7":        "monitors_co2_emissions",
        "ge8d":       "adopt_energy_management",
        "empl":       "num_employees",
        "some_foreign": "has_foreign_ownership",
        "b1":         "legal_status",
        "b4":         "has_female_owner",
        "b3a":        "owner_is_manager",
        "b7":         "manager_experience_years",
        "b7_sq":      "manager_experience_years_sq",
        "b7a":        "manager_is_female",
        "l10":        "has_training_programs",
        "h1":         "introduced_new_products",
        "h5":         "introduced_new_process",
        "h8":         "has_rd_spending",
        "e6_clean":   "uses_foreign_tech_license",
        "c22b":       "has_website_social_media",
        "k6":         "has_bank_account",
        "k7":         "has_overdraft_facility",
        "k21":        "has_external_audit",
        "ge3":        "experienced_climate_damage",
        "d1a1a":      "main_activity_type",
        "b8":         "has_quality_certification",
        "exporter":   "is_exporter",
        "a14y":       "interview_year",
    })

    # Convert to appropriate dtypes for regression
    df["legal_status"]       = df["legal_status"].astype("category")
    df["main_activity_type"] = df["main_activity_type"].astype("category")
    df["interview_year"]     = df["interview_year"].astype(str)   # treated as factor

    # Join country-level controls
    df = df.merge(country_level_vars, on="country_name", how="left")

    return df


ES_firm_level = build_firm_level()

# Quick NA counts (mirrors sapply(... sum(is.na)))
print("\nNA counts for key variables:")
key_vars = [
    "monitors_co2_emissions", "adopt_energy_management", "num_employees",
    "ln_firm_age", "has_foreign_ownership", "legal_status", "has_female_owner",
    "owner_is_manager", "manager_experience_years", "manager_experience_years_sq",
    "manager_is_female", "has_training_programs", "introduced_new_products",
    "introduced_new_process", "has_rd_spending", "uses_foreign_tech_license",
    "has_website_social_media", "has_bank_account", "has_overdraft_facility",
    "has_external_audit", "experienced_climate_damage", "main_activity_type",
    "has_quality_certification", "is_exporter", "has_government_contract",
    "gdp_per_capita_ppp", "industry_share_gdp", "EPI.new",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "br_env", "interview_year",
]
print(ES_firm_level[[c for c in key_vars if c in ES_firm_level.columns]]
      .isna().sum().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# 4.  REGRESSIONS  (Firm_Level_Survey.R — regression section)
# ══════════════════════════════════════════════════════════════════════════════

# ── Shared formula building blocks ────────────────────────────────────────────
# statsmodels uses C() for categorical variables; pymer4 uses pandas Categorical
FIRM_RHS = (
    "num_employees + ln_firm_age + has_foreign_ownership + "
    "C(legal_status) + has_female_owner + owner_is_manager + "
    "manager_experience_years + manager_experience_years_sq + manager_is_female + "
    "has_training_programs + introduced_new_products + introduced_new_process + "
    "has_rd_spending + uses_foreign_tech_license + has_website_social_media + "
    "has_bank_account + has_overdraft_facility + has_external_audit + "
    "experienced_climate_damage + C(main_activity_type) + "
    "has_quality_certification + is_exporter + has_government_contract"
)

# For pymer4 (R-style): pandas Categorical columns are auto-treated as factors
FIRM_RHS_R = FIRM_RHS.replace("C(legal_status)", "legal_status") \
                      .replace("C(main_activity_type)", "main_activity_type")

COUNTRY_V1 = "gdp_per_capita_ppp + industry_share_gdp"
WGI_V      = ("control_corruption_est + gov_effectiveness_est + "
              "political_stability_est + regulatory_quality_est + "
              "rule_of_law_est + voice_accountability_est")

# Model-specific country-level RHS blocks (mirrors model naming in R)
COUNTRY_RHS = {
    "fandc_1":     COUNTRY_V1,
    "fandc_2":     f"{COUNTRY_V1} + {WGI_V}",
    "fandc_3":     f"{COUNTRY_V1} + {WGI_V} + `EPI.new`",
    "fandcandBR_1": f"gdp_per_capita_ppp + {WGI_V} + br_env",
}

OUTCOMES = [("monitors_co2_emissions", "co"), ("adopt_energy_management", "em")]


# ── GLM logit with country + year fixed effects (model_firm_1_*) ──────────────
def logit_fe(outcome, data=ES_firm_level):
    """Logistic regression with country and year fixed effects + clustered SEs.

    Mirrors:  glm(..., family=binomial) + coeftest(vcovCL, cluster=~country_name)

    Two pre-fit filters applied (both mirror what R's glm does silently):
    1. Countries with no outcome variation (complete separation).
    2. Countries whose dummy is all-zero in the complete-case design matrix
       (they appear in the data but have no observations after NaN-dropping all
       formula variables — their dummies are collinear with the intercept).
    """
    import patsy

    # Filter 1: drop countries with no variation in outcome
    variation = (data.groupby("country_name")[outcome]
                 .apply(lambda s: s.dropna().nunique()))
    data_fe = data[data["country_name"].isin(variation[variation > 1].index)].copy()

    formula = f"{outcome} ~ {FIRM_RHS} + C(country_name) + C(interview_year)"

    # Filter 2: drop countries with no complete observations in the full model
    _, X_temp = patsy.dmatrices(formula, data=data_fe,
                                return_type="dataframe", NA_action="drop")
    empty_dummies = [c for c in X_temp.columns
                     if c.startswith("C(country_name)[T.") and X_temp[c].sum() == 0]
    if empty_dummies:
        bad = {c.split("[T.")[1].rstrip("]") for c in empty_dummies}
        print(f"    Dropping {len(bad)} country/ies with no complete obs: {sorted(bad)}")
        data_fe = data_fe[~data_fe["country_name"].isin(bad)].copy()

    # Build aligned groups: use the patsy complete-case index so len(groups)
    # exactly matches the estimation sample (statsmodels requires this).
    y_cc, _ = patsy.dmatrices(formula, data=data_fe,
                               return_type="dataframe", NA_action="drop")
    groups_cc = data_fe.loc[y_cc.index, "country_name"].values

    result = smf.logit(formula, data=data_fe).fit(
        maxiter=300, disp=False,
        cov_type="cluster",
        cov_kwds={"groups": groups_cc},
    )
    return result


# ── Logit with country controls (WDI + WGI + EPI), no country FE ─────────────
def logit_ctrl(outcome, data=ES_firm_level):
    """Logit with firm-level RHS + WDI/WGI/EPI controls, no country FE.
    Clustered SEs at country level.
    """
    import patsy
    # Replace backtick-quoted `EPI.new` with patsy's Q() quoting (backticks not
    # supported in all patsy versions when called directly via dmatrices).
    country_ctrl = COUNTRY_RHS["fandc_3"].replace("`EPI.new`", "Q('EPI.new')")
    formula = (f"{outcome} ~ {FIRM_RHS} + "
               f"{country_ctrl} + C(interview_year)")

    y_cc, _ = patsy.dmatrices(formula, data=data,
                               return_type="dataframe", NA_action="drop")
    groups_cc = data.loc[y_cc.index, "country_name"].values

    result = smf.logit(formula, data=data).fit(
        maxiter=300, disp=False,
        cov_type="cluster",
        cov_kwds={"groups": groups_cc},
    )
    return result


# ── Logit with B-Ready env score instead of EPI ───────────────────────────────
def logit_ctrl_br(outcome, data=ES_firm_level):
    """Logit with firm-level RHS + WDI/WGI/B-Ready controls (no EPI, no country FE).
    Clustered SEs at country level.
    """
    import patsy
    # Drop countries with no outcome variation (prevents singular Hessian)
    variation = (data.groupby("country_name")[outcome]
                 .apply(lambda s: s.dropna().nunique()))
    data_br = data[data["country_name"].isin(variation[variation > 1].index)].copy()

    # B-Ready data only spans 2022-23; year dummies for later years are all-zero
    # → rank deficiency.  Detect and remove zero-variance year dummies.
    for country_suffix in [f"{COUNTRY_V1} + {WGI_V} + br_env",
                           f"{COUNTRY_V1} + br_env"]:
        formula = f"{outcome} ~ {FIRM_RHS} + {country_suffix} + C(interview_year)"
        _, X_temp = patsy.dmatrices(formula, data=data_br,
                                     return_type="dataframe", NA_action="drop")
        bad_years = set()
        for c in X_temp.columns:
            if "interview_year" in c and X_temp[c].sum() == 0:
                try:
                    bad_years.add(float(c.split("[T.")[1].rstrip("]")))
                except (IndexError, ValueError):
                    pass
        if bad_years:
            print(f"    Dropped year dummies: {sorted(bad_years)}")
            data_br = data_br[~data_br["interview_year"].isin(bad_years)].copy()
        try:
            y_cc, _ = patsy.dmatrices(formula, data=data_br,
                                       return_type="dataframe", NA_action="drop")
            groups_cc = data_br.loc[y_cc.index, "country_name"].values
            return smf.logit(formula, data=data_br).fit(
                maxiter=300, disp=False,
                cov_type="cluster",
                cov_kwds={"groups": groups_cc},
            )
        except np.linalg.LinAlgError:
            print(f"    Singular Hessian with '{country_suffix}', trying simpler spec…")
    raise RuntimeError(f"Could not fit B-Ready logit for {outcome}")


# ── GLMER with random country intercept (model_fandc_* / model_fandcandBR_*) ──
def glmer(outcome, country_rhs, data=ES_firm_level):
    """Mixed-effects logit with random country intercept.

    Mirrors:  glmer(..., family=binomial, control=glmerControl(optimizer='bobyqa'))
    Requires: pymer4 (which calls R's lme4 under the hood).
    """
    if not PYMER4:
        raise RuntimeError("pymer4 is required for GLMER models")
    formula = (f"{outcome} ~ {FIRM_RHS_R} + {country_rhs} + "
               f"C(interview_year) + (1 | country_name)")
    m = Lmer(formula, data=data, family="binomial")
    m.fit(REML=False,
          control="glmerControl(optimizer='bobyqa', optCtrl=list(maxfun=200000))",
          verbose=False)
    return m


# ── Fit everything ─────────────────────────────────────────────────────────────
print("\n=== GLM Logit — country + year fixed effects ===")
glm_models = {}
for outcome, suffix in OUTCOMES:
    name = f"model_firm_1_{suffix}"
    print(f"  Fitting {name}…")
    glm_models[name] = logit_fe(outcome)
    print(glm_models[name].summary().tables[1])

# Expose as module-level names matching the R convention
model_firm_1_co = glm_models["model_firm_1_co"]
model_firm_1_em = glm_models["model_firm_1_em"]

glmer_models = {}
if PYMER4:
    print("\n=== GLMER — random country intercept ===")
    for spec, country_rhs in COUNTRY_RHS.items():
        for outcome, suffix in OUTCOMES:
            name = f"model_{spec}_{suffix}"
            print(f"  Fitting {name}…")
            glmer_models[name] = glmer(outcome, country_rhs)
            print(glmer_models[name].coefs)

# Convenience references matching R names
for _name, _model in glmer_models.items():
    globals()[_name] = _model


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RANDOM FOREST  (ML.R)
# ══════════════════════════════════════════════════════════════════════════════
RF_FEATURES = [
    "num_employees", "ln_firm_age", "has_foreign_ownership",
    "has_female_owner", "owner_is_manager", "manager_experience_years",
    "manager_is_female", "has_training_programs", "introduced_new_products",
    "introduced_new_process", "has_rd_spending", "uses_foreign_tech_license",
    "has_website_social_media", "has_bank_account", "has_overdraft_facility",
    "has_external_audit", "experienced_climate_damage", "has_quality_certification",
    "is_exporter", "has_government_contract",
    "gdp_per_capita_ppp", "industry_share_gdp",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "EPI.new", "main_activity_type",
]

# Human-readable labels and thematic categories (mirrors ML.R case_when blocks)
_VAR_META = {
    # variable: (label, category)
    "has_training_programs":     ("Training programs",                "Firm Capabilities"),
    "has_quality_certification": ("Quality certification",            "Firm Capabilities"),
    "has_rd_spending":           ("R&D spending",                     "Firm Capabilities"),
    "uses_foreign_tech_license": ("Foreign technology",               "Firm Capabilities"),
    "introduced_new_process":    ("Process innovation",               "Firm Capabilities"),
    "introduced_new_products":   ("Product innovation",               "Firm Capabilities"),
    "has_external_audit":        ("External audit",                   "Firm Capabilities"),
    "has_bank_account":          ("Bank account",                     "Financial Access"),
    "has_overdraft_facility":    ("Overdraft facility",               "Financial Access"),
    "num_employees":             ("Firm size (log employees)",        "Firm Characteristics"),
    "ln_firm_age":               ("Firm age (log)",                   "Firm Characteristics"),
    "has_foreign_ownership":     ("Foreign ownership",                "Firm Characteristics"),
    "has_government_contract":   ("Government contract",              "Market Linkages"),
    "is_exporter":               ("Exporter status",                  "Market Linkages"),
    "experienced_climate_damage":("Climate shock",                    "Climate Exposure"),
    "EPI.new":                   ("EPI",                              "ENV Governance"),
    "br_env":                    ("B-Ready Env. Score",               "ENV Governance"),
    "gdp_per_capita_ppp":        ("GDP per capita (PPP)",             "Country Context"),
    "industry_share_gdp":        ("Industry share of GDP",            "Country Context"),
    "control_corruption_est":    ("Control of corruption",            "Governance"),
    "gov_effectiveness_est":     ("Government effectiveness",         "Governance"),
    "political_stability_est":   ("Political stability",              "Governance"),
    "regulatory_quality_est":    ("Regulatory quality",               "Governance"),
    "rule_of_law_est":           ("Rule of law",                      "Governance"),
    "voice_accountability_est":  ("Voice & accountability",           "Governance"),
    "owner_is_manager":          ("Owner is manager",                 "Firm Characteristics"),
    "manager_is_female":         ("Female manager",                   "Firm Characteristics"),
    "has_female_owner":          ("Female owner",                     "Firm Characteristics"),
    "manager_experience_years":  ("Manager experience",               "Firm Characteristics"),
    "has_website_social_media":  ("Website/social media",             "Firm Capabilities"),
    "main_activity_type":        ("Industry sector",                  "Firm Characteristics"),
}

# Variables to include in the marginal-effects forest plot (20 predictors of interest)
FIRM_VARS_PLOT = [
    # Firm Capabilities
    "has_training_programs", "has_quality_certification", "has_rd_spending",
    "uses_foreign_tech_license", "introduced_new_process", "introduced_new_products",
    "has_external_audit", "has_website_social_media",
    # Financial Access
    "has_bank_account", "has_overdraft_facility",
    # Firm Characteristics
    "num_employees", "ln_firm_age", "has_foreign_ownership",
    "owner_is_manager", "manager_is_female", "has_female_owner",
    "manager_experience_years",
    # Market Linkages
    "has_government_contract", "is_exporter",
    # Climate Exposure
    "experienced_climate_damage",
]

CATEGORY_ORDER = [
    "Firm Capabilities",
    "Financial Access",
    "Firm Characteristics",
    "Market Linkages",
    "Climate Exposure",
]

COUNTRY_VARS_PLOT = [
    "gdp_per_capita_ppp", "industry_share_gdp",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "EPI.new", "br_env",
]

COUNTRY_CATEGORY_ORDER = ["Country Context", "Governance", "ENV Governance"]


def plot_marginal_effects(result, outcome, suffix):
    """Extract marginal effects from a fitted logit result and produce a forest plot.

    Uses get_margeff(at='mean', method='dydx', dummy=True): marginal effects
    evaluated at the mean of all covariates.  'at=overall' (proper AME) would
    require allocating a ~7 GiB Jacobian (136 params × 50 k obs × 136 params)
    and crashes with OOM; at='mean' produces nearly identical estimates for
    logit and is the standard approach for large FE models.
    Binary predictors are treated as discrete 0→1 changes (dummy=True).
    CIs reflect the clustered SEs baked into the fitted result.
    """
    # 1. Compute marginal effects at the mean
    mfx = result.get_margeff(at="mean", method="dydx", dummy=True).summary_frame()

    # Robust column access by position (statsmodels names vary slightly across versions)
    cols      = mfx.columns.tolist()
    ame_col   = cols[0]   # dy/dx
    pval_col  = cols[3]   # Pr(>|z|)
    ci_lo_col = cols[4]   # Conf. Int. Low
    ci_hi_col = cols[5]   # Conf. Int. Hi.

    # 2. Filter to the 20 predictors of interest
    mfx = mfx[mfx.index.isin(FIRM_VARS_PLOT)].copy()
    if mfx.empty:
        print(f"  Warning: no FIRM_VARS_PLOT variables found in AMEs for {suffix}. Skipping.")
        return

    # 3. Map each variable to its label and category
    mfx["label"]    = [_VAR_META.get(v, (v, "Other"))[0] for v in mfx.index]
    mfx["category"] = [_VAR_META.get(v, (v, "Other"))[1] for v in mfx.index]

    # 4. Sort: by CATEGORY_ORDER (primary) then descending |AME| within each category
    cat_rank       = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    mfx["cat_rank"] = [cat_rank.get(c, len(CATEGORY_ORDER)) for c in mfx["category"]]
    mfx["abs_ame"]  = mfx[ame_col].abs()
    mfx = mfx.sort_values(["cat_rank", "abs_ame"], ascending=[True, False]).reset_index()

    # 5. Build y-axis positions with a 1.5-unit blank-row gap between categories
    y_positions = []
    y = 0.0
    prev_cat = None
    for _, row in mfx.iterrows():
        cat = row["category"]
        if prev_cat is not None and cat != prev_cat:
            y += 1.5
        y_positions.append(y)
        prev_cat = cat
        y += 1.0
    mfx["y_pos"] = y_positions

    # Colour palette (one colour per category, reuses Set2 palette like RF plot)
    palette    = plt.cm.Set2
    cat_colors = {c: palette(i / max(len(CATEGORY_ORDER) - 1, 1))
                  for i, c in enumerate(CATEGORY_ORDER)}

    # 6. Draw the forest plot
    fig, ax = plt.subplots(figsize=(9, 10))

    plot_rows = []
    for _, row in mfx.iterrows():
        color = cat_colors.get(row["category"], "grey")
        yp    = row["y_pos"]
        ame   = row[ame_col]
        ci_lo = row[ci_lo_col]
        ci_hi = row[ci_hi_col]

        ax.plot(ame, yp, "o", color=color, markersize=6, zorder=3)
        ax.hlines(yp, ci_lo, ci_hi, color=color, linewidth=1.5, zorder=2)
        plot_rows.append((row, color))

    # 8. Zero-effect reference line
    ax.axvline(0, linestyle="--", color="grey", linewidth=0.8)

    # y-axis: variable labels
    ax.set_yticks(mfx["y_pos"])
    ax.set_yticklabels(mfx["label"], fontsize=9)
    ax.invert_yaxis()

    # Significance stars placed to the right of each CI bar
    x_lo, x_hi   = ax.get_xlim()
    star_offset   = (x_hi - x_lo) * 0.02
    for row, color in plot_rows:
        pval = row[pval_col]
        stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        if stars:
            ax.text(row[ci_hi_col] + star_offset, row["y_pos"], stars,
                    va="center", ha="left", fontsize=8, color=color)

    # 7. Category labels — bold text to the left of each group using blended transform
    # (x in axes coords [0..1], y in data coords)
    yaxis_trans = ax.get_yaxis_transform()
    for cat in CATEGORY_ORDER:
        cat_rows = mfx[mfx["category"] == cat]
        if cat_rows.empty:
            continue
        y_mid = cat_rows["y_pos"].mean()
        ax.text(-0.28, y_mid, cat,
                transform=yaxis_trans,
                va="center", ha="right",
                fontweight="bold", fontsize=8,
                color=cat_colors.get(cat, "grey"),
                clip_on=False)

    # 9. Final decorations
    ax.set_xlabel("Average Marginal Effect (pp)")
    outcome_title = {
        "monitors_co2_emissions":  "P(Monitors CO\u2082 Emissions)",
        "adopt_energy_management": "P(Adopts Energy Management)",
    }.get(outcome, outcome)
    ax.set_title(f"Marginal Effects on {outcome_title}", fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()

    # 10. Save
    for ext in ("png", "pdf"):
        path = OUT / f"marginal_effects_{suffix}.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.close(fig)
    print(f"  Saved marginal_effects_{suffix}.png/.pdf")


def plot_marginal_effects_ctrl(result, outcome, suffix, result_br=None):
    """2-panel forest plot: firm-level AMEs (left) + country-level AMEs (right).

    result     — EPI controls model (WDI/WGI/EPI, no country FE)
    result_br  — B-Ready controls model (WDI/WGI/br_env); its br_env AME is
                 appended to the right panel under "ENV Governance"
    """
    mfx = result.get_margeff(at="mean", method="dydx", dummy=True).summary_frame()

    # Patsy names Q('EPI.new') columns as "Q('EPI.new')" — normalise back to
    # the plain variable name so COUNTRY_VARS_PLOT filtering works.
    mfx.index = mfx.index.str.replace(r"Q\('(.+?)'\)", r"\1", regex=True)

    # Append br_env AME from the B-Ready model (different spec)
    if result_br is not None:
        mfx_br = result_br.get_margeff(at="mean", method="dydx", dummy=True).summary_frame()
        if "br_env" in mfx_br.index:
            mfx = pd.concat([mfx, mfx_br.loc[["br_env"]]])

    cols      = mfx.columns.tolist()
    ame_col   = cols[0]
    pval_col  = cols[3]
    ci_lo_col = cols[4]
    ci_hi_col = cols[5]

    def _build_panel(ax, var_list, cat_order):
        """Draw a forest plot on ax for the given variable list and category order."""
        panel = mfx[mfx.index.isin(var_list)].copy()
        if panel.empty:
            ax.set_visible(False)
            return

        panel["label"]    = [_VAR_META.get(v, (v, "Other"))[0] for v in panel.index]
        panel["category"] = [_VAR_META.get(v, (v, "Other"))[1] for v in panel.index]

        cat_rank       = {c: i for i, c in enumerate(cat_order)}
        panel["cat_rank"] = [cat_rank.get(c, len(cat_order)) for c in panel["category"]]
        panel["abs_ame"]  = panel[ame_col].abs()
        panel = panel.sort_values(["cat_rank", "abs_ame"], ascending=[True, False]).reset_index()

        # y positions with 1.5-unit gap between categories
        y_positions = []
        y = 0.0
        prev_cat = None
        for _, row in panel.iterrows():
            cat = row["category"]
            if prev_cat is not None and cat != prev_cat:
                y += 1.5
            y_positions.append(y)
            prev_cat = cat
            y += 1.0
        panel["y_pos"] = y_positions

        palette    = plt.cm.Set2
        all_cats   = CATEGORY_ORDER + COUNTRY_CATEGORY_ORDER
        cat_colors = {c: palette(i / max(len(all_cats) - 1, 1))
                      for i, c in enumerate(all_cats)}

        plot_rows = []
        for _, row in panel.iterrows():
            color = cat_colors.get(row["category"], "grey")
            yp    = row["y_pos"]
            ame   = row[ame_col]
            ci_lo = row[ci_lo_col]
            ci_hi = row[ci_hi_col]
            ax.plot(ame, yp, "o", color=color, markersize=6, zorder=3)
            ax.hlines(yp, ci_lo, ci_hi, color=color, linewidth=1.5, zorder=2)
            plot_rows.append((row, color))

        ax.axvline(0, linestyle="--", color="grey", linewidth=0.8)
        ax.set_yticks(panel["y_pos"])
        ax.set_yticklabels(panel["label"], fontsize=9)
        ax.invert_yaxis()

        x_lo, x_hi  = ax.get_xlim()
        star_offset  = (x_hi - x_lo) * 0.02
        for row, color in plot_rows:
            pval  = row[pval_col]
            stars = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
            if stars:
                ax.text(row[ci_hi_col] + star_offset, row["y_pos"], stars,
                        va="center", ha="left", fontsize=8, color=color)

        yaxis_trans = ax.get_yaxis_transform()
        for cat in cat_order:
            cat_rows = panel[panel["category"] == cat]
            if cat_rows.empty:
                continue
            y_mid = cat_rows["y_pos"].mean()
            ax.text(-0.28, y_mid, cat,
                    transform=yaxis_trans,
                    va="center", ha="right",
                    fontweight="bold", fontsize=8,
                    color=cat_colors.get(cat, "grey"),
                    clip_on=False)

        ax.set_xlabel("Average Marginal Effect (pp)")
        ax.grid(axis="x", linestyle=":", alpha=0.4)

    outcome_title = {
        "monitors_co2_emissions":  "CO\u2082 Monitoring",
        "adopt_energy_management": "Energy Management",
    }.get(outcome, outcome)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 10))
    _build_panel(ax_left,  FIRM_VARS_PLOT,    CATEGORY_ORDER)
    _build_panel(ax_right, COUNTRY_VARS_PLOT, COUNTRY_CATEGORY_ORDER)

    ax_left.set_title("Firm-Level Predictors", fontweight="bold", pad=8)
    ax_right.set_title("Country-Level Controls", fontweight="bold", pad=8)

    fig.suptitle(
        f"Marginal Effects \u2014 {outcome_title} (Controls model: WDI + WGI + EPI)",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"marginal_effects_ctrl_{suffix}.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.close(fig)
    print(f"  Saved marginal_effects_ctrl_{suffix}.png/.pdf")


def plot_epi_vs_country_fe(result, outcome, suffix):
    """Scatter plot of EPI score vs country fixed effect from the logit model.

    Each point is a country.  The y-axis is the country's log-odds fixed
    effect (relative to the omitted reference country), which captures
    baseline green-adoption propensity net of all firm-level covariates.
    An OLS trend line tests whether environmentally ambitious countries
    (high EPI) have systematically higher baseline adoption.

    Note: uses country fixed effects from the firm-level logit
    (model_firm_1_*), not GLMER random intercepts, because pymer4 may not
    be available.  Fixed effects and random intercepts serve the same
    conceptual role here.
    """
    # -- Extract country FE coefficients and p-values ----------------------
    params  = result.params
    pvalues = result.pvalues
    mask    = params.index.str.startswith("C(country_name)[T.")

    def _parse_name(idx):
        return idx.split("[T.")[1].rstrip("]")

    fe_df = pd.DataFrame({
        "country_name": [_parse_name(i) for i in params.index[mask]],
        "fe_coef":      params[mask].values,
        "pval":         pvalues[mask].values,
    })

    # -- Merge with EPI (use ES_firm_level which already has EPI joined) ---
    country_epi = (ES_firm_level[["country_name", "EPI.new"]]
                   .drop_duplicates("country_name")
                   .dropna(subset=["EPI.new"]))
    fe_df = fe_df.merge(country_epi, on="country_name", how="inner")

    if fe_df.empty:
        print(f"  No EPI matches for {suffix}. Skipping.")
        return

    # -- OLS trend line + Pearson r ----------------------------------------
    x     = fe_df["EPI.new"].values
    y     = fe_df["fe_coef"].values
    slope, intercept = np.polyfit(x, y, 1)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept
    corr  = np.corrcoef(x, y)[0, 1]

    # -- Select countries to label -----------------------------------------
    # Priority: biggest residuals from the OLS fit (most informative
    # deviations), plus EPI extremes so axes are anchored with real names.
    fe_df["residual"] = np.abs(y - (slope * x + intercept))
    to_label = pd.concat([
        fe_df.nlargest(5, "residual"),   # biggest deviators
        fe_df.nlargest(4, "EPI.new"),    # highest EPI
        fe_df.nsmallest(4, "EPI.new"),   # lowest EPI
    ]).drop_duplicates("country_name")

    # -- Plot --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 7))

    sig = fe_df["pval"] < 0.05
    ax.scatter(fe_df.loc[~sig, "EPI.new"], fe_df.loc[~sig, "fe_coef"],
               s=45, alpha=0.45, color="steelblue", zorder=2,
               label="p ≥ 0.05")
    ax.scatter(fe_df.loc[sig, "EPI.new"], fe_df.loc[sig, "fe_coef"],
               s=60, alpha=0.80, color="navy", zorder=3,
               label="p < 0.05")

    ax.plot(x_fit, y_fit, color="firebrick", linewidth=1.8, zorder=4,
            label=f"OLS trend  (r = {corr:+.2f})")

    ax.axhline(0, linestyle="--", color="grey", linewidth=0.8, zorder=1,
               label="Reference country (FE = 0)")

    for _, row in to_label.iterrows():
        ax.annotate(
            row["country_name"],
            xy=(row["EPI.new"], row["fe_coef"]),
            xytext=(5, 3), textcoords="offset points",
            fontsize=7, color="#333333",
        )

    # -- Decorations -------------------------------------------------------
    outcome_title = {
        "monitors_co2_emissions":  "CO\u2082 Monitoring",
        "adopt_energy_management": "Energy Management Adoption",
    }.get(outcome, outcome)

    ax.set_xlabel("EPI Score (2024)", fontsize=11)
    ax.set_ylabel("Country Fixed Effect (log-odds, vs. reference country)", fontsize=11)
    ax.set_title(
        f"EPI Score vs Country Baseline — {outcome_title}\n"
        f"Country fixed effects from firm-level logit  "
        f"(n\u202f=\u202f{len(fe_df)} countries with EPI data)",
        fontweight="bold", pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(linestyle=":", alpha=0.4)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"epi_vs_country_fe_{suffix}.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.close(fig)
    print(f"  Saved epi_vs_country_fe_{suffix}.png/.pdf")


def plot_bready_vs_adoption(data=ES_firm_level):
    """Scatter plot: B-Ready Environmental Score vs country-level green adoption rate.

    Each point is a country.  Adoption rate = share of surveyed firms
    with a positive outcome.  Bubble size is proportional to the number
    of firm observations so that countries with thin coverage are
    visually down-weighted.  Both outcomes are shown side-by-side.
    """
    # Country-level summary: adoption rate + br_env (same for all firms in country)
    outcome_titles = {
        "monitors_co2_emissions":  "CO\u2082 Monitoring",
        "adopt_energy_management": "Energy Management Adoption",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=False)

    for ax, (outcome, suffix) in zip(axes, OUTCOMES):
        grp = (
            data.groupby("country_name", observed=True)
            .agg(
                adoption_rate=(outcome,  "mean"),
                n_firms       =(outcome,  "count"),
                br_env        =("br_env", "first"),
            )
            .dropna(subset=["adoption_rate", "br_env"])
            .reset_index()
        )

        if grp.empty:
            ax.set_visible(False)
            continue

        x = grp["br_env"].values
        y = grp["adoption_rate"].values

        # Bubble area scaled to [40, 300] across the range of firm counts
        n     = grp["n_firms"].values
        sizes = 40 + (n - n.min()) / max(n.max() - n.min(), 1) * 260

        # OLS trend + Pearson r
        slope, intercept = np.polyfit(x, y, 1)
        x_fit = np.linspace(x.min(), x.max(), 200)
        y_fit = slope * x_fit + intercept
        corr  = np.corrcoef(x, y)[0, 1]

        ax.scatter(x, y, s=sizes, alpha=0.60, color="steelblue",
                   edgecolors="white", linewidths=0.5, zorder=2)
        ax.plot(x_fit, y_fit, color="firebrick", linewidth=1.8, zorder=3,
                label=f"OLS trend  (r\u202f=\u202f{corr:+.2f})")

        # Label: top residuals + x-axis extremes
        grp["residual"] = np.abs(y - (slope * x + intercept))
        to_label = pd.concat([
            grp.nlargest(5, "residual"),
            grp.nlargest(3, "br_env"),
            grp.nsmallest(3, "br_env"),
        ]).drop_duplicates("country_name").head(14)

        for _, row in to_label.iterrows():
            ax.annotate(
                row["country_name"],
                xy=(row["br_env"], row["adoption_rate"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7, color="#333333",
            )

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.set_xlabel("B-Ready Environmental Score", fontsize=11)
        ax.set_ylabel("Firm-level Adoption Rate", fontsize=11)
        ax.set_title(
            f"{outcome_titles.get(outcome, outcome)}\n"
            f"n\u202f=\u202f{len(grp)} countries",
            fontweight="bold", pad=8,
        )
        ax.legend(fontsize=9, framealpha=0.85)
        ax.grid(linestyle=":", alpha=0.4)
        ax.text(
            0.99, 0.02, "Bubble size \u221d surveyed firms",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="grey",
        )

    fig.suptitle(
        "B-Ready Environmental Score vs Green Adoption Rate",
        fontweight="bold", fontsize=13,
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"bready_vs_adoption.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.close(fig)
    print("  Saved bready_vs_adoption.png/.pdf")


def plot_adoption_heatmap(df, out_dir):
    """Heatmap of green adoption rates by sector × income group (two panels)."""
    import seaborn as sns

    outcomes = [
        ("monitors_co2_emissions",  "CO\u2082 Monitoring"),
        ("adopt_energy_management", "Energy Management"),
    ]

    # Filter: drop rows missing GDP, sector, or either outcome
    needed = ["gdp_per_capita_ppp", "main_activity_type"] + [o for o, _ in outcomes]
    df2 = df.dropna(subset=needed).copy()

    # Income groups from country-level GDP quartile (unique country values)
    country_gdp = (
        df2[["country_name", "gdp_per_capita_ppp"]]
        .drop_duplicates("country_name")
        .set_index("country_name")["gdp_per_capita_ppp"]
    )
    q_labels = ["Low", "Lower-Mid", "Upper-Mid", "High"]
    income_map = pd.qcut(country_gdp, q=4, labels=q_labels).to_dict()
    df2["income_group"] = df2["country_name"].map(income_map)

    # Column and row orders
    col_order = ["Low", "Lower-Mid", "Upper-Mid", "High"]
    row_order  = ["Manufacturing", "Services", "Retail", "Wholesale",
                  "Hotel", "Restaurant", "Construction"]

    # Build pivot matrices
    matrices = {}
    for outcome, _ in outcomes:
        pivot = (
            df2.groupby(["main_activity_type", "income_group"], observed=True)[outcome]
            .mean()
            .mul(100)
            .unstack("income_group")
            .reindex(index=row_order, columns=col_order)
        )
        matrices[outcome] = pivot

    # Shared color scale
    vmax = max(m.max().max() for m in matrices.values())

    HARRISBLUE = "#1A3E5E"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, (outcome, title), cbar in zip(
        [ax1, ax2], outcomes, [False, True]
    ):
        cbar_kws = {"label": "Adoption rate (%)"} if cbar else {}
        sns.heatmap(
            matrices[outcome],
            ax=ax,
            annot=False,
            cmap="YlOrRd",
            vmin=0, vmax=vmax,
            linewidths=0.5, linecolor="white",
            cbar=cbar, cbar_kws=cbar_kws,
        )
        ax.set_title(title, fontsize=11, color=HARRISBLUE, fontweight="bold")
        ax.set_xlabel("Income Group", fontsize=9)
        ax.set_ylabel("Sector" if ax is ax1 else "", fontsize=9)

    fig.suptitle(
        "Green Adoption Rate by Sector and Income Group (%)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = out_dir / f"adoption_heatmap.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.close(fig)
    print("  Saved adoption_heatmap.png/.pdf")


def run_random_forest(outcome="adopt_energy_management",
                      features=RF_FEATURES,
                      data=ES_firm_level):
    """Train RF classifier and plot variable importance.

    Mirrors randomForest(..., ntree=500, mtry=sqrt(p), nodesize=10)
    """
    rf_data = (data[[outcome] + features]
               .dropna()
               .copy())
    # One-hot encode industry sector (string/categorical column)
    rf_data = pd.get_dummies(rf_data, columns=["main_activity_type"], drop_first=True)

    X = rf_data.drop(columns=[outcome])
    y = rf_data[outcome].astype(int)

    np.random.seed(123)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=10,
        oob_score=True,
        random_state=123,
        n_jobs=-1,
    )
    rf.fit(X, y)

    print(f"\nOOB Score: {rf.oob_score_:.4f}  |  OOB Error: {1 - rf.oob_score_:.4f}")

    # Build importance dataframe — strip dummy suffixes to match _VAR_META keys
    imp = (pd.DataFrame({"variable": X.columns,
                         "MeanDecreaseGini": rf.feature_importances_})
           .sort_values("MeanDecreaseGini", ascending=False)
           .reset_index(drop=True))

    imp["base_var"] = imp["variable"].str.replace(r"_[A-Za-z ]+$", "", regex=True)
    imp["variable_label"] = (imp["variable"].map({k: v[0] for k, v in _VAR_META.items()})
                             .fillna(imp["base_var"].map({k: v[0] for k, v in _VAR_META.items()}))
                             .fillna(imp["variable"]))
    imp["category"] = (imp["variable"].map({k: v[1] for k, v in _VAR_META.items()})
                       .fillna(imp["base_var"].map({k: v[1] for k, v in _VAR_META.items()}))
                       .fillna("Other"))

    print("\nTop 20 Variable Importances:")
    print(imp.head(20)[["variable_label", "MeanDecreaseGini", "category"]].to_string(index=False))

    # ── Plot (mirrors ggplot barh in ML.R) ────────────────────────────────────
    top20   = imp.head(20)
    cats    = list(top20["category"].unique())
    palette = plt.cm.Set2
    colors  = {c: palette(i / max(len(cats) - 1, 1)) for i, c in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top20["variable_label"], top20["MeanDecreaseGini"],
            color=[colors[c] for c in top20["category"]])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Decrease in Gini Impurity")
    ax.set_title("Variable Importance: Energy Management Adoption\n"
                 "Random Forest (500 trees)", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in cats]
    ax.legend(handles, cats, title="Variable Type", bbox_to_anchor=(1, 1))
    ax.text(0.5, -0.06, "Note: Higher values indicate greater importance in prediction.",
            transform=ax.transAxes, ha="center", fontsize=9)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = OUT / f"rf_variable_importance.{ext}"
        try:
            fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        except PermissionError:
            print(f"  Warning: could not save {path.name} (file locked — close it in your viewer)")
    plt.show()

    imp.to_csv(OUT / "rf_importance.csv", index=False)
    return rf, imp


rf_model, importance_df = run_random_forest()

print("\n=== Marginal Effects Forest Plots ===")
for outcome, suffix in OUTCOMES:
    print(f"  Computing AMEs for {outcome}…")
    plot_marginal_effects(glm_models[f"model_firm_1_{suffix}"], outcome, suffix)

print("\n=== Controls-model logit (WDI + WGI + EPI) ===")
ctrl_models = {}
for outcome, suffix in OUTCOMES:
    print(f"  Fitting EPI controls model for {outcome}…")
    ctrl_models[suffix] = logit_ctrl(outcome)

print("\n=== Controls-model logit (WDI + WGI + B-Ready) ===")
ctrl_models_br = {}
for outcome, suffix in OUTCOMES:
    print(f"  Fitting B-Ready controls model for {outcome}…")
    ctrl_models_br[suffix] = logit_ctrl_br(outcome)

print("\n=== Marginal Effects Forest Plots — Controls Model ===")
for outcome, suffix in OUTCOMES:
    print(f"  Computing AMEs for {outcome}…")
    plot_marginal_effects_ctrl(ctrl_models[suffix], outcome, suffix,
                               result_br=ctrl_models_br[suffix])

print("\n=== EPI vs Country Fixed Effect Plots ===")
for outcome, suffix in OUTCOMES:
    print(f"  Plotting {outcome}…")
    plot_epi_vs_country_fe(glm_models[f"model_firm_1_{suffix}"], outcome, suffix)

print("\n=== B-Ready Environmental Score vs Adoption Rate ===")
plot_bready_vs_adoption()

print("\n=== Adoption Rate Heatmap (Sector × Income Group) ===")
plot_adoption_heatmap(ES_firm_level, OUT)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  ALTAIR VISUALIZATIONS  (interactive HTML + PNG if vl-convert-python installed)
# ══════════════════════════════════════════════════════════════════════════════

try:
    import altair as alt
    ALTAIR = True
except ImportError:
    ALTAIR = False
    print("altair not found — Altair plots will be skipped.\n"
          "Install: pip install altair")

if ALTAIR:

    _MAROON = "#800000"

    # One colour per thematic category (mirrors Set2 approach in matplotlib plots)
    _CAT_COLORS = {
        "Firm Capabilities":    "#66c2a5",
        "Financial Access":     "#fc8d62",
        "Firm Characteristics": "#8da0cb",
        "Market Linkages":      "#e78ac3",
        "Climate Exposure":     "#a6d854",
    }

    def _save_altair(chart, name):
        """Save as HTML (always) and PNG (requires vl-convert-python)."""
        html_path = OUT / f"{name}.html"
        chart.save(str(html_path))
        print(f"  Saved {name}.html")
        try:
            chart.save(str(OUT / f"{name}.png"), scale_factor=2)
            print(f"  Saved {name}.png")
        except Exception:
            print(f"  Note: PNG skipped for {name} — install vl-convert-python")

    # ── 6a. Descriptive: green adoption rates by GDP income quartile ──────────

    def altair_adoption_by_income(data=ES_firm_level):
        """Grouped bar chart: adoption rates by GDP per capita quartile.

        Uses log GDP per capita (already in data) to assign country-level
        income quartiles, then shows firm-level adoption rates per quartile
        for both outcomes side-by-side.
        """
        df = (data[["country_name", "monitors_co2_emissions",
                     "adopt_energy_management", "gdp_per_capita_ppp"]]
              .dropna(subset=["gdp_per_capita_ppp"])
              .copy())

        # Assign quartile at country level (one label per country)
        country_gdp = (df.groupby("country_name", observed=True)["gdp_per_capita_ppp"]
                       .first().reset_index())
        country_gdp["income_group"] = pd.qcut(
            country_gdp["gdp_per_capita_ppp"], q=4,
            labels=["Q1 — Lowest", "Q2", "Q3", "Q4 — Highest"],
        )
        df = df.merge(country_gdp[["country_name", "income_group"]], on="country_name")

        long = (
            df.groupby("income_group", observed=True)
            .agg(co2=("monitors_co2_emissions", "mean"),
                 em=("adopt_energy_management", "mean"),
                 n=("monitors_co2_emissions", "count"))
            .reset_index()
            .melt(id_vars=["income_group", "n"],
                  value_vars=["co2", "em"],
                  var_name="outcome", value_name="rate")
        )
        long["Outcome"] = long["outcome"].map(
            {"co2": "CO₂ Monitoring", "em": "Energy Management"}
        )
        long["rate_pct"] = (long["rate"] * 100).round(1)

        chart = (
            alt.Chart(long)
            .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("income_group:N",
                        sort=["Q1 — Lowest", "Q2", "Q3", "Q4 — Highest"],
                        title="GDP per Capita Quartile (log PPP)"),
                y=alt.Y("rate:Q", title="Adoption Rate",
                        axis=alt.Axis(format=".0%")),
                color=alt.Color("Outcome:N",
                                scale=alt.Scale(range=[_MAROON, "#aaaaaa"]),
                                legend=alt.Legend(title="")),
                xOffset="Outcome:N",
                tooltip=[
                    alt.Tooltip("income_group:N", title="Income Group"),
                    alt.Tooltip("Outcome:N"),
                    alt.Tooltip("rate_pct:Q", title="Adoption Rate (%)", format=".1f"),
                    alt.Tooltip("n:Q", title="Firm Observations"),
                ],
            )
            .properties(
                width=480, height=320,
                title=alt.TitleParams(
                    "Green Practice Adoption by Income Level",
                    subtitle="Share of firms with CO₂ monitoring or energy management, by GDP per capita quartile",
                    fontSize=14, fontWeight="bold",
                ),
            )
        )
        _save_altair(chart, "altair_adoption_by_income")
        return chart

    # ── 6b. Descriptive: adoption rates by firm characteristics ───────────────

    def altair_adoption_by_firm_chars(data=ES_firm_level):
        """Dot plot: adoption rates by training, certification, R&D, and firm size.

        Each panel shows one characteristic (facet rows); dots compare Yes/No
        groups or firm size quartiles for both outcomes.
        """
        chars = {
            "has_training_programs":    "Training Programs",
            "has_quality_certification": "Quality Certification",
            "has_rd_spending":           "R&D Spending",
        }
        rows = []

        for var, label in chars.items():
            col = data[var].dropna()
            for outcome, out_label in [
                ("monitors_co2_emissions", "CO₂ Monitoring"),
                ("adopt_energy_management", "Energy Management"),
            ]:
                grp = (
                    data[[var, outcome]].dropna()
                    .groupby(var, observed=True)[outcome]
                    .agg(["mean", "count"])
                    .reset_index()
                )
                grp.columns = [var, "rate", "n"]
                grp["group"] = grp[var].apply(
                    lambda v: "Yes" if float(v) == 1 else "No"
                )
                grp["characteristic"] = label
                grp["outcome"]        = out_label
                rows.append(grp[["characteristic", "group", "outcome", "rate", "n"]])

        # Firm size quartiles from log employment
        df_size = data[["monitors_co2_emissions", "adopt_energy_management",
                         "num_employees"]].dropna(subset=["num_employees"]).copy()
        df_size["group"] = pd.qcut(
            df_size["num_employees"], q=4,
            labels=["Q1 Small", "Q2", "Q3", "Q4 Large"],
        ).astype(str)
        for outcome, out_label in [
            ("monitors_co2_emissions", "CO₂ Monitoring"),
            ("adopt_energy_management", "Energy Management"),
        ]:
            grp = (
                df_size.groupby("group", observed=True)[outcome]
                .agg(["mean", "count"])
                .reset_index()
            )
            grp.columns = ["group", "rate", "n"]
            grp["characteristic"] = "Firm Size (log employees)"
            grp["outcome"]        = out_label
            rows.append(grp[["characteristic", "group", "outcome", "rate", "n"]])

        long = pd.concat(rows, ignore_index=True)
        long["rate_pct"] = (long["rate"] * 100).round(1)

        dots = (
            alt.Chart(long)
            .mark_point(filled=True, size=100)
            .encode(
                x=alt.X("rate:Q", title="Adoption Rate",
                        axis=alt.Axis(format=".0%")),
                y=alt.Y("group:N", title=""),
                color=alt.Color("outcome:N",
                                scale=alt.Scale(range=[_MAROON, "#aaaaaa"]),
                                legend=alt.Legend(title="")),
                tooltip=[
                    alt.Tooltip("characteristic:N", title="Characteristic"),
                    alt.Tooltip("group:N",           title="Group"),
                    alt.Tooltip("outcome:N",         title="Outcome"),
                    alt.Tooltip("rate_pct:Q",        title="Rate (%)", format=".1f"),
                    alt.Tooltip("n:Q",               title="Observations"),
                ],
            )
        )

        chart = (
            dots
            .facet(
                row=alt.Row("characteristic:N", title="",
                            header=alt.Header(labelAngle=0, labelAlign="left")),
            )
            .properties(
                title=alt.TitleParams(
                    "Adoption Rates by Firm Characteristics",
                    subtitle="Share of firms monitoring CO₂ or adopting energy management",
                    fontSize=14, fontWeight="bold",
                ),
            )
            .resolve_scale(y="independent")
        )
        _save_altair(chart, "altair_adoption_by_firm_chars")
        return chart

    # ── 6c. Marginal effects forest plot ─────────────────────────────────────

    def altair_marginal_effects(glm_models=glm_models):
        """Interactive forest plot of average marginal effects, both outcomes.

        Mirrors plot_marginal_effects() but uses Altair. Each panel is one
        outcome; variables are sorted by category then |AME|. Hover for details.
        """
        records = []
        for outcome, suffix in OUTCOMES:
            result = glm_models[f"model_firm_1_{suffix}"]
            mfx    = result.get_margeff(at="mean", method="dydx", dummy=True).summary_frame()
            cols   = mfx.columns.tolist()

            mfx = mfx[mfx.index.isin(FIRM_VARS_PLOT)].copy()
            mfx["label"]    = [_VAR_META.get(v, (v, "Other"))[0] for v in mfx.index]
            mfx["category"] = [_VAR_META.get(v, (v, "Other"))[1] for v in mfx.index]
            mfx["ame"]      = mfx[cols[0]]
            mfx["pval"]     = mfx[cols[3]]
            mfx["ci_lo"]    = mfx[cols[4]]
            mfx["ci_hi"]    = mfx[cols[5]]
            mfx["sig"]      = mfx["pval"].apply(
                lambda p: "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else "ns"))
            )
            mfx["ame_pp"]   = (mfx["ame"] * 100).round(2)
            mfx["outcome"]  = (
                "CO₂ Monitoring" if outcome == "monitors_co2_emissions"
                else "Energy Management"
            )

            cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
            mfx["cat_rank"] = mfx["category"].map(lambda c: cat_rank.get(c, 99))
            mfx["abs_ame"]  = mfx["ame"].abs()
            mfx = mfx.sort_values(["cat_rank", "abs_ame"], ascending=[True, False])
            records.append(mfx.reset_index(drop=False))

        df = pd.concat(records, ignore_index=True)

        # Consistent y-sort order taken from one outcome panel
        y_order      = df[df["outcome"] == "CO₂ Monitoring"]["label"].tolist()
        cat_domain   = list(_CAT_COLORS.keys())
        cat_range    = [_CAT_COLORS[c] for c in cat_domain]

        shared_color = alt.Color(
            "category:N",
            scale=alt.Scale(domain=cat_domain, range=cat_range),
            legend=alt.Legend(title="Variable Group"),
        )

        base = alt.Chart(df)

        ci_bars = base.mark_rule(strokeWidth=2.5).encode(
            x=alt.X("ci_lo:Q", title="Average Marginal Effect (percentage points)"),
            x2="ci_hi:Q",
            y=alt.Y("label:N", sort=y_order, title=""),
            color=shared_color,
        )

        dots = base.mark_point(filled=True, size=65).encode(
            x="ame:Q",
            y=alt.Y("label:N", sort=y_order),
            color=shared_color,
            tooltip=[
                alt.Tooltip("label:N",    title="Variable"),
                alt.Tooltip("category:N", title="Group"),
                alt.Tooltip("ame_pp:Q",   title="AME (pp)", format=".2f"),
                alt.Tooltip("sig:N",      title="Significance"),
            ],
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"x": [0]}))
            .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=1)
            .encode(x="x:Q")
        )

        panel = (zero_line + ci_bars + dots).properties(width=300, height=380)

        chart = (
            panel
            .facet(facet=alt.Facet("outcome:N", title=""), columns=2)
            .properties(
                title=alt.TitleParams(
                    "Marginal Effects on Green Practice Adoption",
                    subtitle="Average marginal effects at the mean with 95% CIs. Clustered SEs by country.",
                    fontSize=14, fontWeight="bold",
                )
            )
        )
        _save_altair(chart, "altair_marginal_effects")
        return chart

    # ── 6d. EPI vs country fixed effect ───────────────────────────────────────

    def altair_epi_vs_country_fe(glm_models=glm_models):
        """Scatter: EPI score vs country fixed effect, both outcomes side-by-side.

        Mirrors plot_epi_vs_country_fe(). Maroon points are statistically
        significant (p < 0.05). Regression line from transform_regression.
        Hover for country name.
        """
        country_epi = (ES_firm_level[["country_name", "EPI.new"]]
                       .drop_duplicates("country_name")
                       .dropna(subset=["EPI.new"]))
        records = []
        for outcome, suffix in OUTCOMES:
            result  = glm_models[f"model_firm_1_{suffix}"]
            params  = result.params
            pvalues = result.pvalues
            mask    = params.index.str.startswith("C(country_name)[T.")
            fe_df   = pd.DataFrame({
                "country_name": [i.split("[T.")[1].rstrip("]") for i in params.index[mask]],
                "fe_coef":      params[mask].values,
                "pval":         pvalues[mask].values,
            })
            fe_df = fe_df.merge(country_epi, on="country_name", how="inner")
            fe_df["significant"] = fe_df["pval"] < 0.05
            fe_df["outcome"] = (
                "CO₂ Monitoring" if outcome == "monitors_co2_emissions"
                else "Energy Management"
            )
            records.append(fe_df)

        df = pd.concat(records, ignore_index=True)

        base = alt.Chart(df)

        scatter = base.mark_point(filled=True, opacity=0.75).encode(
            x=alt.X("EPI.new:Q", title="EPI Score (2024)"),
            y=alt.Y("fe_coef:Q", title="Country Fixed Effect (log-odds)"),
            color=alt.condition(
                alt.datum.significant,
                alt.value(_MAROON),
                alt.value("#bbbbbb"),
            ),
            size=alt.condition(
                alt.datum.significant,
                alt.value(65),
                alt.value(35),
            ),
            tooltip=[
                alt.Tooltip("country_name:N", title="Country"),
                alt.Tooltip("EPI.new:Q",       title="EPI Score",     format=".1f"),
                alt.Tooltip("fe_coef:Q",       title="Fixed Effect",  format=".3f"),
                alt.Tooltip("pval:Q",          title="p-value",       format=".3f"),
            ],
        )

        regression = (
            base
            .transform_regression("EPI.new", "fe_coef")
            .mark_line(color="firebrick", strokeWidth=2)
        )

        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(strokeDash=[4, 4], color="gray", strokeWidth=1)
            .encode(y="y:Q")
        )

        panel = (zero_line + regression + scatter).properties(width=300, height=280)

        chart = (
            panel
            .facet(facet=alt.Facet("outcome:N", title=""), columns=2)
            .properties(
                title=alt.TitleParams(
                    "EPI Score vs Country Baseline Adoption",
                    subtitle="Country fixed effects from firm-level logit. Maroon = p < 0.05. Hover for country.",
                    fontSize=14, fontWeight="bold",
                )
            )
        )
        _save_altair(chart, "altair_epi_vs_country_fe")
        return chart

    # ── 6e. B-Ready environmental score vs adoption rate ─────────────────────

    def altair_bready_vs_adoption(data=ES_firm_level):
        """Bubble chart: B-Ready env score vs country-level adoption rate.

        Mirrors plot_bready_vs_adoption(). Bubble size ∝ surveyed firms.
        Hover for country name and adoption rate.
        """
        records = []
        for outcome, suffix in OUTCOMES:
            grp = (
                data.groupby("country_name", observed=True)
                .agg(adoption_rate=(outcome, "mean"),
                     n_firms=(outcome, "count"),
                     br_env=("br_env", "first"))
                .dropna(subset=["adoption_rate", "br_env"])
                .reset_index()
            )
            grp["outcome"]  = (
                "CO₂ Monitoring" if outcome == "monitors_co2_emissions"
                else "Energy Management"
            )
            grp["rate_pct"] = (grp["adoption_rate"] * 100).round(1)
            records.append(grp)

        df = pd.concat(records, ignore_index=True)

        base = alt.Chart(df)

        bubbles = base.mark_point(filled=True, opacity=0.70).encode(
            x=alt.X("br_env:Q", title="B-Ready Environmental Score"),
            y=alt.Y("adoption_rate:Q", title="Adoption Rate",
                    axis=alt.Axis(format=".0%")),
            size=alt.Size("n_firms:Q",
                          scale=alt.Scale(range=[30, 500]),
                          legend=alt.Legend(title="Surveyed Firms")),
            color=alt.value(_MAROON),
            tooltip=[
                alt.Tooltip("country_name:N", title="Country"),
                alt.Tooltip("br_env:Q",       title="B-Ready Score",   format=".1f"),
                alt.Tooltip("rate_pct:Q",     title="Adoption Rate (%)", format=".1f"),
                alt.Tooltip("n_firms:Q",      title="Surveyed Firms"),
            ],
        )

        regression = (
            base
            .transform_regression("br_env", "adoption_rate")
            .mark_line(color="firebrick", strokeWidth=2)
        )

        panel = (regression + bubbles).properties(width=300, height=280)

        chart = (
            panel
            .facet(facet=alt.Facet("outcome:N", title=""), columns=2)
            .properties(
                title=alt.TitleParams(
                    "B-Ready Environmental Score vs Green Adoption Rate",
                    subtitle="Each bubble is a country. Bubble size ∝ number of surveyed firms. Hover for details.",
                    fontSize=14, fontWeight="bold",
                )
            )
        )
        _save_altair(chart, "altair_bready_vs_adoption")
        return chart

    # ── 6f. Random forest variable importance ─────────────────────────────────

    def altair_rf_importance(imp_df=importance_df):
        """Horizontal bar chart: top-20 RF variable importances, colored by category.

        Mirrors the matplotlib RF importance plot in run_random_forest().
        """
        top20 = imp_df.head(20).copy()
        cat_domain = list(top20["category"].unique())
        cat_range  = [_CAT_COLORS.get(c, "#cccccc") for c in cat_domain]

        chart = (
            alt.Chart(top20)
            .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
            .encode(
                x=alt.X("MeanDecreaseGini:Q", title="Mean Decrease in Gini Impurity"),
                y=alt.Y("variable_label:N", sort="-x", title=""),
                color=alt.Color("category:N",
                                scale=alt.Scale(domain=cat_domain, range=cat_range),
                                legend=alt.Legend(title="Variable Group")),
                tooltip=[
                    alt.Tooltip("variable_label:N", title="Variable"),
                    alt.Tooltip("category:N",        title="Group"),
                    alt.Tooltip("MeanDecreaseGini:Q", title="Importance", format=".4f"),
                ],
            )
            .properties(
                width=480, height=380,
                title=alt.TitleParams(
                    "Variable Importance: Energy Management Adoption",
                    subtitle="Random Forest (500 trees). Top 20 features by Mean Decrease in Gini.",
                    fontSize=14, fontWeight="bold",
                ),
            )
        )
        _save_altair(chart, "altair_rf_importance")
        return chart

    # ── Run all Altair plots ───────────────────────────────────────────────────

    print("\n=== Altair Visualizations ===")
    print("  Descriptive: adoption by income quartile…")
    altair_adoption_by_income()
    print("  Descriptive: adoption by firm characteristics…")
    altair_adoption_by_firm_chars()
    print("  Marginal effects forest plot…")
    altair_marginal_effects()
    print("  EPI vs country fixed effects…")
    altair_epi_vs_country_fe()
    print("  B-Ready vs adoption rate…")
    altair_bready_vs_adoption()
    print("  Random forest variable importance…")
    altair_rf_importance()
