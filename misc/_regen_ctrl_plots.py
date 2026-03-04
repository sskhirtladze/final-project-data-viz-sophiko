"""Regenerate marginal_effects_ctrl_*.pdf/png from pre-built ES_firm_level.csv.

Run this instead of the full analysis.py when the Stata file loading fails
due to numpy 2.3.x memory issues.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import patsy
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
OUT      = REPO_DIR / "data" / "derived-data"

# ── Load pre-built firm-level dataset ────────────────────────────────────────
print("Loading ES_firm_level.csv …")
ES_firm_level = pd.read_csv(OUT / "ES_firm_level.csv", low_memory=False)
print(f"  {len(ES_firm_level):,} rows, {len(ES_firm_level.columns)} cols")

# ── Check required columns ────────────────────────────────────────────────────
need = [
    "monitors_co2_emissions", "adopt_energy_management",
    "num_employees", "ln_firm_age", "has_foreign_ownership",
    "has_female_owner", "owner_is_manager", "manager_experience_years",
    "manager_experience_years_sq", "manager_is_female",
    "has_training_programs", "introduced_new_products", "introduced_new_process",
    "has_rd_spending", "uses_foreign_tech_license", "has_website_social_media",
    "has_bank_account", "has_overdraft_facility", "has_external_audit",
    "experienced_climate_damage", "has_quality_certification",
    "is_exporter", "has_government_contract",
    "legal_status", "main_activity_type", "interview_year",
    "gdp_per_capita_ppp", "industry_share_gdp",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "EPI.new", "country_name",
]
missing = [c for c in need if c not in ES_firm_level.columns]
if missing:
    print(f"  WARNING: missing columns: {missing}")
else:
    print("  All required columns present.")

# ── Formula components (mirrors analysis.py) ─────────────────────────────────
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

COUNTRY_V1 = "gdp_per_capita_ppp + industry_share_gdp"
WGI_V = ("control_corruption_est + gov_effectiveness_est + "
         "political_stability_est + regulatory_quality_est + "
         "rule_of_law_est + voice_accountability_est")
FANDC_3    = f"{COUNTRY_V1} + {WGI_V} + Q('EPI.new')"
FANDC_BR   = f"{COUNTRY_V1} + {WGI_V} + br_env"

OUTCOMES = [("monitors_co2_emissions", "co"), ("adopt_energy_management", "em")]

# ── Variable metadata ─────────────────────────────────────────────────────────
_VAR_META = {
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
}

FIRM_VARS_PLOT = [
    "has_training_programs", "has_quality_certification", "has_rd_spending",
    "uses_foreign_tech_license", "introduced_new_process", "introduced_new_products",
    "has_external_audit", "has_website_social_media",
    "has_bank_account", "has_overdraft_facility",
    "num_employees", "ln_firm_age", "has_foreign_ownership",
    "owner_is_manager", "manager_is_female", "has_female_owner",
    "manager_experience_years",
    "has_government_contract", "is_exporter",
    "experienced_climate_damage",
]

CATEGORY_ORDER = ["Firm Capabilities", "Financial Access",
                  "Firm Characteristics", "Market Linkages", "Climate Exposure"]

COUNTRY_VARS_PLOT = [
    "gdp_per_capita_ppp", "industry_share_gdp",
    "control_corruption_est", "gov_effectiveness_est", "political_stability_est",
    "regulatory_quality_est", "rule_of_law_est", "voice_accountability_est",
    "EPI.new", "br_env",
]

COUNTRY_CATEGORY_ORDER = ["Country Context", "Governance", "ENV Governance"]


# ── logit helpers ─────────────────────────────────────────────────────────────
def _fit_ctrl(formula, data):
    y_cc, _ = patsy.dmatrices(formula, data=data,
                               return_type="dataframe", NA_action="drop")
    groups_cc = data.loc[y_cc.index, "country_name"].values
    return smf.logit(formula, data=data).fit(
        maxiter=300, disp=False,
        cov_type="cluster",
        cov_kwds={"groups": groups_cc},
    )

def logit_ctrl(outcome, data=None):
    if data is None:
        data = ES_firm_level
    return _fit_ctrl(f"{outcome} ~ {FIRM_RHS} + {FANDC_3} + C(interview_year)", data)

def logit_ctrl_br(outcome, data=None):
    """Tries GDP+WGI+br_env first; falls back to GDP+br_env if Hessian singular.

    B-Ready data only spans 2022-23; year dummies for 2024/2025 are all-zero
    in this subset, causing rank deficiency.  We detect and remove zero-variance
    dummies before fitting.
    """
    if data is None:
        data = ES_firm_level
    variation = (data.groupby("country_name")[outcome]
                 .apply(lambda s: s.dropna().nunique()))
    data_br = data[data["country_name"].isin(variation[variation > 1].index)].copy()

    # Keep only years present in the B-Ready subset
    br_years = data_br["interview_year"].dropna().unique().tolist()
    data_br = data_br[data_br["interview_year"].isin(br_years)].copy()

    for suffix in [FANDC_BR, f"{COUNTRY_V1} + br_env"]:
        formula = f"{outcome} ~ {FIRM_RHS} + {suffix} + C(interview_year)"
        # Detect zero-variance dummies (remaining all-zero columns after year filter)
        _, X_temp = patsy.dmatrices(formula, data=data_br,
                                     return_type="dataframe", NA_action="drop")
        zero_cols = [c for c in X_temp.columns
                     if c != "Intercept" and X_temp[c].sum() == 0]
        if zero_cols:
            # Drop interview_year levels that are all-zero
            bad_years = set()
            for c in zero_cols:
                if "interview_year" in c:
                    try:
                        bad_years.add(float(c.split("[T.")[1].rstrip("]")))
                    except (IndexError, ValueError):
                        pass
            if bad_years:
                data_br = data_br[~data_br["interview_year"].isin(bad_years)].copy()
                print(f"    Dropped year dummies: {sorted(bad_years)}")
        try:
            return _fit_ctrl(formula, data_br)
        except np.linalg.LinAlgError:
            print(f"    Singular Hessian with '{suffix}', trying simpler spec…")
    raise RuntimeError(f"Could not fit B-Ready logit for {outcome}")


# ── plot_marginal_effects_ctrl ────────────────────────────────────────────────
def plot_marginal_effects_ctrl(result, outcome, suffix, result_br=None):
    mfx = result.get_margeff(at="mean", method="dydx", dummy=True).summary_frame()
    mfx.index = mfx.index.str.replace(r"Q\('(.+?)'\)", r"\1", regex=True)

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
        panel = mfx[mfx.index.isin(var_list)].copy()
        if panel.empty:
            ax.set_visible(False)
            return
        panel["label"]    = [_VAR_META.get(v, (v, "Other"))[0] for v in panel.index]
        panel["category"] = [_VAR_META.get(v, (v, "Other"))[1] for v in panel.index]
        cat_rank = {c: i for i, c in enumerate(cat_order)}
        panel["cat_rank"] = [cat_rank.get(c, len(cat_order)) for c in panel["category"]]
        panel["abs_ame"]  = panel[ame_col].abs()
        panel = panel.sort_values(["cat_rank", "abs_ame"], ascending=[True, False]).reset_index()
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
            ax.plot(row[ame_col], yp, "o", color=color, markersize=6, zorder=3)
            ax.hlines(yp, row[ci_lo_col], row[ci_hi_col], color=color, linewidth=1.5, zorder=2)
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
            ax.text(-0.28, cat_rows["y_pos"].mean(), cat,
                    transform=yaxis_trans, va="center", ha="right",
                    fontweight="bold", fontsize=8,
                    color=cat_colors.get(cat, "grey"), clip_on=False)
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
            print(f"  Warning: could not save {path.name} (file locked)")
    plt.close(fig)
    print(f"  Saved marginal_effects_ctrl_{suffix}.png/.pdf")


# ── Main ──────────────────────────────────────────────────────────────────────
print("\n=== Controls-model logit (WDI + WGI + EPI) ===")
ctrl_models = {}
for outcome, suffix in OUTCOMES:
    print(f"  Fitting EPI model for {outcome} …")
    ctrl_models[suffix] = logit_ctrl(outcome)

print("\n=== Controls-model logit (WDI + WGI + B-Ready) ===")
ctrl_models_br = {}
for outcome, suffix in OUTCOMES:
    print(f"  Fitting B-Ready model for {outcome} …")
    ctrl_models_br[suffix] = logit_ctrl_br(outcome)

print("\n=== Marginal Effects Forest Plots — Controls Model ===")
for outcome, suffix in OUTCOMES:
    print(f"  Computing AMEs for {outcome} …")
    plot_marginal_effects_ctrl(ctrl_models[suffix], outcome, suffix,
                               result_br=ctrl_models_br[suffix])

print("\nDone.")
