import pandas as pd, numpy as np
from pathlib import Path
OUT = Path(__file__).resolve().parent.parent / "data" / "derived-data"
df = pd.read_csv(OUT / "ES_firm_level.csv", low_memory=False)

br = df["br_env"].dropna()
print(f"br_env: {br.nunique()} unique values, {df['br_env'].notna().sum()} non-null rows "
      f"out of {len(df)} ({df['br_env'].notna().mean():.1%})")
print(f"br_env range: {br.min():.1f} – {br.max():.1f}")

# countries with br_env
ctry = (df[["country_name","br_env"]].dropna()
        .drop_duplicates("country_name")
        .sort_values("br_env"))
print(f"\n{len(ctry)} countries have br_env")

# correlation of br_env with WGI vars (country level)
wgi = ["control_corruption_est","gov_effectiveness_est","political_stability_est",
       "regulatory_quality_est","rule_of_law_est","voice_accountability_est"]
cc = (df[["country_name","br_env"] + wgi]
      .drop_duplicates("country_name")
      .dropna())
print(f"\nCorrelations with br_env ({len(cc)} countries):")
print(cc[["br_env"] + wgi].corr()["br_env"].drop("br_env").round(3))

# outcome variation in br_env countries
for outcome in ["monitors_co2_emissions", "adopt_energy_management"]:
    sub = df[df["br_env"].notna()][[outcome,"country_name"]].dropna()
    var = sub.groupby("country_name")[outcome].nunique()
    print(f"\n{outcome}: {(var>1).sum()} of {len(var)} BR countries have variation")
