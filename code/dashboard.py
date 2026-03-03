"""
Environmental Governance & Green Adoption — Streamlit Dashboard
===============================================================
Reads pre-built outputs from data/derived-data/.
Run analysis.py first to generate ES_firm_level.csv.

Run:
    conda activate DAP
    pip install streamlit plotly   # once, if not already installed
    streamlit run code/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Environmental Governance Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS & DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
REPO_DIR = Path(__file__).resolve().parent.parent
DERIVED  = REPO_DIR / "data" / "derived-data"


@st.cache_data
def load_country_data():
    """Aggregate ES_firm_level.csv to one row per country."""
    es = pd.read_csv(DERIVED / "ES_firm_level.csv", low_memory=False)

    grp = (
        es.groupby("country_name", observed=True)
        .agg(
            iso_code             = ("Country Code",            "first"),
            co2_monitoring_rate  = ("monitors_co2_emissions",  "mean"),
            energy_mgmt_rate     = ("adopt_energy_management", "mean"),
            n_firms              = ("monitors_co2_emissions",  "count"),
            epi_score            = ("EPI.new",                 "first"),
            br_env               = ("br_env",                  "first"),
            gdp_per_capita       = ("gdp_per_capita_ppp",      "first"),
            control_corruption   = ("control_corruption_est",  "first"),
            gov_effectiveness    = ("gov_effectiveness_est",   "first"),
            political_stability  = ("political_stability_est", "first"),
            regulatory_quality   = ("regulatory_quality_est",  "first"),
            rule_of_law          = ("rule_of_law_est",         "first"),
            voice_accountability = ("voice_accountability_est","first"),
            br_pc1               = ("PC1",                     "first"),
            reg_fr               = ("reg_fr",                  "first"),
            pub_ser              = ("pub_ser",                  "first"),
            op_eff               = ("op_eff",                  "first"),
        )
        .reset_index()
    )

    # Convert rates to percentages for display
    grp["co2_monitoring_pct"] = grp["co2_monitoring_rate"] * 100
    grp["energy_mgmt_pct"]    = grp["energy_mgmt_rate"]    * 100

    return grp


country_df = load_country_data()

# ══════════════════════════════════════════════════════════════════════════════
# 2.  VARIABLE CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════
# (display label → (column, colorscale))
COLORBY = {
    # Green adoption
    "CO₂ Monitoring Rate (%)":        ("co2_monitoring_pct",  "RdYlGn"),
    "Energy Management Rate (%)":     ("energy_mgmt_pct",     "RdYlGn"),
    # Environmental indices
    "EPI Score (2024)":               ("epi_score",           "RdYlGn"),
    "B-Ready Environmental Score":    ("br_env",              "RdYlGn"),
    "B-Ready PC1":                    ("br_pc1",              "RdYlGn"),
    "B-Ready Regulatory Framework":   ("reg_fr",              "RdYlGn"),
    "B-Ready Public Services":        ("pub_ser",             "RdYlGn"),
    "B-Ready Operational Efficiency": ("op_eff",              "RdYlGn"),
    # WGI governance
    "Control of Corruption":          ("control_corruption",  "RdYlGn"),
    "Government Effectiveness":       ("gov_effectiveness",   "RdYlGn"),
    "Political Stability":            ("political_stability", "RdYlGn"),
    "Regulatory Quality":             ("regulatory_quality",  "RdYlGn"),
    "Rule of Law":                    ("rule_of_law",         "RdYlGn"),
    "Voice & Accountability":         ("voice_accountability","RdYlGn"),
    # Economic
    "GDP per Capita (log PPP)":       ("gdp_per_capita",      "Blues"),
}

# Human-readable hover labels
HOVER_LABELS = {
    "co2_monitoring_pct":  "CO₂ Monitoring (%)",
    "energy_mgmt_pct":     "Energy Mgmt (%)",
    "epi_score":           "EPI Score",
    "br_env":              "B-Ready Env. Score",
    "n_firms":             "Surveyed firms",
}

WGI_ROWS = [
    ("Control of Corruption",    "control_corruption"),
    ("Government Effectiveness", "gov_effectiveness"),
    ("Political Stability",      "political_stability"),
    ("Regulatory Quality",       "regulatory_quality"),
    ("Rule of Law",              "rule_of_law"),
    ("Voice & Accountability",   "voice_accountability"),
]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🗺 Map Controls")

    # Group options in the selectbox using option groups
    color_label = st.selectbox(
        "Color map by:",
        options=list(COLORBY.keys()),
        index=0,
        help="Recolors the choropleth. Click any country to see its full profile.",
    )
    color_var, color_scale = COLORBY[color_label]

    st.markdown("---")
    st.markdown("**Country selector**")
    st.caption("Use this if map click doesn't register.")
    country_names   = sorted(country_df["country_name"].dropna().tolist())
    sidebar_country = st.selectbox("Country", ["(none)"] + country_names)

    st.markdown("---")
    st.caption(
        "Data: World Bank Enterprise Survey, B-Ready Index, "
        "EPI 2024, WDI, WGI. Adoption rates = share of surveyed firms."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 4.  TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.title("🌍 Environmental Governance & Green Adoption Dashboard")
st.caption(
    "Click any country on the map to see its full indicator profile. "
    "Change the coloring variable from the sidebar dropdown."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHOROPLETH MAP
# ══════════════════════════════════════════════════════════════════════════════
plot_df = country_df.dropna(subset=["iso_code", color_var])

fig_map = px.choropleth(
    plot_df,
    locations  = "iso_code",
    color      = color_var,
    hover_name = "country_name",
    hover_data = {
        "iso_code":           False,
        "co2_monitoring_pct": ":.1f",
        "energy_mgmt_pct":    ":.1f",
        "epi_score":          ":.1f",
        "br_env":             ":.1f",
        "n_firms":            True,
    },
    color_continuous_scale = color_scale,
    projection = "natural earth",
    labels     = {**HOVER_LABELS, color_var: color_label},
)
fig_map.update_layout(
    height = 520,
    margin = dict(l=0, r=0, t=10, b=0),
    coloraxis_colorbar = dict(
        title     = color_label,
        thickness = 15,
        len       = 0.6,
        tickfont  = dict(size=10),
    ),
    paper_bgcolor = "#f9f9f9",
)
fig_map.update_geos(
    showframe      = False,
    showcoastlines = True,
    coastlinecolor = "LightGrey",
    showland       = True,
    landcolor      = "#f0f0f0",
    showocean      = True,
    oceancolor     = "#e8f4f8",
    bgcolor        = "#f9f9f9",
)

# Map click interaction requires Streamlit ≥ 1.35
try:
    map_event = st.plotly_chart(
        fig_map,
        use_container_width=True,
        on_select="rerun",
        key="world_map",
    )
except TypeError:
    st.plotly_chart(fig_map, use_container_width=True)
    map_event = None
    st.info(
        "ℹ️ Map-click interaction requires Streamlit ≥ 1.35.  "
        "Use the country selector in the sidebar instead."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 6.  RESOLVE SELECTED COUNTRY  (map click takes priority over sidebar)
# ══════════════════════════════════════════════════════════════════════════════
selected_name = None

if map_event is not None:
    selection = getattr(map_event, "selection", None)
    if selection is not None:
        pts = getattr(selection, "points", [])
        if pts:
            iso_clicked = pts[0].get("location")
            hit = country_df[country_df["iso_code"] == iso_clicked]
            if not hit.empty:
                selected_name = hit.iloc[0]["country_name"]

if selected_name is None and sidebar_country != "(none)":
    selected_name = sidebar_country

# ══════════════════════════════════════════════════════════════════════════════
# 7.  COUNTRY PROFILE PANEL
# ══════════════════════════════════════════════════════════════════════════════
st.divider()

if selected_name is None:
    st.info("👆 Click a country on the map, or pick one from the sidebar, to view its profile.")
    st.stop()

row = country_df[country_df["country_name"] == selected_name].iloc[0]
st.subheader(f"📊 {selected_name}")

# --- helper: st.metric with ±delta vs cross-country mean ---
def _metric(label, value, col_name, fmt=".1f", suffix=""):
    if pd.isna(value):
        st.metric(label, "N/A")
        return
    avg   = country_df[col_name].mean(skipna=True)
    delta = value - avg
    st.metric(
        label,
        f"{value:{fmt}}{suffix}",
        delta       = f"{delta:+{fmt}} vs avg",
        delta_color = "normal",
    )

col1, col2, col3 = st.columns([1.2, 1.2, 1.8])

# ── Column 1: Green adoption ────────────────────────────────────────────────
with col1:
    st.markdown("**🌿 Green Adoption**")
    _metric(
        "CO₂ Monitoring Rate",
        row["co2_monitoring_pct"], "co2_monitoring_pct",
        fmt=".1f", suffix="%",
    )
    _metric(
        "Energy Management Rate",
        row["energy_mgmt_pct"], "energy_mgmt_pct",
        fmt=".1f", suffix="%",
    )
    n = row["n_firms"]
    st.metric("Surveyed Firms", f"{int(n):,}" if pd.notna(n) else "N/A")

    st.markdown("**🏭 B-Ready**")
    _metric("Environmental Score", row["br_env"],   "br_env",   fmt=".1f")
    _metric("PC1 (env)",           row["br_pc1"],   "br_pc1",   fmt=".2f")
    _metric("Regulatory Framework",row["reg_fr"],   "reg_fr",   fmt=".1f")
    _metric("Public Services",     row["pub_ser"],  "pub_ser",  fmt=".1f")
    _metric("Operational Eff.",    row["op_eff"],   "op_eff",   fmt=".1f")

# ── Column 2: Environmental & economic ──────────────────────────────────────
with col2:
    st.markdown("**🌏 Environmental & Economic**")
    _metric("EPI Score (2024)",         row["epi_score"],      "epi_score",      fmt=".1f")
    _metric("GDP per Capita (log PPP)", row["gdp_per_capita"], "gdp_per_capita", fmt=".2f")

    st.markdown("**🏛 WGI (raw estimates)**")
    for label, var in WGI_ROWS:
        _metric(label, row[var], var, fmt=".2f")

# ── Column 3: WGI bar chart vs cross-country average ────────────────────────
with col3:
    st.markdown("**🏛 Governance Profile vs Cross-country Average**")

    wgi_labels = [l for l, _ in WGI_ROWS]
    wgi_vals   = [row[v]                          for _, v in WGI_ROWS]
    wgi_avgs   = [country_df[v].mean(skipna=True) for _, v in WGI_ROWS]

    wgi_df = pd.DataFrame({
        "Indicator": wgi_labels,
        "Country":   wgi_vals,
        "Average":   wgi_avgs,
    }).dropna(subset=["Country"])

    if wgi_df.empty:
        st.write("No WGI data available.")
    else:
        fig_wgi = go.Figure()

        fig_wgi.add_trace(go.Bar(
            y            = wgi_df["Indicator"],
            x            = wgi_df["Country"],
            orientation  = "h",
            name         = selected_name,
            marker_color = "steelblue",
            opacity      = 0.80,
        ))
        fig_wgi.add_trace(go.Scatter(
            y      = wgi_df["Indicator"],
            x      = wgi_df["Average"],
            mode   = "markers",
            name   = "Cross-country avg",
            marker = dict(symbol="diamond", size=10, color="firebrick"),
        ))
        fig_wgi.add_vline(
            x=0, line_dash="dash", line_color="grey", line_width=1
        )
        fig_wgi.update_layout(
            height     = 300,
            margin     = dict(l=0, r=10, t=10, b=30),
            showlegend = True,
            legend     = dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=10),
            ),
            xaxis_title = "Estimate  (WGI scale ≈ −2.5 to +2.5)",
            yaxis       = dict(tickfont=dict(size=11)),
            bargap      = 0.35,
            plot_bgcolor= "white",
        )
        st.plotly_chart(fig_wgi, use_container_width=True)

    # Adoption rate comparison bar chart
    st.markdown("**🌿 Adoption Rates vs Average**")

    adopt_labels = ["CO₂ Monitoring", "Energy Mgmt"]
    adopt_vals   = [row["co2_monitoring_pct"], row["energy_mgmt_pct"]]
    adopt_avgs   = [
        country_df["co2_monitoring_pct"].mean(skipna=True),
        country_df["energy_mgmt_pct"].mean(skipna=True),
    ]

    adopt_df = pd.DataFrame({
        "Outcome": adopt_labels,
        "Country": adopt_vals,
        "Average": adopt_avgs,
    }).dropna(subset=["Country"])

    if not adopt_df.empty:
        fig_adopt = go.Figure()
        fig_adopt.add_trace(go.Bar(
            x            = adopt_df["Outcome"],
            y            = adopt_df["Country"],
            name         = selected_name,
            marker_color = "steelblue",
            opacity      = 0.80,
        ))
        fig_adopt.add_trace(go.Scatter(
            x      = adopt_df["Outcome"],
            y      = adopt_df["Average"],
            mode   = "markers",
            name   = "Cross-country avg",
            marker = dict(symbol="diamond", size=10, color="firebrick"),
        ))
        fig_adopt.update_layout(
            height      = 220,
            margin      = dict(l=0, r=0, t=10, b=10),
            showlegend  = True,
            legend      = dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=10),
            ),
            yaxis_title = "Adoption Rate (%)",
            yaxis       = dict(range=[0, 100]),
            plot_bgcolor= "white",
        )
        st.plotly_chart(fig_adopt, use_container_width=True)
