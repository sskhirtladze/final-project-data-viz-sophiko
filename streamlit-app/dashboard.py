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
    """Load pre-built country-level dashboard dataset."""
    return pd.read_csv(DERIVED / "dashboard_country_data.csv")


country_df = load_country_data()

# ══════════════════════════════════════════════════════════════════════════════
# 2.  VARIABLE CATALOGUE
# ══════════════════════════════════════════════════════════════════════════════
COLORBY = {
    "EPI — Overall":                ("epi_score",          "RdYlGn"),
    "CO₂ Monitoring Rate (%)":      ("co2_monitoring_pct", "RdYlGn"),
    "Energy Management Rate (%)":   ("energy_mgmt_pct",    "RdYlGn"),
    "B-Ready Environmental Score":  ("br_env",             "RdYlGn"),
    "EPI — Ecosystem Vitality":     ("epi_eco",            "RdYlGn"),
    "EPI — Environmental Health":   ("epi_hlt",            "RdYlGn"),
    "EPI — Biodiversity & Habitat": ("epi_bdh",            "RdYlGn"),
    "EPI — Climate Change":         ("epi_cch",            "RdYlGn"),
    "EPI — Air Quality":            ("epi_air",            "RdYlGn"),
    "EPI — Water & Sanitation":     ("epi_h2o",            "RdYlGn"),
    "EPI — Waste Management":       ("epi_wmg",            "RdYlGn"),
    "EPI — Agriculture":            ("epi_agr",            "RdYlGn"),
}

HOVER_LABELS = {
    "co2_monitoring_pct": "CO₂ Monitoring (%)",
    "energy_mgmt_pct":    "Energy Mgmt (%)",
    "epi_score":          "EPI Score",
    "br_env":             "B-Ready Env. Score",
    "n_firms":            "Surveyed firms",
}

EPI_SUBS = [
    ("Ecosystem Vitality",    "epi_eco"),
    ("Environmental Health",  "epi_hlt"),
    ("Biodiversity & Habitat","epi_bdh"),
    ("Climate Change",        "epi_cch"),
    ("Air Quality",           "epi_air"),
    ("Water & Sanitation",    "epi_h2o"),
    ("Waste Management",      "epi_wmg"),
    ("Agriculture",           "epi_agr"),
]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🗺 Map Controls")

    color_label = st.selectbox(
        "Select Indicator:",
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
st.title("Environmental Governance Dashboard")
st.caption(
    "Click any country on the map to see its full profile. "
    "Change the indicator from the sidebar dropdown."
)

# ══════════════════════════════════════════════════════════════════════════════
# 5.  CHOROPLETH MAP  (all countries with an ISO code; missing data → grey)
# ══════════════════════════════════
plot_df = country_df.dropna(subset=["iso_code"])

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

# ── Column 1: Green adoption + B-Ready ──────────────────────────────────────
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
    _metric("Environmental Score", row["br_env"], "br_env", fmt=".1f")

# ── Column 2: EPI overall + sub-indicators ──────────────────────────────────
with col2:
    st.markdown("**🌏 EPI (Environmental Performance Index)**")
    _metric("EPI Score (2024)", row["epi_score"], "epi_score", fmt=".1f")
    st.markdown("*Sub-indicators:*")
    for label, var in EPI_SUBS:
        _metric(label, row[var], var, fmt=".1f")

# ── Column 3: EPI sub-indicators bar chart + adoption bar chart ─────────────
with col3:
    st.markdown("**🌏 EPI Sub-indicators vs Cross-country Average**")

    epi_df_plot = pd.DataFrame({
        "Indicator": [l for l, _ in EPI_SUBS],
        "Country":   [row[v] for _, v in EPI_SUBS],
        "Average":   [country_df[v].mean(skipna=True) for _, v in EPI_SUBS],
    }).dropna(subset=["Country"])

    if epi_df_plot.empty:
        st.write("No EPI data available.")
    else:
        fig_epi = go.Figure()
        fig_epi.add_trace(go.Bar(
            y            = epi_df_plot["Indicator"],
            x            = epi_df_plot["Country"],
            orientation  = "h",
            name         = selected_name,
            marker_color = "steelblue",
            opacity      = 0.80,
        ))
        fig_epi.add_trace(go.Scatter(
            y      = epi_df_plot["Indicator"],
            x      = epi_df_plot["Average"],
            mode   = "markers",
            name   = "Cross-country avg",
            marker = dict(symbol="diamond", size=10, color="firebrick"),
        ))
        fig_epi.update_layout(
            height       = 300,
            margin       = dict(l=0, r=10, t=10, b=30),
            showlegend   = True,
            legend       = dict(
                orientation="h", yanchor="bottom", y=1.02, x=0,
                font=dict(size=10),
            ),
            xaxis_title  = "EPI Score (0–100)",
            yaxis        = dict(tickfont=dict(size=11)),
            bargap       = 0.35,
            plot_bgcolor = "white",
        )
        st.plotly_chart(fig_epi, use_container_width=True)

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
