"""
Hotel No-Show Interactive Dashboard
Run: streamlit run app.py
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hotel No-Show Dashboard",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading & feature engineering ───────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fea_table.csv")
    df["booking_value"] = df["Price_SGD"] * df["Length_of_Stay_Days"]

    # Segment-level no_show_rate and retention_rate for CLV
    seg = df.groupby(["Branch", "country"]).agg(
        seg_no_show_rate=("NoShow", "mean"),
        seg_first_time_rate=("first_time", lambda x: (x == "Yes").mean()),
    ).reset_index()
    seg["seg_retention_rate"] = 1 - seg["seg_first_time_rate"]
    seg["seg_churn_rate"] = seg["seg_first_time_rate"].clip(0.01, 0.99)

    df = df.merge(seg, on=["Branch", "country"], how="left")
    df["expected_value"] = df["booking_value"] * (1 - df["seg_no_show_rate"])
    df["CLV"] = df["expected_value"] / df["seg_churn_rate"]
    return df

df = load_data()

TIER_COLORS = {
    "Low (<25%)":      "#27AE60",
    "Medium (25-40%)": "#F39C12",
    "High (40-55%)":   "#E67E22",
    "Critical (>55%)": "#C0392B",
}

def assign_tier(rate):
    if rate < 0.25:   return "Low (<25%)"
    elif rate < 0.40: return "Medium (25-40%)"
    elif rate < 0.55: return "High (40-55%)"
    else:             return "Critical (>55%)"

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/hotel.png", width=60)
st.sidebar.title("Drill-Down Filters")

all_branches = sorted(df["Branch"].unique())
all_countries = sorted(df["country"].unique())
all_rooms = sorted(df["RoomType"].unique())
all_platforms = sorted(df["platform"].unique())

sel_branch = st.sidebar.multiselect(
    "Branch", all_branches, default=all_branches, help="Filter by hotel branch"
)
sel_country = st.sidebar.multiselect(
    "Country", all_countries, default=all_countries, help="Filter by guest country of origin"
)
sel_room = st.sidebar.multiselect(
    "Room Type", all_rooms, default=all_rooms
)
sel_platform = st.sidebar.multiselect(
    "Booking Platform", all_platforms, default=all_platforms
)
sel_first_time = st.sidebar.radio(
    "Guest Type", ["All", "First-Time Only", "Returning Only"], index=0
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: fea_table.csv · 97,777 bookings")

# ── Apply filters ─────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_branch:
    fdf = fdf[fdf["Branch"].isin(sel_branch)]
if sel_country:
    fdf = fdf[fdf["country"].isin(sel_country)]
if sel_room:
    fdf = fdf[fdf["RoomType"].isin(sel_room)]
if sel_platform:
    fdf = fdf[fdf["platform"].isin(sel_platform)]
if sel_first_time == "First-Time Only":
    fdf = fdf[fdf["first_time"] == "Yes"]
elif sel_first_time == "Returning Only":
    fdf = fdf[fdf["first_time"] == "No"]

# ── Header ───────────────────────────────────────────────────────────────────
st.title("🏨 Hotel No-Show Analytics Dashboard")
st.caption("Cohort analysis · CLV estimation · Risk scoring — filter using the sidebar to drill down by branch or country.")

if fdf.empty:
    st.warning("No data matches the current filters. Adjust the sidebar selection.")
    st.stop()

# ── KPI Row ───────────────────────────────────────────────────────────────────
total = len(fdf)
no_show_rate = fdf["NoShow"].mean()
avg_clv = fdf["CLV"].mean()
avg_bv = fdf["booking_value"].mean()
avg_los = fdf["Length_of_Stay_Days"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Bookings", f"{total:,}")
k2.metric("No-Show Rate", f"{no_show_rate:.1%}",
          delta=f"{no_show_rate - df['NoShow'].mean():.1%} vs overall",
          delta_color="inverse")
k3.metric("Avg Booking Value", f"${avg_bv:,.0f}")
k4.metric("Avg Simplified CLV", f"${avg_clv:,.0f}")
k5.metric("Avg Length of Stay", f"{avg_los:.1f} nights")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Cohort Analysis",
    "💰 CLV Estimation",
    "⚠️ Risk Scoring",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — COHORT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("No-Show Rate by Cohort Dimension")

    # ── Row 1: Branch + Country ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        g = fdf.groupby("Branch").agg(
            no_show_rate=("NoShow", "mean"),
            n=("NoShow", "count"),
        ).reset_index().sort_values("no_show_rate", ascending=False)
        g["label"] = (g["no_show_rate"] * 100).round(1).astype(str) + "%"

        fig = px.bar(g, x="Branch", y="no_show_rate", text="label",
                     color="Branch", color_discrete_sequence=["#1976D2", "#FF5722"],
                     title="No-Show Rate by Branch")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="No-Show Rate",
                          showlegend=False, height=350)
        fig.add_hline(y=df["NoShow"].mean(), line_dash="dot",
                      line_color="red", annotation_text="Overall avg")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        g = fdf.groupby("country").agg(
            no_show_rate=("NoShow", "mean"),
            n=("NoShow", "count"),
        ).reset_index().sort_values("no_show_rate", ascending=False)
        g["label"] = (g["no_show_rate"] * 100).round(1).astype(str) + "%"

        fig = px.bar(g, x="country", y="no_show_rate", text="label",
                     color="country",
                     title="No-Show Rate by Country")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="No-Show Rate",
                          showlegend=False, height=350)
        fig.add_hline(y=df["NoShow"].mean(), line_dash="dot",
                      line_color="red", annotation_text="Overall avg")
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Platform + First-Time ──────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        g = fdf.groupby("platform").agg(
            no_show_rate=("NoShow", "mean"),
            n=("NoShow", "count"),
        ).reset_index().sort_values("no_show_rate", ascending=False)
        g["label"] = (g["no_show_rate"] * 100).round(1).astype(str) + "%"

        fig = px.bar(g, x="platform", y="no_show_rate", text="label",
                     color="platform", color_discrete_sequence=px.colors.qualitative.Safe,
                     title="No-Show Rate by Booking Platform")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
        fig.add_hline(y=df["NoShow"].mean(), line_dash="dot", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        g = fdf.groupby("first_time").agg(
            no_show_rate=("NoShow", "mean"),
            n=("NoShow", "count"),
        ).reset_index()
        g["label"] = (g["no_show_rate"] * 100).round(1).astype(str) + "%"
        g["Guest Type"] = g["first_time"].map({"Yes": "First-Time", "No": "Returning"})

        fig = px.bar(g, x="Guest Type", y="no_show_rate", text="label",
                     color="Guest Type", color_discrete_map={
                         "First-Time": "#E53935", "Returning": "#43A047"},
                     title="No-Show Rate: First-Time vs Returning Guest")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=350)
        fig.add_hline(y=df["NoShow"].mean(), line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)

    # ── Heatmap: Branch × Country ─────────────────────────────────────────────
    st.subheader("Cross-Tab Heatmap: Branch × Country")
    pivot = fdf.groupby(["Branch", "country"])["NoShow"].mean().unstack().round(3)

    fig = px.imshow(
        pivot,
        text_auto=".1%",
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=0.75,
        title="No-Show Rate Heatmap (Branch × Country)",
        aspect="auto",
    )
    fig.update_layout(height=280, coloraxis_colorbar_title="No-Show Rate")
    st.plotly_chart(fig, use_container_width=True)

    # ── Detailed cohort table ─────────────────────────────────────────────────
    with st.expander("Full Cohort Table (Branch × Country × Platform × First-Time)"):
        cohort = fdf.groupby(["Branch", "country", "platform", "first_time"]).agg(
            n=("NoShow", "count"),
            no_show_rate=("NoShow", "mean"),
        ).reset_index()
        cohort["no_show_pct"] = (cohort["no_show_rate"] * 100).round(1)
        cohort["risk_tier"] = cohort["no_show_rate"].apply(assign_tier)
        cohort = cohort.sort_values("no_show_rate", ascending=False)
        cohort.columns = ["Branch", "Country", "Platform", "First-Time", "N", "No-Show Rate", "No-Show %", "Risk Tier"]
        st.dataframe(
            cohort.style.background_gradient(subset=["No-Show %"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CLV ESTIMATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Simplified Customer Lifetime Value Estimation")
    st.markdown(
        """
        **Formula:** `CLV = (Price_SGD × LoS × P_show) / churn_rate`
        where `P_show = 1 − segment no-show rate` and `churn_rate ≈ first-time guest rate` (loyalty proxy).
        """
    )

    # Segment-level CLV
    clv_seg = fdf.groupby(["Branch", "country"]).agg(
        n=("NoShow", "count"),
        no_show_rate=("NoShow", "mean"),
        first_time_rate=("first_time", lambda x: (x == "Yes").mean()),
        avg_price=("Price_SGD", "mean"),
        avg_los=("Length_of_Stay_Days", "mean"),
        avg_clv=("CLV", "mean"),
        total_expected_value=("expected_value", "sum"),
    ).reset_index()
    clv_seg = clv_seg.round(2)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            clv_seg.sort_values("avg_clv", ascending=True),
            x="avg_clv", y="Branch",
            color="country", barmode="group",
            orientation="h",
            title="Average CLV by Branch & Country (SGD)",
            labels={"avg_clv": "Avg CLV (SGD)", "Branch": ""},
        )
        fig.update_layout(height=380, xaxis_tickprefix="$", legend_title="Country")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            clv_seg,
            x="no_show_rate", y="avg_clv",
            size="n", color="country",
            symbol="Branch",
            hover_name="country",
            hover_data={"Branch": True, "avg_price": ":.2f", "avg_los": ":.1f", "n": True},
            title="CLV vs No-Show Rate (bubble = bookings count)",
            labels={"no_show_rate": "No-Show Rate", "avg_clv": "Avg CLV (SGD)"},
        )
        fig.update_layout(height=380, xaxis_tickformat=".0%", yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

    # CLV vs Price × LoS scatter
    st.subheader("CLV Components: Price vs Length of Stay")
    sample = fdf.sample(min(5000, len(fdf)), random_state=42)
    fig = px.scatter(
        sample,
        x="Length_of_Stay_Days", y="Price_SGD",
        color="CLV",
        color_continuous_scale="Viridis",
        facet_col="Branch",
        opacity=0.6,
        hover_data=["country", "RoomType", "CLV"],
        title="Price × Length of Stay coloured by CLV (sample of 5,000)",
        labels={"Price_SGD": "Price/Night (SGD)", "Length_of_Stay_Days": "Nights"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Segment table
    with st.expander("Segment CLV Table"):
        display_cols = {
            "Branch": "Branch", "country": "Country", "n": "N",
            "avg_price": "Avg Price/Night", "avg_los": "Avg Nights",
            "no_show_rate": "No-Show Rate", "first_time_rate": "First-Timer Rate",
            "avg_clv": "Avg CLV (SGD)", "total_expected_value": "Total Expected Rev (SGD)",
        }
        tbl = clv_seg[list(display_cols.keys())].rename(columns=display_cols)
        tbl = tbl.sort_values("Avg CLV (SGD)", ascending=False)
        st.dataframe(
            tbl.style
               .format({"Avg Price/Night": "${:.0f}", "Avg Nights": "{:.1f}",
                        "No-Show Rate": "{:.1%}", "First-Timer Rate": "{:.1%}",
                        "Avg CLV (SGD)": "${:,.0f}", "Total Expected Rev (SGD)": "${:,.0f}"})
               .background_gradient(subset=["Avg CLV (SGD)"], cmap="Greens"),
            use_container_width=True,
            hide_index=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — RISK SCORING
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("No-Show Risk Scoring by Branch / Country / Room Type")
    st.markdown(
        "Each **Branch × Country × Room Type** cell is assigned a risk tier based on its historical no-show rate."
    )

    risk = fdf.groupby(["Branch", "country", "RoomType"]).agg(
        n=("NoShow", "count"),
        no_show_rate=("NoShow", "mean"),
    ).reset_index()
    risk["no_show_pct"] = (risk["no_show_rate"] * 100).round(1)
    risk["risk_tier"] = risk["no_show_rate"].apply(assign_tier)
    risk["risk_tier_order"] = risk["risk_tier"].map({
        "Low (<25%)": 0, "Medium (25-40%)": 1, "High (40-55%)": 2, "Critical (>55%)": 3
    })

    # ── Tier summary bar ─────────────────────────────────────────────────────
    tier_counts = risk["risk_tier"].value_counts().reindex(list(TIER_COLORS.keys())).fillna(0)
    fig_tier = px.bar(
        x=tier_counts.index, y=tier_counts.values,
        color=tier_counts.index,
        color_discrete_map=TIER_COLORS,
        text=tier_counts.values.astype(int),
        title="Segments per Risk Tier (Branch × Country × Room Type)",
        labels={"x": "Risk Tier", "y": "# Segments"},
    )
    fig_tier.update_traces(textposition="outside")
    fig_tier.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig_tier, use_container_width=True)

    # ── Heatmaps side by side ────────────────────────────────────────────────
    st.subheader("Risk Heatmap: Country × Room Type per Branch")
    branches_in_filter = fdf["Branch"].unique().tolist()
    hmap_cols = st.columns(len(branches_in_filter))

    for col, branch in zip(hmap_cols, sorted(branches_in_filter)):
        sub = risk[risk["Branch"] == branch]
        if sub.empty:
            col.info(f"No data for {branch}")
            continue
        pivot = sub.pivot(index="country", columns="RoomType", values="no_show_rate")
        fig = px.imshow(
            pivot, text_auto=".1%",
            color_continuous_scale="RdYlGn_r", zmin=0.05, zmax=0.75,
            title=f"{branch} Branch",
            aspect="auto",
        )
        fig.update_layout(height=320, coloraxis_showscale=False,
                          margin=dict(t=50, b=10, l=10, r=10))
        col.plotly_chart(fig, use_container_width=True)

    # ── Drill-down: risk ranked list ─────────────────────────────────────────
    st.subheader("Risk-Ranked Segment Table")

    drill_branch = st.selectbox(
        "Drill into Branch:", ["All"] + sorted(df["Branch"].unique().tolist()), key="risk_branch"
    )
    drill_country = st.selectbox(
        "Drill into Country:", ["All"] + sorted(df["country"].unique().tolist()), key="risk_country"
    )

    risk_drill = risk.copy()
    if drill_branch != "All":
        risk_drill = risk_drill[risk_drill["Branch"] == drill_branch]
    if drill_country != "All":
        risk_drill = risk_drill[risk_drill["country"] == drill_country]

    risk_drill = risk_drill.sort_values("no_show_rate", ascending=False)

    fig_rank = px.bar(
        risk_drill,
        x="no_show_pct",
        y=risk_drill["Branch"] + " / " + risk_drill["country"] + " / " + risk_drill["RoomType"],
        orientation="h",
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        text="no_show_pct",
        labels={"x": "No-Show %", "y": "Segment"},
        title="No-Show Rate by Segment (high → low)",
    )
    fig_rank.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_rank.update_layout(
        height=max(350, len(risk_drill) * 28),
        yaxis={"categoryorder": "total ascending"},
        legend_title="Risk Tier",
        xaxis_range=[0, min(100, risk_drill["no_show_pct"].max() * 1.15)],
    )
    st.plotly_chart(fig_rank, use_container_width=True)

    # Raw table
    with st.expander("Full Risk Table"):
        display = risk_drill[["Branch", "country", "RoomType", "n", "no_show_pct", "risk_tier"]].copy()
        display.columns = ["Branch", "Country", "Room Type", "N", "No-Show %", "Risk Tier"]
        st.dataframe(
            display.style.background_gradient(subset=["No-Show %"], cmap="RdYlGn_r"),
            use_container_width=True,
            hide_index=True,
        )
