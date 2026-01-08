import streamlit as st
import pandas as pd
import pyodbc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YouTube Revenue Insights",
    layout="wide"
)

# -----------------------------
# GLOBAL STYLE (Glossy Look)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #F5F5F5;
}
h1, h2, h3 {
    color: #FF0000;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 YouTube Ad Revenue – Insights Dashboard")
st.markdown(
    "<p style='color:#181818;font-size:18px;'>"
    "Interactive analysis to understand key drivers of YouTube ad revenue."
    "</p>",
    unsafe_allow_html=True
)

# -----------------------------
# SQL SERVER CONNECTION (SSMS)
# -----------------------------
@st.cache_data
def load_data():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=VIGNESH\\SQLEXPRESS;"
        "DATABASE=YT;"
        "Trusted_Connection=yes;"
    )
    df = pd.read_sql("SELECT * FROM YouTubeData", conn)
    conn.close()
    return df

df = load_data()

# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.header("🔎 Filter Data")

year = st.sidebar.multiselect(
    "Year", sorted(df["year"].unique()), default=sorted(df["year"].unique())
)

Quarter = st.sidebar.multiselect(
    "Quarter", sorted(df["Quarter"].unique()), default=sorted(df["Quarter"].unique())
)

category = st.sidebar.multiselect(
    "Category", df["category"].unique(), df["category"].unique()
)

country = st.sidebar.multiselect(
    "Country", df["country"].unique(), df["country"].unique()
)

device = st.sidebar.multiselect(
    "Device", df["device"].unique(), df["device"].unique()
)

top_n = st.sidebar.slider("Top N YouTubers", 5, 20, 10)

views_range = st.sidebar.slider(
    "Views Range",
    int(df["views"].min()),
    int(df["views"].max()),
    (int(df["views"].min()), int(df["views"].max()))
)

filtered_df = df[
    (df["category"].isin(category)) &
    (df["country"].isin(country)) &
    (df["device"].isin(device)) &
    (df["views"].between(views_range[0], views_range[1]))
]
# ======================================================
# 1️⃣ KPI SECTION (AGGREGATED)
# ======================================================
col1, col2, col3 = st.columns(3)

col1.metric("💰 Total Ad Revenue", f"${filtered_df['ad_revenue_usd'].sum():,.0f}")
col2.metric("🎥 Total Videos", filtered_df["video_id"].nunique())
col3.metric("⏱️ Total Watch Time", f"{filtered_df['watch_time_minutes'].sum():,.0f}")



# -----------------------------------
# VISUAL 1 : Top 10 Categories Performance
# Clustered Column + Line Chart
# -----------------------------------
import plotly.graph_objects as go

st.subheader("📊 Top 10 Performing Categories (Revenue & Watch Time)")
YT_COLORS = ["#FF0000", "#181818", "#9E9E9E", "#BDBDBD", "#E53935", "#616161"]
# -----------------------------
# Aggregate + Order data
# -----------------------------
top_cat = (
    filtered_df
    .groupby("category", as_index=False)
    .agg(
        total_revenue=("ad_revenue_usd", "sum"),
        total_watch_time=("watch_time_minutes", "sum")
    )
    .sort_values("total_revenue", ascending=False)
    .head(10)
)

# -----------------------------
# Create Plotly Figure
# -----------------------------
fig = go.Figure()

# Bar: Total Ad Revenue
fig.add_trace(go.Bar(
    x=top_cat["category"],
    y=top_cat["total_revenue"],
    name="Total Ad Revenue",
    marker_color=YT_COLORS,
    hovertemplate=
        "<b>Category:</b> %{x}<br>" +
        "<b>Total Revenue:</b> $%{y:,.0f}<extra></extra>"
))

# Line: Average Watch Time
fig.add_trace(go.Scatter(
    x=top_cat["category"],
    y=top_cat["total_watch_time"],
    name="Total Watch Time",
    mode="lines+markers",
    yaxis="y2",
    marker=dict(color="#181818", size=7),
    hovertemplate=
        "<b>Category:</b> %{x}<br>" +
        "<b>Avg Watch Time:</b> %{y:.2f} mins<extra></extra>"
))

YT_COLORS = ["#FF0000", "#181818", "#9E9E9E", "#BDBDBD", "#E53935", "#616161"]

# -----------------------------
# Layout (Dual Axis)
# -----------------------------
fig.update_layout(
    height=450,
    xaxis=dict(
        title="Category",
        tickangle=-30
    ),
    yaxis=dict(
        title="Total Ad Revenue (USD)"
    ),
    yaxis2=dict(
        title="Total Watch Time (Minutes)",
        overlaying="y",
        side="right"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(t=60, b=60),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# VISUAL 2: Revenue by Category
# -----------------------------
st.subheader("📊 Revenue by Quarter")

rev_quarter = (
    filtered_df
    .groupby("Quarter", as_index=False)["ad_revenue_usd"]
    .sum()
)

rev_quarter["Quarter"] = rev_quarter["Quarter"].astype(str)

fig_q = px.bar(
    rev_quarter,
    x="Quarter",
    y="ad_revenue_usd",
    color="Quarter",                     # 🔥 REQUIRED
    color_discrete_sequence=YT_COLORS,
    height=350,
    labels={"ad_revenue_usd": "Total Revenue (USD)"}
)

fig_q.update_traces(
    hovertemplate=
        "<b>Quarter:</b> Q%{x}<br>" +
        "<b>Total Revenue:</b> $%{y:,.0f}<extra></extra>"
)

fig_q.update_layout(
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig_q, use_container_width=True)



# -----------------------------
# VISUAL 3: Revenue by Country
# -----------------------------
st.subheader("🌍 Average Revenue by Country")

rev_country = (
    filtered_df
    .groupby("country", as_index=False)["ad_revenue_usd"].mean().sort_values("ad_revenue_usd", ascending=False)
)

fig_country = px.bar(
    rev_country,
    x="country",
    y="ad_revenue_usd",
    color="country",
    color_discrete_sequence=YT_COLORS,
    height=450,
    hover_data={
        "country": True,
        "ad_revenue_usd": ":,.2f"
    },
    labels={"ad_revenue_usd": "Avg Revenue (USD)"}
)

st.plotly_chart(fig_country, use_container_width=True)


# -----------------------------
# VISUAL 4: Device-wise Revenue Share
# -----------------------------
st.subheader("📱 Revenue Distribution by Device")

device_rev = (
    filtered_df
    .groupby("device", as_index=False)["ad_revenue_usd"]
    .sum()
)

fig_device = px.pie(
    device_rev,
    names="device",
    values="ad_revenue_usd",
    hole=0.45,
    color_discrete_sequence=YT_COLORS,
    height=450,
    hover_data={
        "device": True,
        "ad_revenue_usd": ":,.2f"
    }
)

fig_device.update_traces(textinfo="percent")
st.plotly_chart(fig_device, use_container_width=True)


# ======================================================
# 5️⃣ TOP YOUTUBERS (PROJECT / VIDEO WISE)
# ======================================================

st.subheader("🏆 Top YouTubers by Ad Revenue")

top_youtubers = (
    filtered_df
    .groupby("video_id", as_index=False)
    .agg(
        total_revenue=("ad_revenue_usd", "sum"),
        video_count=("video_id", "count")
    )
    .sort_values("total_revenue", ascending=False)
    .head(top_n)
)

fig_top = px.bar(
    top_youtubers,
    x="video_id",
    y="total_revenue",
    color="video_id",                      # ✅ discrete coloring
    color_discrete_sequence=YT_COLORS,     # ✅ YouTube colors
    text="total_revenue",
    height=350,
    labels={"total_revenue": "Ad Revenue (USD)"}
)

fig_top.update_traces(
    texttemplate="$%{text:,.0f}",
    textposition="outside",
    hovertemplate=
        "<b>Video ID:</b> %{x}<br>" +
        "<b>Total Revenue:</b> $%{y:,.0f}<extra></extra>"
)

fig_top.update_layout(
    showlegend=False,
    template="plotly_white"
)

st.plotly_chart(fig_top, use_container_width=True)

