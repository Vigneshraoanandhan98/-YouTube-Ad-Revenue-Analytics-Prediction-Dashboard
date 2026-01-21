import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib


@st.cache_data
def load_data():
    return pd.read_csv("Cleaned_Yt.csv")

df = load_data()

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Revenue Insights", "🔮 Revenue Prediction"]
)

if page == "🏠 Home":
    st.title("📺 YouTube Ad Revenue Analytics & Prediction")
    
    st.markdown("""
    ### 🚀 What this app does
    - 📊 Analyze YouTube ad revenue trends
    - 🔍 Identify key revenue drivers
    - 🔮 Predict ad revenue using Machine Learning
    
    ### 🛠️ Tech Stack
    - Python, Pandas
    - Plotly, Streamlit
    - Gradient Boosting Regression
    
    👉 Use the sidebar to navigate.
    """)
elif page == "📊 Revenue Insights":
    st.title("📊 YouTube Ad Revenue – Insights Dashboard")

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
    # VISUAL 2: Revenue by Quarter (Horizontal)
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
        x="ad_revenue_usd",          # 👉 Revenue on X-axis
        y="Quarter",                 # 👉 Quarter on Y-axis
        orientation="h",             # 🔥 TRUE horizontal bar
        color="Quarter",             # keep existing colors
        color_discrete_sequence=YT_COLORS,
        height=350,
        labels={"ad_revenue_usd": "Total Revenue (USD)"}
    )

    fig_q.update_traces(
        hovertemplate=
            "<b>Quarter:</b> Q%{y}<br>" +
            "<b>Total Revenue:</b> $%{x:,.0f}<extra></extra>"
    )

    fig_q.update_layout(
        showlegend=False,
        template="plotly_white",
        xaxis_title="Total Revenue (USD)",
        yaxis_title="Quarter"
    )

    st.plotly_chart(fig_q, use_container_width=True)





    # -----------------------------
    # VISUAL 3: Revenue by Country
    # -----------------------------
    import plotly.graph_objects as go

    st.subheader("🌍 Average Revenue by Country")

    rev_country = (
        filtered_df
        .groupby("country", as_index=False)["ad_revenue_usd"]
        .mean()
        .sort_values("ad_revenue_usd", ascending=False)
    )

    fig_country = go.Figure()

    fig_country.add_trace(
        go.Scatter(
            x=rev_country["country"],
            y=rev_country["ad_revenue_usd"],
            mode="lines+markers",
            marker=dict(
                size=8,
                color=YT_COLORS[:len(rev_country)]  # ✅ existing colors
            ),
            line=dict(width=3),
            hovertemplate=
                "<b>Country:</b> %{x}<br>" +
                "<b>Avg Revenue:</b> $%{y:,.2f}<extra></extra>"
        )
    )

    fig_country.update_layout(
        height=450,                     # ✅ same size
        template="plotly_white",
        xaxis_title="Country",
        yaxis_title="Avg Revenue (USD)",
        showlegend=False
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
    
    # ======================================================
    # Page 2 : 🔮 YouTube Ad Revenue Predictor
    # =====================================================
    
    
    
elif page == "🔮 Revenue Prediction":
    st.title("🔮 YouTube Ad Revenue Predictor")
    st.write("Predict estimated ad revenue based on video performance.")

    # Load model
    model = joblib.load("Gradient Boosting Regressor.pkl")
    model_features = joblib.load("model_features.pkl")

    # Inputs
    views = st.number_input("Views", min_value=10, value=1000)
    likes = st.number_input("Likes", min_value=10, value=100)
    comments = st.number_input("Comments", min_value=10, value=10)
    watch_time = st.number_input("Watch Time (minutes)", min_value=10.0, value=50.0)
    video_length = st.number_input("Video Length (minutes)", min_value=5.0, value=5.0)
    subscribers = st.number_input("Subscribers", min_value=10, value=1000)

    category = st.selectbox("Category", ["Entertainment", "Gaming", "Lifestyle", "Music", "Tech"])
    device = st.selectbox("Device", ["Mobile", "TV", "Tablet"])
    country = st.selectbox("Country", ["US", "IN", "UK", "CA", "DE"])

    # Feature engineering
    engagement_rate = (likes + comments) / views if views > 0 else 0
    st.metric("Engagement Rate", f"{engagement_rate*100:.2f}%")

    # Build input dataframe
    input_data = pd.DataFrame({
        "views": [views],
        "likes": [likes],
        "comments": [comments],
        "watch_time_minutes": [watch_time],
        "video_length_minutes": [video_length],
        "subscribers": [subscribers],
        "engagement_rate": [engagement_rate]
    })

    # One-hot encoding
    for col in model_features:
        if col.startswith("category_"):
            input_data[col] = 1 if col == f"category_{category}" else 0
        elif col.startswith("device_"):
            input_data[col] = 1 if col == f"device_{device}" else 0
        elif col.startswith("country_"):
            input_data[col] = 1 if col == f"country_{country}" else 0

    input_data = input_data.reindex(columns=model_features, fill_value=0)

    if st.button("🔮 Predict Ad Revenue"):
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Ad Revenue: ${prediction[0]:.2f}")
        st.balloons()



