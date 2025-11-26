import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------
st.set_page_config(
    page_title="AAPL Market Mood Lab",
    layout="wide"
)

st.title("🍏 AAPL Market Mood Lab")
st.markdown(
    """
This app explores **Apple's market moods** by clustering
14-day (or 7-day) windows of news sentiment and stock features.
Use the sidebar to change the window size, number of clusters,
and whether event flags are included.
"""
)

# ---------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------
NEWS_DAILY_PATH = "apple_daily_sentiment.csv"

@st.cache_data
def load_news():
    daily = pd.read_csv(NEWS_DAILY_PATH)
    daily["date"] = pd.to_datetime(daily["date"])
    # keep column names consistent with the rest of your code:
    # date, n_articles, sentiment_mean, sentiment_std,
    # neg_mean, neu_mean, pos_mean
    return daily

@st.cache_data
def load_events(path="apple_events.csv"):
    events = pd.read_csv(path, parse_dates=["event_date"])
    return events


@st.cache_data
def load_prices(start, end):
    # Download AAPL prices
    data = yf.download("AAPL", start=start, end=end, progress=False)

    # If columns are MultiIndex (e.g. ('Adj Close','AAPL')), flatten them
    import pandas as pd
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Move index to a normal column called 'date'
    data = data.reset_index().rename(columns={"Date": "date"})

    # Decide which price column to use
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

    # Create return and volume-change features
    data["ret1_d"] = data[price_col].pct_change()
    data["ret3_d"] = data[price_col].pct_change(3)
    data["vol_chg"] = data["Volume"].pct_change()

    return data


# ---------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------
def merge_price_news(prices, daily_sent):
    df = pd.merge(prices, daily_sent, on="date", how="left")

    fill_cols = [
        "n_articles", "sentiment_mean", "sentiment_std",
        "neg_mean", "neu_mean", "pos_mean"
    ]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


def add_event_flags(df, events, window_event=7):
    df = df.copy()
    df["date_only"] = df["date"].dt.date

    for col in ["event", "wwdc", "iphone", "mac", "spring", "services"]:
        df[f"{col}_shift"] = 0.0

    for _, row in events.iterrows():
        d = row["event_date"].date()
        etype = row["event_type"]

        mask = (
            (df["date_only"] >= d - pd.Timedelta(days=window_event)) &
            (df["date_only"] <= d + pd.Timedelta(days=window_event))
        )

        df.loc[mask, "event_shift"] = 1.0
        df.loc[mask, f"{etype}_shift"] = 1.0

    return df


def build_window_features(df, window=14):
    out = df.copy()

    #  sentiment
    out[f"sent_mean_{window}"] = out["sentiment_mean"].rolling(window).mean()
    out[f"sent_std_{window}"] = out["sentiment_std"].rolling(window).mean()
    out[f"neg_mean_{window}"] = out["neg_mean"].rolling(window).mean()
    out[f"neu_mean_{window}"] = out["neu_mean"].rolling(window).mean()
    out[f"pos_mean_{window}"] = out["pos_mean"].rolling(window).mean()

    # returns and volatility
    out[f"ret_mean_{window}"] = out["ret1_d"].rolling(window).mean()
    out[f"ret_std_{window}"] = out["ret1_d"].rolling(window).std()

    # volume
    out[f"vol_mean_{window}"] = out["Volume"].rolling(window).mean()
    out[f"vol_std_{window}"] = out["Volume"].rolling(window).std()

    # news volume
    out[f"news_mean_{window}"] = out["n_articles"].rolling(window).mean()

    # volume change metrics (optional)
    out[f"vol_chg_mean_{window}"] = out["vol_chg"].rolling(window).mean()
    out[f"vol_chg_std_{window}"] = out["vol_chg"].rolling(window).std()

    out = out.iloc[window:].reset_index(drop=True)
    return out


def get_feature_sets(window, include_events=True):
    base = [
        f"sent_mean_{window}", f"sent_std_{window}",
        f"ret_mean_{window}", f"ret_std_{window}",
        f"vol_mean_{window}", f"vol_std_{window}",
        f"news_mean_{window}",
        f"neg_mean_{window}", f"neu_mean_{window}", f"pos_mean_{window}",
        f"vol_chg_mean_{window}", f"vol_chg_std_{window}",
    ]
    if include_events:
        base += [
            "event_shift", "wwdc_shift", "iphone_shift",
            "mac_shift", "spring_shift", "services_shift",
        ]
    return base


# ---------------------------------------------------------
# Clustering
# ---------------------------------------------------------
def run_kmeans(df_in, feature_cols, k):
    df_clean = df_in.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    return df_clean, X_scaled, kmeans, labels, sil, scaler


def name_clusters(df_final, centers, feature_cols, window):
    # thresholds from final windows
    sent_hi = df_final[f"sent_mean_{window}"].quantile(0.66)
    ret_hi = df_final[f"ret_mean_{window}"].quantile(0.66)
    std_hi = df_final[f"sent_std_{window}"].quantile(0.66)
    vol_hi = df_final[f"vol_std_{window}"].quantile(0.66)

    def _name(row):
        if row[f"sent_mean_{window}"] > sent_hi and row[f"ret_mean_{window}"] > 0:
            return "Hype Bull"
        if (row[f"sent_std_{window}"] > std_hi) or (row[f"vol_std_{window}"] > vol_hi):
            return "Volatile Optimistic"
        return "Calm Neutral Uptrend"

    names = [ _name(r) for _, r in centers.iterrows() ]
    return names


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
st.sidebar.header("Controls")

window = st.sidebar.selectbox("Window size (days)", [7, 14], index=1)
k = st.sidebar.selectbox("Number of clusters (k)", [3, 4], index=0)
include_events = st.sidebar.checkbox("Include event flags as features", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data files expected:**  \n- `apple_news_data.csv`  \n- `apple_events.csv`")

# ---------------------------------------------------------
# Pipeline: load → features → cluster
# ---------------------------------------------------------
with st.spinner("Loading and preparing data..."):
    daily_sent = load_news()
    events = load_events()

    start = daily_sent["date"].min() - pd.Timedelta(days=60)
    end = daily_sent["date"].max() + pd.Timedelta(days=60)
    prices = load_prices(start, end)

    df_all = merge_price_news(prices, daily_sent)
    df_all = add_event_flags(df_all, events, window_event=7)
    df_win = build_window_features(df_all, window=window)

    feature_cols = get_feature_sets(window, include_events=include_events)

    df_final, X_scaled, kmeans, labels, sil, scaler = run_kmeans(df_win, feature_cols, k)

    df_final["cluster"] = labels

    # cluster centers in original scale
    centers_scaled = kmeans.cluster_centers_
    centers = pd.DataFrame(
        scaler.inverse_transform(centers_scaled),
        columns=feature_cols
    )

    cluster_names = name_clusters(df_final, centers, feature_cols, window)
    cluster_map = {i: cluster_names[i] for i in range(len(cluster_names))}
    df_final["mood"] = df_final["cluster"].map(cluster_map)

st.success(f"Clustering done. Silhouette score: **{sil:.3f}**")

# ---------------------------------------------------------
# Layout: charts and tables
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Timeline", "Cluster Centers", "Events & Moods"])

# ---------- Timeline ----------
with tab1:
    st.subheader("Market Moods Timeline")

    mood_colors = {
        "Hype Bull": "green",
        "Volatile Optimistic": "orange",
        "Calm Neutral Uptrend": "blue",
    }

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.scatter(
        df_final["date"],
        df_final[f"sent_mean_{window}"],
        c=df_final["mood"].map(mood_colors),
        s=20,
        alpha=0.8,
    )

    for _, row in events.iterrows():
        ax.axvline(row["event_date"], color="red", linestyle="--", alpha=0.3)
        ax.text(
            row["event_date"],
            ax.get_ylim()[1],
            row["event_type"],
            rotation=90,
            color="red",
            fontsize=8,
            ha="center",
            va="bottom",
        )

    ax.set_title(f"Apple Market Moods Timeline ({window}-day Window)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Sentiment Mean ({window}-day)")
    ax.grid(alpha=0.2)

    st.pyplot(fig)

# ---------- Cluster centers ----------
with tab2:
    st.subheader("Cluster Centers (Original Scale)")
    centers_disp = centers.copy()
    centers_disp.insert(0, "mood", cluster_names)
    numeric_cols = centers_disp.select_dtypes(include="number").columns

    st.dataframe(
        centers_disp.style.format("{:.3f}", subset=numeric_cols)
    )
    #st.dataframe(centers_disp.style.format("{:.3f}"))

    st.markdown("### Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        centers_disp.set_index("mood"),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax2,
    )
    st.pyplot(fig2)

# ---------- Events & moods ----------
with tab3:
    #
    st.subheader(f"Mood Distribution Around Events (±7 days, {window}-day windows)")
    window_event = 7
    records = []

    for _, row in events.iterrows():
        d = row["event_date"]
        etype = row["event_type"]

        mask = (
            (df_final["date"] >= d - pd.Timedelta(days=window_event)) &
            (df_final["date"] <= d + pd.Timedelta(days=window_event))
        )
        subset = df_final.loc[mask]

        for mood, count in subset["mood"].value_counts().items():
            records.append({"event_type": etype, "mood": mood, "count": count})

    if len(records) == 0:
        st.info("No windows fall inside ±7 days of events with current settings.")
    else:
        mood_event_df = pd.DataFrame(records)

        # --- build probabilities dataframe ---
        grp = (
            mood_event_df
            .groupby(["event_type", "mood"], as_index=False)["count"]
            .sum()
        )
        grp["prob"] = grp.groupby("event_type")["count"].transform(
            lambda x: x / x.sum()
        )
        mood_prob = grp  # columns: event_type, mood, count, prob

        col1, col2 = st.columns(2)

        # ---- LEFT: Counts ----
        with col1:
            st.markdown("**Counts (days in each mood around events)**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=mood_event_df,
                x="event_type",
                y="count",
                hue="mood",
                ax=ax3,
            )
            ax3.set_xlabel("Event Type")
            ax3.set_ylabel("Count")
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)

        # ---- RIGHT: Probabilities ----
        with col2:
            st.markdown("**Probabilities**")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=mood_prob,     # ⬅️ IMPORTANT: use mood_prob here
                x="event_type",
                y="prob",           # ⬅️ this column exists in mood_prob
                hue="mood",
                ax=ax4,
            )
            ax4.set_xlabel("Event Type")
            ax4.set_ylabel("Probability")
            ax4.tick_params(axis="x", rotation=45)
            st.pyplot(fig4)

st.markdown("---")
st.markdown(
    f"""
**Current configuration**

- Window size: **{window} days**
- k: **{k} clusters**
- Event features included: **{include_events}**
- Silhouette score: **{sil:.3f}**
"""
)
