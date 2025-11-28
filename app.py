import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="AAPL Market Mood Lab",
    layout="wide"
)

st.title("AAPL Market Mood Lab")

st.markdown(
    """
This app explores **Apple's market moods** by clustering 7-day or 14-day windows
of news sentiment and stock features.

You’ll see:

- A **timeline** of moods over time
- **Cluster centers** (both original scale and z-score heatmap)
- **Event–mood interactions** around Apple product / WWDC events
- **Moods per year** to see regime shifts over time
- **Recommendations per cluster** for different stakeholders
"""
)

# -------------------------------------------------------------------
# File paths (must exist in repo root)
# -------------------------------------------------------------------
NEWS_DAILY_PATH = "apple_daily_sentiment.csv"
EVENTS_PATH = "apple_events.csv"

# -------------------------------------------------------------------
# Data loading (cached)
# -------------------------------------------------------------------
@st.cache_data
def load_news() -> pd.DataFrame:
    daily = pd.read_csv(NEWS_DAILY_PATH)
    # expected columns: date, n_articles, sentiment_mean, sentiment_std,
    #                   neg_mean, neu_mean, pos_mean
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


@st.cache_data
def load_events(path: str = EVENTS_PATH) -> pd.DataFrame:
    events = pd.read_csv(path, parse_dates=["event_date"])
    return events


@st.cache_data
def load_prices(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data = yf.download("AAPL", start=start, end=end, progress=False)

    # Flatten MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index().rename(columns={"Date": "date"})

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"

    data["ret1_d"] = data[price_col].pct_change()
    data["ret3_d"] = data[price_col].pct_change(3)
    data["vol_chg"] = data["Volume"].pct_change()

    return data


# -------------------------------------------------------------------
# Feature engineering helpers
# -------------------------------------------------------------------
def merge_price_news(prices: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(prices, daily_sent, on="date", how="left")

    fill_cols = [
        "n_articles",
        "sentiment_mean",
        "sentiment_std",
        "neg_mean",
        "neu_mean",
        "pos_mean",
    ]
    df[fill_cols] = df[fill_cols].fillna(0.0)
    return df


def add_event_flags(df: pd.DataFrame, events: pd.DataFrame, window_event: int = 7) -> pd.DataFrame:
    """Add daily event indicator columns (same logic as in the notebook)."""
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


def build_window_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    out = df.copy()

    # sentiment
    out[f"sent_mean_{window}"] = out["sentiment_mean"].rolling(window).mean()
    out[f"sent_std_{window}"] = out["sentiment_std"].rolling(window).mean()
    out[f"neg_mean_{window}"] = out["neg_mean"].rolling(window).mean()
    out[f"neu_mean_{window}"] = out["neu_mean"].rolling(window).mean()
    out[f"pos_mean_{window}"] = out["pos_mean"].rolling(window).mean()

    # returns & volatility
    out[f"ret_mean_{window}"] = out["ret1_d"].rolling(window).mean()
    out[f"ret_std_{window}"] = out["ret1_d"].rolling(window).std()

    # volume
    out[f"vol_mean_{window}"] = out["Volume"].rolling(window).mean()
    out[f"vol_std_{window}"] = out["Volume"].rolling(window).std()

    # news volume
    out[f"news_mean_{window}"] = out["n_articles"].rolling(window).mean()

    # volume change (not used for clustering, but kept if you want later)
    out[f"vol_chg_mean_{window}"] = out["vol_chg"].rolling(window).mean()
    out[f"vol_chg_std_{window}"] = out["vol_chg"].rolling(window).std()

    # drop the first (window-1) rows with NaNs
    out = out.iloc[window:].reset_index(drop=True)
    return out


def get_feature_set_for_window(window: int, include_events: bool) -> list[str]:
    """
    Feature set that matches your grid-search table.

    - When include_events = False -> 'no_events'
    - When include_events = True  -> 'full'
    """
    if window == 14:
        base = [
            "sent_mean_14",
            "sent_std_14",
            "ret_mean_14",
            "ret_std_14",
            "vol_mean_14",
            "vol_std_14",
            "news_mean_14",
        ]
    elif window == 7:
        base = [
            "sent_mean_7",
            "sent_std_7",
            "ret_mean_7",
            "ret_std_7",
            "vol_mean_7",
            "vol_std_7",
            "news_mean_7",
        ]
    else:
        raise ValueError("Window must be 7 or 14")

    if include_events:
        base += [
            "event_shift",
            "wwdc_shift",
            "iphone_shift",
            "mac_shift",
            "spring_shift",
            "services_shift",
        ]
    return base


# -------------------------------------------------------------------
# Clustering
# -------------------------------------------------------------------
def run_kmeans(df_in: pd.DataFrame, feature_cols: list[str], k: int):
    df_clean = df_in.dropna(subset=feature_cols).copy()
    X = df_clean[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Same config as notebook
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    return df_clean, X_scaled, kmeans, labels, sil, scaler


# Fixed cluster name mapping from your final notebook
CLUSTER_NAME_MAP = {
    0: "The Quiet Accumulation",
    1: "The Media Hype Cycle",
    2: "High Volatility/Correction",
}

MOOD_COLORS = {
    "The Quiet Accumulation": "blue",
    "The Media Hype Cycle": "orange",
    "High Volatility/Correction": "green",
}

# -------------------------------------------------------------------
# Recommendations per cluster & audience
# -------------------------------------------------------------------
RECOMMENDATIONS = {
    "The Quiet Accumulation": {
        "Corporate investor / fund manager": [
            "Treat this as the **base-case regime**: sentiment is mildly positive and volatility is low.",
            "Keep AAPL as a **core holding**, with small tilts instead of big timing bets.",
            "Use this period to **harvest steady alpha** from stock-picking and relative value, not from leverage."
        ],
        "Retail investor": [
            "This is a good time for **regular dollar-cost averaging** instead of trying to time entries.",
            "Avoid checking the portfolio every hour – the story here is **slow, steady compounding**.",
            "Use small dips as chances to **top up quality positions**, not to panic sell."
        ],
        "Apple leadership / strategy team": [
            "Focus on **execution and ecosystem stickiness** rather than heavy promotional pushes.",
            "Use the calm regime to **test incremental price changes** or bundle experiments.",
            "Keep guidance realistic; the goal here is to **reinforce trust** rather than chase headlines."
        ],
    },
    "The Media Hype Cycle": {
        "Corporate investor / fund manager": [
            "Recognize this as a **sentiment overshoot**: news and volume spike faster than fundamentals.",
            "Trim positions **into strength** and lock in gains instead of expanding risk just because the mood is euphoric.",
            "Consider **options-based hedges** to protect against a sharp reversal once the hype fades."
        ],
        "Retail investor": [
            "Beware of **FOMO** – each headline makes it tempting to buy at the top.",
            "If you add exposure, **keep position sizes small** and define your exit plan in advance.",
            "Avoid chasing short-term price jumps; keep focusing on your **long-term thesis**."
        ],
        "Apple leadership / strategy team": [
            "This is a prime window for **flagship launches and service upsell** – attention is already high.",
            "Communication should emphasize **delivery and user value**, not just big promises.",
            "Avoid overly aggressive guidance; expectations are already elevated, so **execution risk** is higher."
        ],
    },
    "High Volatility/Correction": {
        "Corporate investor / fund manager": [
            "Treat this as a **risk-management regime**: sentiment is fragile and volatility is elevated.",
            "Reduce gross and net exposure, rotate partially into **defensives or cash**, and add back risk gradually.",
            "Use volatility to **scale in over time** if fundamentals remain intact, instead of trying to catch the exact bottom."
        ],
        "Retail investor": [
            "Do not react purely out of fear – avoid **dumping everything at once**.",
            "Decide in advance which part of the position is **long-term conviction** vs. what you are willing to cut.",
            "If you feel emotional, keep new money in cash until the mood stabilizes and your plan is clear."
        ],
        "Apple leadership / strategy team": [
            "Lean into **long-term messaging**: roadmap, ecosystem, and durable cash flows.",
            "Signal confidence carefully – for example through **buybacks, stable guidance, or insider alignment**.",
            "Avoid noisy, experimental announcements; focus communication on **core products and reliability**."
        ],
    },
}

# -------------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------------
st.sidebar.header("Controls")

# 14-day default to match the main config
window = st.sidebar.selectbox("Window size (days)", [7, 14], index=1)

# Only k=3 to match your table
k = st.sidebar.selectbox("Number of clusters (k)", [3], index=0)

include_events = st.sidebar.checkbox(
    "Include event flags as features",
    value=False,
    help="Checked = 'full' (with events). Unchecked = 'no_events'."
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Data files expected in repo root:**  \n"
    "- `apple_daily_sentiment.csv`  \n"
    "- `apple_events.csv`"
)

# -------------------------------------------------------------------
# Pipeline: load → features → cluster
# -------------------------------------------------------------------
with st.spinner("Loading and preparing data..."):
    daily_sent = load_news()
    events = load_events()

    start = daily_sent["date"].min() - pd.Timedelta(days=60)
    end = daily_sent["date"].max() + pd.Timedelta(days=60)
    prices = load_prices(start, end)

    df_all = merge_price_news(prices, daily_sent)
    df_all = add_event_flags(df_all, events, window_event=7)

    df_win = build_window_features(df_all, window=window)

    feature_cols = get_feature_set_for_window(window, include_events)

    df_final, X_scaled, kmeans, labels, sil, scaler = run_kmeans(
        df_win, feature_cols, k
    )
    df_final["cluster"] = labels
    df_final["mood"] = df_final["cluster"].map(CLUSTER_NAME_MAP)

    centers_orig = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=feature_cols,
    )

    centers_z = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=feature_cols,
    )
    centers_z.insert(0, "mood", [CLUSTER_NAME_MAP[i] for i in range(k)])
    centers_z = centers_z.set_index("mood")

feature_label = "full" if include_events else "no_events"

st.success(
    f"Clustering done. Silhouette score: **{sil:.6f}** "
    f"(window=**{window}**, k=**{k}**, features=**{feature_label}**)"
)

# -------------------------------------------------------------------
# Layout: tabs
# -------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Timeline", "Cluster Centers", "Events & Moods", "Moods per Year", "Recommendations"]
)

# -------------------------------------------------------------------
# Tab 1 – Timeline
# -------------------------------------------------------------------
with tab1:
    st.subheader(f"Market Moods Timeline ({window}-day Window)")

    fig, ax = plt.subplots(figsize=(14, 5))

    colors = df_final["mood"].map(MOOD_COLORS)

    ax.scatter(
        df_final["date"],
        df_final[f"sent_mean_{window}"],
        c=colors,
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

    plt.tight_layout()
    st.pyplot(fig)

# -------------------------------------------------------------------
# Tab 2 – Cluster Centers (original + z-score)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Cluster Centers (Original Scale)")
    centers_disp = centers_orig.copy()
    centers_disp.insert(0, "mood", [CLUSTER_NAME_MAP[i] for i in range(k)])
    numeric_cols = centers_disp.select_dtypes(include="number").columns

    st.dataframe(centers_disp.style.format("{:.3f}", subset=numeric_cols))

    st.markdown("### Cluster Centers Heatmap (Z-Score scaled)")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        centers_z,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax2,
    )
    ax2.set_title(f"Cluster Centers (Z-Score scaled, {window}-day, k={k})")
    st.pyplot(fig2)

# -------------------------------------------------------------------
# Tab 3 – Events & Moods
# -------------------------------------------------------------------
with tab3:
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
            records.append(
                {"event_type": etype, "mood": mood, "count": int(count)}
            )

    if len(records) == 0:
        st.info("No windows fall inside ±7 days of events with current settings.")
    else:
        mood_event_df = pd.DataFrame(records)

        grp = (
            mood_event_df
            .groupby(["event_type", "mood"], as_index=False)["count"]
            .sum()
        )
        grp["prob"] = grp.groupby("event_type")["count"].transform(
            lambda x: x / x.sum()
        )
        mood_prob = grp

        col1, col2 = st.columns(2)

        # LEFT: Counts
        with col1:
            st.markdown("**Counts (days in each mood around events)**")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=mood_event_df,
                x="event_type",
                y="count",
                hue="mood",
                palette=MOOD_COLORS,
                ax=ax3,
            )
            ax3.set_xlabel("Event Type")
            ax3.set_ylabel("Count")
            ax3.tick_params(axis="x", rotation=45)
            st.pyplot(fig3)

        # RIGHT: Probabilities
        with col2:
            st.markdown("**Probabilities (share of windows by mood)**")
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=mood_prob,
                x="event_type",
                y="prob",
                hue="mood",
                palette=MOOD_COLORS,
                ax=ax4,
            )
            ax4.set_xlabel("Event Type")
            ax4.set_ylabel("Probability")
            ax4.tick_params(axis="x", rotation=45)
            st.pyplot(fig4)

# -------------------------------------------------------------------
# Tab 4 – Moods per Year
# -------------------------------------------------------------------
with tab4:
    st.subheader("Moods per Year")

    df_final["year"] = df_final["date"].dt.year
    mood_counts = (
        df_final.groupby(["year", "mood"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    colors = [MOOD_COLORS[m] for m in mood_counts.columns]

    fig5, ax5 = plt.subplots(figsize=(10, 5))
    mood_counts.plot(kind="bar", ax=ax5, color=colors)
    ax5.set_xlabel("Year")
    ax5.set_ylabel("Number of windows")
    ax5.set_title("Moods per Year")
    ax5.legend(title="Mood")
    ax5.tick_params(axis="x", rotation=45)

    st.pyplot(fig5)

# -------------------------------------------------------------------
# Tab 5 – Interactive Recommendations
# -------------------------------------------------------------------
with tab5:
    st.subheader("Cluster-based Recommendations")

    st.markdown(
        "Select a **market mood** and **audience** to see practical recommendations "
        "based on the patterns we found in the clusters."
    )

    mood_list = list(CLUSTER_NAME_MAP.values())
    selected_mood = st.selectbox("Market mood", mood_list, index=0)

    audience_options = [
        "Corporate investor / fund manager",
        "Retail investor",
        "Apple leadership / strategy team",
    ]
    selected_audience = st.radio("Audience", audience_options, index=0)

    st.markdown(f"### {selected_mood} — {selected_audience}")

    for bullet in RECOMMENDATIONS[selected_mood][selected_audience]:
        st.markdown(f"- {bullet}")

# -------------------------------------------------------------------
# Footer – current config
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"""
**Current configuration**

- Window size: **{window} days**
- k: **{k} clusters**
- Features: **{feature_label}**
- Silhouette score: **{sil:.6f}**
"""
)
