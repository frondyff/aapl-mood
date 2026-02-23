🍏 AAPL Market Mood Lab

Clustering Apple’s market sentiment using news + price features

This project explores Apple’s market “moods” by combining:

Daily AAPL-related news sentiment

Market indicators (returns, volume, volatility)

Event windows (WWDC, iPhone launch, etc.)

Rolling window feature engineering

K-Means clustering

The goal is not price prediction.
The goal is to understand market regimes driven by sentiment + behavior.

🔎 Project Objective

Instead of asking:

“Can we predict tomorrow’s price?”

We ask:

“What emotional & behavioral states does the market cycle through?”

We use clustering to uncover recurring market moods such as:

The Quiet Accumulation

The Media Hype Cycle

High Volatility / Correction

🧠 Methodology Overview
1️⃣ Data Sources

AAPL-related news articles (sentiment-enriched)

AAPL historical price data (via yfinance)

Apple event calendar (WWDC, iPhone, Mac, etc.)

2️⃣ Preprocessing

Remove duplicates & invalid entries

Standardize timestamps → daily level

Align news dates with trading days

Aggregate sentiment daily (mean, std, count)

3️⃣ Feature Engineering

We combine emotional signals with market behavior.

Sentiment Features

Daily mean sentiment

Sentiment volatility (std)

Positive / neutral / negative proportions

Article volume

Market Features

1-day & 3-day returns

Volume change

Rolling volatility

Rolling Window Features

7-day and 14-day rolling:

Mean sentiment

Std sentiment

Mean returns

Volatility

News intensity

This prevents data leakage by using only past information for each window.

Event Features

±7-day flags for:

WWDC

iPhone launch

Mac event

Spring/Services events

4️⃣ Clustering

We use:

StandardScaler

KMeans (random_state=42)

Silhouette score for evaluation

We tested multiple configurations:

Window	k	Features	Silhouette
14	3	full	0.3936
14	3	no_events	0.4421
14	4	full	0.4283
14	4	no_events	0.3409
📊 Key Insights

Rolling windows reveal momentum, not daily noise

Hype periods cluster with high sentiment + article volume

Correction regimes show elevated volatility

Event windows amplify media-driven clusters

📈 Visualizations

The project includes:

Timeline of market moods

Cluster center heatmaps (original & z-score scaled)

Mood distribution around Apple events

Moods per year

🚀 Streamlit App

The interactive dashboard allows:

Window selection (7 / 14 days)

Cluster selection (k=3 or k=4)

Toggle event features

View heatmaps (scaled & original)

Event-mood analysis

Interactive recommendations per cluster

To run locally:

pip install -r requirements.txt
streamlit run app.py
📂 Project Structure
aapl-mood/
│
├── app.py
├── Project_AAPL_Market_Mood_Clustering_final_submit.ipynb
├── apple_daily_sentiment.csv
├── apple_events.csv
├── requirements.txt
└── README.md
🔒 Reproducibility

Python 3.12+

random_state=42

Rolling features use strictly past data (no leakage)

🧭 Stakeholder Value
Corporate Investment Manager / Fund

Identify regime shifts

Risk-adjust exposure

Detect hype-driven overextension

Retail Investor

Avoid buying during peak hype

Recognize accumulation phases

Apple (Company Perspective)

Understand media-driven volatility

Align communication with sentiment cycles

⚠️ Disclaimer

This project is for academic and analytical purposes only.
It does not constitute investment advice.
