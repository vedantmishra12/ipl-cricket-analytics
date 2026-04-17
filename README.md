# 🏏 IPL Cricket Analytics Dashboard + ML Predictor

Live dashboard and match predictor built on 15 seasons of IPL data (2008–2025).

## 🔗 Live App
https://ipl-cricket-analytics-gbr8kbsbshffhwctyjvuye.streamlit.app/

## 📊 Dashboard Features
- **Overview** — KPI cards, top batters, top bowlers, dismissal breakdown
- **Batting Stats** — Run scorers, strike rates, filterable by season
- **Bowling Stats** — Wicket takers, economy rates
- **Venue Analysis** — Highest scoring grounds, avg run rate
- **Team Performance** — Runs and run rate by season, multi-team comparison

## 🤖 ML Match Predictor
- **Model:** XGBoost
- **Accuracy:** 58% | Macro F1: 0.55
- **Features:** Rolling win rate (last 10 matches), venue-specific win rate, 
  head-to-head record, win rate differentials, team/venue/season encodings
- **Improvement over baseline:** +5% accuracy, +0.17 Macro F1 vs original Random Forest

## 📦 Data
- Source: [Cricsheet](https://cricsheet.org)
- 2380 match files, 15 seasons, 19 teams, 496 batters, 396 bowlers, 41 venues

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit, Plotly
