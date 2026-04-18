# app.py — IPL Cricket Analytics Dashboard + ML Predictor
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import json
import numpy as np

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="IPL Cricket Analytics",
    page_icon="🏏",
    layout="wide"
)

# ── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data():
    raw         = pd.read_csv('streamlit_data/ipl_raw.csv')
    batters     = pd.read_csv('streamlit_data/top_batters.csv')
    bowlers     = pd.read_csv('streamlit_data/top_bowlers.csv')
    team_runs   = pd.read_csv('streamlit_data/team_runs.csv')
    venues      = pd.read_csv('streamlit_data/venue_stats.csv')
    dismissals  = pd.read_csv('streamlit_data/dismissal_stats.csv')
    ml_df       = pd.read_csv('streamlit_data/ml_df.csv')
    return raw, batters, bowlers, team_runs, venues, dismissals, ml_df

@st.cache_resource
def load_ml():
    with open('streamlit_data/ipl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('streamlit_data/le_team.pkl', 'rb') as f:
        le_team = pickle.load(f)
    with open('streamlit_data/le_venue.pkl', 'rb') as f:
        le_venue = pickle.load(f)
    with open('streamlit_data/le_season.pkl', 'rb') as f:
        le_season = pickle.load(f)
    with open('streamlit_data/teams.json', 'r') as f:
        teams_list = json.load(f)
    with open('streamlit_data/venues.json', 'r') as f:
        venues_list = json.load(f)
    with open('streamlit_data/seasons.json', 'r') as f:
        seasons_list = json.load(f)
    with open('streamlit_data/team_winrate.json', 'r') as f:
        team_winrate = json.load(f)
    return model, le_team, le_venue, le_season, teams_list, venues_list, seasons_list, team_winrate

df, top_batters, top_bowlers, team_runs, venue_stats, dismissal_stats, ml_df = load_data()
model, le_team, le_venue, le_season, teams_list, venues_list, seasons_list, team_winrate = load_ml()

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/8/84/IPL_2022_Logo.svg", width=200)
st.sidebar.title("🏏 IPL Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🏏 Batting Stats",
    "🎳 Bowling Stats",
    "🏟️ Venue Analysis",
    "📈 Team Performance",
    "🤖 Match Predictor"
])

seasons = sorted(df['season'].unique(), key=str)
selected_season = st.sidebar.selectbox("Filter by Season", ["All Seasons"] + list(seasons))

# ── Filter by season ──────────────────────────────────────
if selected_season != "All Seasons":
    df_filtered = df[df['season'] == selected_season]
else:
    df_filtered = df

# ── Helper ────────────────────────────────────────────────
def dark_layout(fig):
    fig.update_layout(
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font_color='white'
    )
    return fig

# ══════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🏏 IPL Cricket Analytics Dashboard")
    st.markdown("### Real data from 2008–2025 IPL seasons")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏟️ Total Matches",    df['match_id'].nunique())
    col2.metric("👥 Total Teams",       df['batting_team'].nunique())
    col3.metric("🧑 Total Players",     df['striker'].nunique())
    col4.metric("🎳 Total Bowlers",     df['bowler'].nunique())

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("🏏 Top 10 Run Scorers")
        fig = px.bar(
            top_batters.head(10),
            x='striker', y='total_runs',
            color='total_runs',
            color_continuous_scale='Oranges',
            labels={'striker': 'Player', 'total_runs': 'Runs'}
        )
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    with col6:
        st.subheader("🎳 Top 10 Wicket Takers")
        fig = px.bar(
            top_bowlers.head(10),
            x='bowler', y='total_wickets',
            color='total_wickets',
            color_continuous_scale='Blues',
            labels={'bowler': 'Bowler', 'total_wickets': 'Wickets'}
        )
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Dismissal Types")
    fig = px.pie(
        dismissal_stats,
        names='wicket_type', values='count',
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(dark_layout(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 2 — BATTING STATS
# ══════════════════════════════════════════════════════════
elif page == "🏏 Batting Stats":
    st.title("🏏 Batting Statistics")
    st.markdown("---")

    batting = df_filtered.groupby('striker').agg(
        total_runs   = ('runs_off_bat', 'sum'),
        total_balls  = ('ball', 'count'),
        dismissals   = ('player_dismissed', 'count')
    ).reset_index()
    batting['strike_rate'] = (batting['total_runs'] / batting['total_balls'] * 100).round(2)
    batting = batting.sort_values('total_runs', ascending=False).reset_index(drop=True)

    top_n = st.slider("Show Top N Players", 5, 50, 10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {top_n} Run Scorers")
        fig = px.bar(
            batting.head(top_n),
            x='striker', y='total_runs',
            color='total_runs',
            color_continuous_scale='Oranges',
            labels={'striker': 'Player', 'total_runs': 'Runs'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    with col2:
        st.subheader(f"Top {top_n} Strike Rates (min 100 balls)")
        sr = batting[batting['total_balls'] >= 100].sort_values('strike_rate', ascending=False)
        fig = px.bar(
            sr.head(top_n),
            x='striker', y='strike_rate',
            color='strike_rate',
            color_continuous_scale='Greens',
            labels={'striker': 'Player', 'strike_rate': 'Strike Rate'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Batting Table")
    st.dataframe(batting.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 3 — BOWLING STATS
# ══════════════════════════════════════════════════════════
elif page == "🎳 Bowling Stats":
    st.title("🎳 Bowling Statistics")
    st.markdown("---")

    bowling = df_filtered.groupby('bowler').agg(
        total_wickets    = ('player_dismissed', 'count'),
        total_balls      = ('ball', 'count'),
        total_runs_given = ('runs_off_bat', 'sum')
    ).reset_index()
    bowling['economy'] = (bowling['total_runs_given'] / bowling['total_balls'] * 6).round(2)
    bowling = bowling.sort_values('total_wickets', ascending=False).reset_index(drop=True)

    top_n = st.slider("Show Top N Bowlers", 5, 50, 10)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Top {top_n} Wicket Takers")
        fig = px.bar(
            bowling.head(top_n),
            x='bowler', y='total_wickets',
            color='total_wickets',
            color_continuous_scale='Blues',
            labels={'bowler': 'Bowler', 'total_wickets': 'Wickets'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    with col2:
        st.subheader(f"Best Economy (min 120 balls)")
        eco = bowling[bowling['total_balls'] >= 120].sort_values('economy')
        fig = px.bar(
            eco.head(top_n),
            x='bowler', y='economy',
            color='economy',
            color_continuous_scale='Reds',
            labels={'bowler': 'Bowler', 'economy': 'Economy Rate'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Full Bowling Table")
    st.dataframe(bowling.head(50), use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 4 — VENUE ANALYSIS
# ══════════════════════════════════════════════════════════
elif page == "🏟️ Venue Analysis":
    st.title("🏟️ Venue Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Highest Scoring Venues")
        fig = px.bar(
            venue_stats.head(10),
            x='venue', y='total_runs',
            color='total_runs',
            color_continuous_scale='Purples',
            labels={'venue': 'Venue', 'total_runs': 'Total Runs'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    with col2:
        st.subheader("Average Run Rate by Venue")
        fig = px.bar(
            venue_stats.head(10),
            x='venue', y='avg_run_rate',
            color='avg_run_rate',
            color_continuous_scale='Teal',
            labels={'venue': 'Venue', 'avg_run_rate': 'Avg Run Rate'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(dark_layout(fig), use_container_width=True)

    st.markdown("---")
    st.subheader("📋 All Venues Table")
    st.dataframe(venue_stats, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 5 — TEAM PERFORMANCE
# ══════════════════════════════════════════════════════════
elif page == "📈 Team Performance":
    st.title("📈 Team Performance Over Seasons")
    st.markdown("---")

    active_teams = [
        'Mumbai Indians', 'Chennai Super Kings', 'Royal Challengers Bengaluru',
        'Kolkata Knight Riders', 'Rajasthan Royals', 'Sunrisers Hyderabad',
        'Delhi Capitals', 'Punjab Kings', 'Gujarat Titans', 'Lucknow Super Giants'
    ]

    selected_teams = st.multiselect(
        "Select Teams to Compare",
        options=sorted(df['batting_team'].unique()),
        default=active_teams[:5]
    )

    team_filtered = team_runs[team_runs['batting_team'].isin(selected_teams)]

    st.subheader("Total Runs by Season")
    fig = px.line(
        team_filtered,
        x='season', y='total_runs',
        color='batting_team',
        markers=True,
        labels={'season': 'Season', 'total_runs': 'Total Runs', 'batting_team': 'Team'}
    )
    st.plotly_chart(dark_layout(fig), use_container_width=True)

    st.subheader("Run Rate by Season")
    fig = px.line(
        team_filtered,
        x='season', y='run_rate',
        color='batting_team',
        markers=True,
        labels={'season': 'Season', 'run_rate': 'Run Rate', 'batting_team': 'Team'}
    )
    st.plotly_chart(dark_layout(fig), use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 6 — ML MATCH PREDICTOR
# ══════════════════════════════════════════════════════════
elif page == "🤖 Match Predictor":
    st.title("🤖 IPL Match Winner Predictor")
    st.markdown("### Powered by XGBoost ML Model (58% accuracy | Macro F1: 0.55)")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏏 Select Match Details")

        team1 = st.selectbox("Team 1", sorted(teams_list), index=0)
        team2 = st.selectbox("Team 2", sorted(teams_list), index=1)
        venue = st.selectbox("Venue", sorted(venues_list))
        season = st.selectbox("Season", sorted(seasons_list, reverse=True))

    with col2:
        st.subheader("📊 Prediction")

        if team1 == team2:
            st.warning("⚠️ Please select two different teams!")
        else:
            if st.button("🔮 Predict Winner!", use_container_width=True):
                try:
                    # Encode inputs
                    t1_enc = le_team.transform([team1])[0]
                    t2_enc = le_team.transform([team2])[0]
                    v_enc  = le_venue.transform([venue])[0]
                    s_enc  = le_season.transform([str(season)])[0]

                    # H2H stats
                    h2h = ml_df[
                        ((ml_df['team1'] == team1) & (ml_df['team2'] == team2)) |
                        ((ml_df['team1'] == team2) & (ml_df['team2'] == team1))
                    ]
                    h2h_t1 = (h2h['winner'] == team1).sum()
                    h2h_t2 = (h2h['winner'] == team2).sum()

                

                    # Win rates
                    wr1 = team_winrate.get(team1, 0.5)
                    wr2 = team_winrate.get(team2, 0.5)

                    # Venue win rates from ml_df
                    venue_matches_t1 = ml_df[(ml_df['team1'] == team1) & (ml_df['venue'] == venue)]
                    venue_wr_t1 = (venue_matches_t1['winner'] == team1).mean() if len(venue_matches_t1) > 0 else 0.5

                    venue_matches_t2 = ml_df[(ml_df['team1'] == team2) & (ml_df['venue'] == venue)]
                    venue_wr_t2 = (venue_matches_t2['winner'] == team2).mean() if len(venue_matches_t2) > 0 else 0.5

                    # Diff features
                    rolling_wr_diff = wr1 - wr2
                    venue_wr_diff   = venue_wr_t1 - venue_wr_t2
                    h2h_diff        = h2h_t1 - h2h_t2

                    # Build feature array — must match training order exactly
                    input_features = np.array([[
                             t1_enc, t2_enc, v_enc, s_enc,
                             wr1, wr2, rolling_wr_diff,
                             venue_wr_t1, venue_wr_t2, venue_wr_diff,
                             h2h_t1, h2h_t2, h2h_diff
                       ]])
 
                    pred   = model.predict(input_features)[0]
                    proba  = model.predict_proba(input_features)[0]

                    winner     = team1 if pred == 1 else team2
                    confidence = max(proba) * 100

                    
                    

                    # Show result
                    st.success(f"🏆 Predicted Winner: **{winner}**")
                    st.metric("Confidence", f"{confidence:.1f}%")

                    # Probability bar chart
                    proba_df = pd.DataFrame({
                        'Team': [team1, team2],
                        'Probability': [proba[1], proba[0]]
                    })

                    fig = px.bar(
                        proba_df,
                        x='Team', y='Probability',
                        color='Probability',
                        color_continuous_scale='Greens',
                        title='Win Probability by Team'
                    )
                    st.plotly_chart(dark_layout(fig), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.subheader("📊 Head to Head Records")

    if team1 != team2:
        h2h = ml_df[
            ((ml_df['team1'] == team1) & (ml_df['team2'] == team2)) |
            ((ml_df['team1'] == team2) & (ml_df['team2'] == team1))
        ]
        t1_wins = (h2h['winner'] == team1).sum()
        t2_wins = (h2h['winner'] == team2).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric(f"{team1} Wins", t1_wins)
        c2.metric("Total Matches", len(h2h))
        c3.metric(f"{team2} Wins", t2_wins)

# ── Footer ─────────────────────────────────────────────────
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & Real IPL Data from Cricsheet")