import fastf1
import pandas as pd


fastf1.Cache.enable_cache('cache')

import streamlit as st


@st.cache_data(show_spinner=True)
def load_session_cached(year, gp, session_type):
    """
    Load and return a FastF1 session and laps. Cached for faster re-use.
    """
    import fastf1
    fastf1.Cache.enable_cache('cache')

    session, laps = load_session(year, gp, session_type)
    return session, laps

def load_session(year, gp, session_type):
    """
    Load a FastF1 session and return (session, laps) tuple.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load()  # Load timing + telemetry data
    laps = session.laps.copy()
    return session, laps


def prepare_laps(session, accurate_only=True):
    """
    Prepare laps with LapTimeSeconds, Team, and Driver info.
    """
    laps = session.laps.copy()

    # Optionally keep only accurate laps
    if accurate_only:
        laps = laps.pick_accurate()

    # Add a numeric LapTime column for easy plotting
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Ensure we have team info
    if 'Team' not in laps.columns or laps['Team'].isna().all():
        # fallback — get from session results
        results = session.results
        driver_team_map = results.set_index('Abbreviation')['TeamName'].to_dict()
        laps.loc[:, 'Team'] = laps['Driver'].map(driver_team_map)

    # Keep useful columns
    columns_to_keep = [
        'Driver', 'Team', 'LapNumber', 'LapTimeSeconds',
        'Compound', 'Position', 'IsAccurate'
    ]
    laps = laps[[c for c in columns_to_keep if c in laps.columns]]

    return laps


import pandas as pd


def get_fastest_lap_leaderboard(laps):
    """
    Return a DataFrame with each driver's fastest lap.
    """
    laps = laps.copy()

    # Ensure LapTimeSeconds exists
    if 'LapTimeSeconds' not in laps.columns:
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Fastest lap per driver
    fastest_laps = laps.loc[laps.groupby('Driver')['LapTimeSeconds'].idxmin()]

    # Sort by lap time
    fastest_laps = fastest_laps.sort_values('LapTimeSeconds')

    # Select columns to display
    leaderboard = fastest_laps[['Driver', 'Team', 'LapNumber', 'LapTime', 'LapTimeSeconds', 'Compound']]

    return leaderboard.reset_index(drop=True)


def get_filtered_laps(laps, drivers=None, accurate_only=True):
    """
    Filter laps for selected drivers and optionally only accurate laps.
    """
    df = laps.copy()
    if accurate_only:
        df = df[df['IsAccurate'] == True]
    if drivers:
        df = df[df['Driver'].isin(drivers)]
    # Convert LapTime to seconds for plotting
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    return df


def get_top_fastest_laps(laps, top_n=10):
    """
    Return the top N fastest laps from the session across all drivers.
    Includes Driver, Team, LapNumber, LapTime, LapTimeSeconds, Compound, and TyreLife.
    """
    df = laps.copy()

    # Make sure LapTimeSeconds exists
    if 'LapTimeSeconds' not in df.columns:
        df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    # Sort all laps by LapTimeSeconds
    df_sorted = df.sort_values('LapTimeSeconds').head(top_n)

    # Select columns to show
    leaderboard = df_sorted[['Driver', 'Team', 'LapNumber', 'LapTime', 'LapTimeSeconds', 'Compound', 'TyreLife']]

    return leaderboard.reset_index(drop=True)


def compute_lap_time_consistency(laps, drivers=None, accurate_only=True):
    """
    Compute lap time consistency per driver.
    Returns a DataFrame with driver and mean of stint lap time std.
    """
    df = laps.copy()
    if accurate_only:
        df = df[df['IsAccurate'] == True]
    if drivers:
        df = df[df['Driver'].isin(drivers)]

    # Convert LapTime to seconds
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    consistency_list = []

    for driver, driver_laps in df.groupby('Driver'):
        stint_std = driver_laps.groupby('Stint')['LapTimeSeconds'].std()
        mean_std = stint_std.mean()  # mean of stint stds
        consistency_list.append({'Driver': driver, 'Consistency': mean_std})

    return pd.DataFrame(consistency_list).sort_values('Consistency')

def get_laps_in_traffic(laps: pd.DataFrame, time_threshold: float = 1.5):
    # Filter accurate laps only
    laps = laps[laps['IsAccurate'] == True].copy()

    # Sort by LapStartTime
    laps = laps.sort_values(by='LapStartTime')

    # Compute traffic per lap
    traffic_data = []

    for _, lap in laps.iterrows():
        driver = lap['Driver']
        position = lap['Position']
        lap_start = lap['LapStartTime']

        # Car ahead = position - 1
        if position > 1:
            ahead_lap = laps[(laps['LapStartTime'] < lap_start) &
                             (laps['Position'] == position - 1)].tail(1)
            if not ahead_lap.empty:
                gap = (lap_start - ahead_lap['LapStartTime'].values[0]).total_seconds()
                in_traffic = gap <= time_threshold
            else:
                in_traffic = False
        else:
            in_traffic = False  # leader has no car ahead

        traffic_data.append({
            'Driver': driver,
            'LapNumber': lap['LapNumber'],
            'InTraffic': in_traffic
        })

    df = pd.DataFrame(traffic_data)
    # Aggregate count per driver
    traffic_stats = df.groupby('Driver')['InTraffic'].sum().reset_index()
    traffic_stats.rename(columns={'InTraffic': 'LapsInTraffic'}, inplace=True)
    return traffic_stats

def filter_race_laps(laps):
    """Return only laps from the race session that are accurate and on-track (green flag)."""
    return laps[
        (laps['IsAccurate']) &
        (laps['TrackStatus'] == '1')  # 1 = green flag
    ].copy()

def get_tire_degradation_data(laps, drivers=None):
    """
    Return accurate laps with LapTime in seconds and tyre compound.
    Optionally filter by selected drivers.
    """
    df = laps[laps['IsAccurate'] == True].copy()
    if drivers:
        df = df[df['Driver'].isin(drivers)]
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
    return df[['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound']]

import plotly.express as px
def plot_telemetry(df, y_column, title):
    """
    Generic telemetry plot for multiple drivers.
    """
    if df.empty:
        return None

    fig = px.line(
        df,
        x='Distance',
        y=y_column,
        color='Driver',
        title=title,
        labels={'Distance': 'Distance (m)', y_column: y_column},
        template='plotly_dark'
    )
    fig.update_layout(height=400)
    return fig