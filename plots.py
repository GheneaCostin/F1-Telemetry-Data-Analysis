import pandas as pd
import plotly.express as px

from telemetry import compute_lap_time_consistency
TEAM_COLORS = {
    "Mercedes": "#00D7B6",
    "Red Bull Racing": "#4781D7",
    "Ferrari": "#ED1131",
    "McLaren": "#F47600",
    "Alpine": "#00A1E8",
    "Racing Bulls": "#6C98FF",
    "Aston Martin": "#229971",
    "Williams": "#1868DB",
    "Kick Sauber": "#01C00E",
    "Haas": "#9C9FA2",
}

def lap_time_boxplot_dark_by_finish(laps, session):
    """
    F1-style dark boxplot for lap times per driver.
    Orders drivers by official finishing position, no gap labels.
    """
    laps = laps.copy()

    # Ensure LapTimeSeconds exists
    if 'LapTimeSeconds' not in laps.columns:
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Get official finishing order
    results = session.results.copy()
    results = results.dropna(subset=['Position'])  # drop drivers with no official position
    driver_order = results.sort_values('Position')['Abbreviation'].tolist()

    # Set driver as categorical to preserve order
    laps['Driver'] = pd.Categorical(laps['Driver'], categories=driver_order, ordered=True)

    # Map team colors
    laps['TeamColor'] = laps['Team'].map(TEAM_COLORS)

    # Create boxplot
    fig = px.box(
        laps,
        x="Driver",
        y="LapTimeSeconds",
        color="Team",
        points=False,
        hover_data=["LapNumber", "Compound"],
        color_discrete_map=TEAM_COLORS,
        category_orders={"Driver": driver_order},
        title="Lap Time Distribution per Driver (Ordered by Finish)",
        labels={"LapTimeSeconds": "Lap Time (s)"},
        width=1400,
        height=700
    )

    fig.update_traces(
        line=dict(width=2),
        opacity=0.9,
        width=0.6
    )

    # Dark F1-style layout
    fig.update_layout(
        plot_bgcolor="#1c1c1c",
        paper_bgcolor="#121212",
        font=dict(color="white", size=14),
        boxmode="group",
        boxgap=0.3,
        xaxis=dict(
            showgrid=False,
            tickangle=-45,
            color="white"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#333333",
            color="white",
            dtick=1
        ),
        legend=dict(title=dict(text="Team", font=dict(color="white"))),
        title_x=0.5,
        title_font=dict(size=22, color="white")
    )

    return fig


def style_top_fastest_laps_table(df):
    """
    Style the top fastest laps table with:
    - Team colors
    - Formatted lap times
    - Rank starting from 1
    - LapNumber and TyreLife as integers
    """
    df = df.copy()

    # Convert LapTime to string for display
    df['LapTimeStr'] = df['LapTime'].apply(lambda x: f"{int(x.total_seconds() // 60)}:{x.total_seconds() % 60:06.3f}")

    # Ensure LapNumber and TyreLife are integers
    df['LapNumber'] = df['LapNumber'].astype(int)
    df['TyreLife'] = df['TyreLife'].astype(int)

    # Round LapTimeSeconds to 3 decimals
    df['LapTimeSeconds'] = df['LapTimeSeconds'].round(3)

    # Reset index for rank display
    df = df.reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    # Apply team color styling
    def team_color(row):
        color = TEAM_COLORS.get(row['Team'], "#888888")
        return [f"background-color: {color}; color: white"] * len(row)

    # Columns to display
    display_df = df[['Driver', 'Team', 'LapNumber', 'LapTimeStr', 'LapTimeSeconds', 'Compound', 'TyreLife']]

    return display_df.style.apply(team_color, axis=1)



def lap_time_consistency_bar(laps, drivers=None, accurate_only=True):
    """
    Creates a bar chart of lap time consistency per driver.
    Only includes laps with IsAccurate==True if accurate_only=True.
    """
    # Compute consistency
    df = compute_lap_time_consistency(laps, drivers=drivers, accurate_only=accurate_only)

    # Map each driver to their team using laps DataFrame
    driver_team_map = laps.drop_duplicates('Driver').set_index('Driver')['Team'].to_dict()
    df['TeamColor'] = df['Driver'].map(lambda d: TEAM_COLORS.get(driver_team_map.get(d, ""), "#888888"))

    # Create the bar chart
    fig = px.bar(
        df,
        x='Driver',
        y='Consistency',
        text='Consistency',
        color='Driver',
        color_discrete_map={row['Driver']: row['TeamColor'] for _, row in df.iterrows()},
        title="Lap Time Consistency per Driver (Mean Stint Std)"
    )

    # Style the chart
    fig.update_yaxes(dtick=0.1)
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font_color='white')

    return fig

def laps_in_traffic_bar(laps_in_traffic_df, session):
    # Add team colors
    driver_info = session.results[['DriverNumber', 'Abbreviation', 'TeamColor']].copy()
    driver_info.rename(columns={'Abbreviation': 'Driver'}, inplace=True)
    df = laps_in_traffic_df.merge(driver_info, on='Driver', how='left')

    # Sort by laps in traffic (descending)
    df = df.sort_values(by='LapsInTraffic', ascending=False)

    fig = px.bar(
        df,
        x='LapsInTraffic',
        y='Driver',
        color='Driver',
        orientation='h',
        color_discrete_map={row['Driver']: f"#{row['TeamColor']}" for _, row in df.iterrows()},
        text='LapsInTraffic',
        title="Laps Spent in Traffic (<1.5s Behind Car Ahead)"
    )

    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Number of Laps in Traffic",
        yaxis_title="Driver",
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white")
    )

    fig.update_traces(
        textposition='outside',
        textfont_size=12
    )

    return fig

def tire_degradation_multi_driver(df):
    """
    Tire degradation for multiple drivers in a single plot.
    X: LapNumber
    Y: LapTimeSeconds
    Color: Compound
    Line style: Driver
    """
    fig = px.line(
        df,
        x='LapNumber',
        y='LapTimeSeconds',
        color='Compound',          # tyre color
        line_group='Driver',       # connect laps per driver
        hover_data=['Driver', 'LapNumber', 'LapTimeSeconds', 'Compound'],
        markers=True,
        title="Tire Degradation per Driver"
    )
    fig.update_layout(
        xaxis_title="Lap Number",
        yaxis_title="Lap Time (s)",
        legend_title="Tyre Compound",
        template="plotly_dark"  # keeps the dark F1-style theme
    )
    return fig


def tire_degradation_plot(laps, drivers=None, normalize=False, smooth_window=2):
    """
    Plot lap time vs tyre life (with optional normalization and smoothing).
    Each driver-compound line shows how lap times evolve as tires age.
    """
    df = laps.copy()

    # Keep only accurate laps
    df = df[df['IsAccurate'] == True]

    # Filter by selected drivers
    if drivers:
        df = df[df['Driver'].isin(drivers)]

    # Convert LapTime to seconds
    df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()

    # Normalize within each stint if requested
    if normalize:
        df['NormLapTime'] = df.groupby(['Driver', 'Stint'])['LapTimeSeconds'].transform(lambda x: x - x.iloc[0])
        y_col = 'NormLapTime'
        y_label = "Δ Lap Time vs First Lap in Stint (s)"
    else:
        y_col = 'LapTimeSeconds'
        y_label = "Lap Time (s)"

    # Smooth using rolling mean
    df[y_col] = df.groupby(['Driver', 'Stint'])[y_col].transform(
        lambda x: x.rolling(smooth_window, min_periods=1).mean()
    )

    # Sort so lines render properly
    df = df.sort_values(by=['Driver', 'Stint', 'TyreLife'])

    # Map tire colors (same as official F1)
    tire_colors = {
        'SOFT': '#ff4d4d',
        'MEDIUM': '#ffd633',
        'HARD': '#b3b3b3',
        'INTERMEDIATE': '#2eb82e',
        'WET': '#1a75ff'
    }

    # Create line plot
    fig = px.line(
        df,
        x='TyreLife',
        y=y_col,
        color='Driver',
        line_dash='Compound',
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['Driver', 'Compound', 'Stint', 'LapNumber', 'TyreLife'],
        labels={'TyreLife': 'Tire Age (laps)', y_col: y_label},
        title="Tire Degradation Trend per Driver"
    )

    # Overlay scatter markers for data points
    fig.add_scatter(
        x=df['TyreLife'],
        y=df[y_col],
        mode='markers',
        marker=dict(size=5, color=df['Compound'].map(tire_colors), opacity=0.8),
        name="Laps"
    )

    # Improve styling
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='gray', title='Tire Life (laps)'),
        yaxis=dict(gridcolor='gray', title=y_label),
        legend=dict(title='Driver / Compound', bgcolor='rgba(0,0,0,0)'),
        hovermode="x unified"
    )

    return fig


import pandas as pd
import plotly.express as px

def team_top_speed_delta_chart(laps):
    laps = laps.copy()
    laps = laps[(laps['IsAccurate'] == True) & (laps['LapTime'].notna())]
    laps['SpeedST'] = pd.to_numeric(laps['SpeedST'], errors='coerce')
    laps = laps.dropna(subset=['SpeedST', 'Team', 'Driver'])
    laps = laps[laps['SpeedST'] < 370]

    driver_top_speeds = (
        laps.groupby(['Team', 'Driver'])['SpeedST']
        .quantile(0.95)
        .reset_index()
    )

    team_top_speeds = (
        driver_top_speeds.groupby('Team')['SpeedST']
        .mean()
        .reset_index()
        .rename(columns={'SpeedST': 'TeamTopSpeed'})
    )

    avg_speed = team_top_speeds['TeamTopSpeed'].mean()
    team_top_speeds['DeltaToAvg'] = team_top_speeds['TeamTopSpeed'] - avg_speed

    team_top_speeds['Color'] = team_top_speeds['Team'].map(TEAM_COLORS)

    fig = px.bar(
        team_top_speeds,
        y='Team',
        x='DeltaToAvg',
        color='Team',
        color_discrete_map=TEAM_COLORS,
        orientation='h',
        text='DeltaToAvg',
        title="Team Top Speed Delta to Session Average (Speed Trap)"
    )
    fig.update_traces(
        texttemplate='%{text:.2f} km/h',
        hovertemplate=(
            "Team: %{y}<br>"
            "Delta to Avg: %{x:.2f} km/h<br>"
            "Top Speed: %{customdata[0]:.1f} km/h"
        ),
        customdata=team_top_speeds[['TeamTopSpeed']].values
    )
    fig.update_layout(
        xaxis_title="Δ to Session Average (km/h)",
        yaxis_title="Team",
        showlegend=False,
        height=600,
    )

    return fig, team_top_speeds


def team_mean_speed_delta_chart(laps_filtered, session_obj):
    # Filter accurate laps and numeric LapTime
    laps = laps_filtered.copy()
    laps = laps[(laps['IsAccurate'] == True) & (laps['LapTime'].notna())]
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

    # Compute mean lap time per driver
    driver_mean_time = laps.groupby(['Driver', 'Team'])['LapTimeSeconds'].mean().reset_index()

    # Get track length in km
    try:
        track_length = session_obj.event['Length']  # in km
    except KeyError:
        track_length = 1  # fallback if not available

    # Compute mean speed for each driver
    driver_mean_time['MeanSpeed'] = track_length / (driver_mean_time['LapTimeSeconds'] / 3600)  # km/h

    # Average mean speed per team
    team_mean_speed = driver_mean_time.groupby('Team')['MeanSpeed'].mean().reset_index()

    # Compute delta to session average
    avg_speed = team_mean_speed['MeanSpeed'].mean()
    team_mean_speed['DeltaToAvg'] = team_mean_speed['MeanSpeed'] - avg_speed

    # Sort for better visualization
    team_mean_speed = team_mean_speed.sort_values('DeltaToAvg', ascending=False)

    # Add team colors
    team_mean_speed['Color'] = team_mean_speed['Team'].map(TEAM_COLORS)

    # Plot horizontal bar chart
    fig = px.bar(
        team_mean_speed,
        y='Team',
        x='DeltaToAvg',
        color='Team',
        color_discrete_map=TEAM_COLORS,
        orientation='h',
        text='DeltaToAvg',
        title="Team Mean Speed Delta to Session Average"
    )

    fig.update_traces(
        texttemplate='%{text:.2f} km/h',
        hovertemplate=(
            "Team: %{y}<br>"
            "Δ to Avg: %{x:.2f} km/h<br>"
            "Mean Speed: %{customdata[0]:.2f} km/h"
        ),
        customdata=team_mean_speed[['MeanSpeed']].values
    )

    fig.update_layout(
        xaxis_title="Δ to Session Average (km/h)",
        yaxis_title="Team",
        showlegend=False,
        height=600,
    )

    return fig, team_mean_speed

def team_top_vs_mean_speed_chart_from_aggregates(team_top_speeds_df, team_mean_speeds_df):
    import plotly.express as px
    import pandas as pd

    team_speeds = pd.merge(team_top_speeds_df, team_mean_speeds_df, on='Team')
    team_speeds['Color'] = team_speeds['Team'].map(TEAM_COLORS)

    fig = px.scatter(
        team_speeds,
        x='MeanSpeed',
        y='TeamTopSpeed',
        color='Team',
        color_discrete_map=TEAM_COLORS,
        text='Team',
        title=" Team Top Speed vs Mean Speed",
        hover_data = ['Team', 'MeanSpeed', 'TeamTopSpeed']
    )

    fig.update_traces(
        textposition='top center',
        hovertemplate=(
            "Team: %{text}<br>"
            "Top Speed: %{y:.1f} km/h<br>"
            "Mean Speed: %{x:.1f} km/h"
        )
    )
    fig.update_layout(
        xaxis_title="Mean Speed (km/h)",
        yaxis_title="Top Speed (km/h)",
        showlegend=False,
        height=600,
    )

    return fig

from plotly.subplots import make_subplots

def team_performance_quadrant(top_speed_fig, mean_speed_fig, scatter_fig, bottom_right_fig=None):
    """
    Arrange 4 charts in a 2x2 quadrant.
    """
    # Create subplot figure with 2x2 layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Team Top Speed Delta",
            "Top vs Mean Speed",
            "Team Mean Speed Delta",
            "" if bottom_right_fig is None else bottom_right_fig.layout.title.text
        ),
        horizontal_spacing=0.15,
        vertical_spacing=0.15
    )

    # Add top-left chart: Top Speed Delta
    for trace in top_speed_fig.data:
        fig.add_trace(trace, row=1, col=1)

    # Add top-right chart: Top vs Mean Scatter
    for trace in scatter_fig.data:
        fig.add_trace(trace, row=1, col=2)

    # Add bottom-left chart: Mean Speed Delta
    for trace in mean_speed_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # Add bottom-right chart if provided
    if bottom_right_fig is not None:
        for trace in bottom_right_fig.data:
            fig.add_trace(trace, row=2, col=2)

    # Update layout for each subplot
    fig.update_layout(
        height=1000, width=1200,
        showlegend=False,
        title_text="Team Performance Quadrant"
    )

    # Optional: adjust axes titles individually
    fig.update_xaxes(title_text="Δ to Session Avg (km/h)", row=1, col=1)
    fig.update_xaxes(title_text="Mean Speed (km/h)", row=1, col=2)
    fig.update_xaxes(title_text="Δ to Session Avg (km/h)", row=2, col=1)

    fig.update_yaxes(title_text="Team", row=1, col=1)
    fig.update_yaxes(title_text="Top Speed (km/h)", row=1, col=2)
    fig.update_yaxes(title_text="Team", row=2, col=1)

    return fig


import pandas as pd
import plotly.express as px
import fastf1
from fastf1 import plotting

TEAM_COLORS = {
    "Mercedes": "#00D7B6",
    "Red Bull Racing": "#4781D7",
    "Ferrari": "#ED1131",
    "McLaren": "#F47600",
    "Alpine": "#00A1E8",
    "Racing Bulls": "#6C98FF",
    "Aston Martin": "#229971",
    "Williams": "#1868DB",
    "Kick Sauber": "#01C00E",
    "Haas": "#9C9FA2",
}

import plotly.graph_objects as go

import pandas as pd
import plotly.graph_objects as go
import fastf1
from fastf1 import plotting
import numpy as np

TEAM_COLORS = {
    "Mercedes": "#00D7B6",
    "Red Bull Racing": "#4781D7",
    "Ferrari": "#ED1131",
    "McLaren": "#F47600",
    "Alpine": "#00A1E8",
    "Racing Bulls": "#6C98FF",
    "Aston Martin": "#229971",
    "Williams": "#1868DB",
    "Kick Sauber": "#01C00E",
    "Haas": "#9C9FA2",
}

import pandas as pd
import numpy as np
import plotly.graph_objects as go

TEAM_COLORS = {
    "Mercedes": "#00D7B6",
    "Red Bull Racing": "#4781D7",
    "Ferrari": "#ED1131",
    "McLaren": "#F47600",
    "Alpine": "#00A1E8",
    "Racing Bulls": "#6C98FF",
    "Aston Martin": "#229971",
    "Williams": "#1868DB",
    "Kick Sauber": "#01C00E",
    "Haas": "#9C9FA2",
}


def track_segments_fastest_team_optimized(laps_df, session_obj, drivers=None, grid_size=50):
    """
    Optimized track map using 2D binning to assign fastest team per segment.

    Parameters:
        laps_df : pd.DataFrame
            Filtered laps with 'Driver', 'LapTimeSeconds', 'IsAccurate'
        session_obj : FastF1 session object
        drivers : list, optional
            Drivers to include
        grid_size : int
            Number of bins along X and Y axes
    """
    laps_df = laps_df.copy()
    if drivers:
        laps_df = laps_df[laps_df['Driver'].isin(drivers)]
    laps_df = laps_df[laps_df['IsAccurate'] == True]

    # Attach telemetry for all laps
    all_tel = []
    for _, row in laps_df.iterrows():
        lap = session_obj.laps.loc[row.name]
        tel = lap.get_telemetry()[['X', 'Y', 'Speed']].copy()
        if tel.empty:
            continue
        tel['Team'] = row['Team']
        all_tel.append(tel)
    combined_tel = pd.concat(all_tel, ignore_index=True)

    # Build 2D bins
    x_bins = np.linspace(combined_tel['X'].min(), combined_tel['X'].max(), grid_size)
    y_bins = np.linspace(combined_tel['Y'].min(), combined_tel['Y'].max(), grid_size)

    # Assign each telemetry point to a bin
    combined_tel['X_bin'] = np.digitize(combined_tel['X'], x_bins) - 1
    combined_tel['Y_bin'] = np.digitize(combined_tel['Y'], y_bins) - 1

    # Find fastest team per bin
    fastest_team_bin = combined_tel.groupby(['X_bin', 'Y_bin']).apply(
        lambda df: df.loc[df['Speed'].idxmax(), 'Team']
    ).to_dict()

    # Reference lap for track line
    fastest_idx = laps_df['LapTimeSeconds'].idxmin()
    ref_lap = session_obj.laps.loc[fastest_idx]
    ref_tel = ref_lap.get_telemetry()[['X', 'Y']].copy()

    # Map ref lap points to bins and assign colors
    ref_tel['X_bin'] = np.digitize(ref_tel['X'], x_bins) - 1
    ref_tel['Y_bin'] = np.digitize(ref_tel['Y'], y_bins) - 1
    ref_tel['Color'] = ref_tel.apply(
        lambda row: TEAM_COLORS.get(fastest_team_bin.get((row['X_bin'], row['Y_bin']), None), '#888888'),
        axis=1
    )

    # Draw colored line segments
    fig = go.Figure()
    for i in range(1, len(ref_tel)):
        fig.add_trace(go.Scatter(
            x=ref_tel['X'].iloc[i - 1:i + 1],
            y=ref_tel['Y'].iloc[i - 1:i + 1],
            mode='lines',
            line=dict(color=ref_tel['Color'].iloc[i], width=3),
            showlegend=False
        ))

    fig.update_layout(
        title="Track Map Colored by Fastest Team (All Accurate Laps, Optimized)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="#121212",
        paper_bgcolor="#121212",
        height=600,
        width=800
    )

    return fig


def get_telemetry_for_drivers(laps_df, session_obj, drivers, lap_number, sample_rate=1):
    """
    Get telemetry for selected drivers and a specific lap.

    Parameters:
        laps_df : pd.DataFrame
            All laps (unfiltered)
        session_obj : FastF1 session object
        drivers : list
            Drivers to include
        lap_number : int
            Lap number to retrieve
        sample_rate : int
            Step to downsample telemetry (1 = all points, 2 = every other point, etc.)

    Returns:
        pd.DataFrame : Combined telemetry for all selected drivers, with columns:
                       ['Driver', 'Distance', 'Speed', 'Throttle', 'Brake', 'nGear', 'RPM']
                       Brake/Throttle are in percentage (0-100%)
    """
    all_tel = []

    for driver in drivers:
        # Pick the lap object for the driver and lap number
        driver_laps = laps_df[laps_df['Driver'] == driver]
        lap_obj = session_obj.laps.loc[driver_laps.index].query(f'LapNumber == {lap_number}')
        if lap_obj.empty:
            continue
        lap_obj = lap_obj.iloc[0]  # single lap

        tel = lap_obj.get_telemetry()
        if tel.empty:
            continue

        # Downsample if requested
        tel = tel.iloc[::sample_rate].copy()

        # Convert Throttle/Brake to percentage
        if 'Throttle' in tel.columns:
            tel['Throttle'] = tel['Throttle'] * 100
        if 'Brake' in tel.columns:
            tel['Brake'] = tel['Brake'] * 100

        # Add driver info
        tel['Driver'] = driver

        # Keep only relevant columns
        tel = tel[['Driver', 'Distance', 'Speed', 'Throttle', 'Brake', 'nGear', 'RPM']]

        all_tel.append(tel)

    if not all_tel:
        return pd.DataFrame(columns=['Driver', 'Distance', 'Speed', 'Throttle', 'Brake', 'nGear', 'RPM'])

    return pd.concat(all_tel, ignore_index=True)