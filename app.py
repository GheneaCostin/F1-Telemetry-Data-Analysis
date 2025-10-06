# app.py
import plotly.express as px
import streamlit as st
import fastf1
from telemetry import get_laps_in_traffic, filter_race_laps, get_tire_degradation_data, plot_telemetry
from telemetry import get_filtered_laps, get_top_fastest_laps
from plots import lap_time_boxplot_dark_by_finish, style_top_fastest_laps_table, tire_degradation_plot, \
    tire_degradation_multi_driver, team_top_speed_delta_chart, team_mean_speed_delta_chart, team_performance_quadrant, \
    get_telemetry_for_drivers
from plots import lap_time_consistency_bar
from plots import laps_in_traffic_bar
from telemetry import get_laps_in_traffic
from plots import tire_degradation_multi_driver
from fastf1 import plotting
from plots import team_top_vs_mean_speed_chart_from_aggregates




# Enable FastF1 cache
fastf1.Cache.enable_cache('cache')

st.title("F1 Telemetry Dashboard")

# ---- Get all GPs for that year ----
@st.cache_data(show_spinner=True)
def get_gp_list(year):
    events = fastf1.get_event_schedule(year)
    # Filter out non-race events
    race_events = events[events['EventName'].str.contains('Grand Prix')]
    return race_events['EventName'].tolist()

# ---- Sidebar: Year selection ----
year = st.sidebar.selectbox("Year", [2024, 2025], index=1)

# ---- Get filtered GP list ----
gp_list = get_gp_list(year)

# ---- Default GP selection ----
default_gp = "Azerbaijan Grand Prix" if "Azerbaijan Grand Prix" in gp_list else gp_list[0]
gp = st.sidebar.selectbox("Grand Prix", gp_list, index=gp_list.index(default_gp))

# ---- Sidebar: Session type ----
session_type = st.sidebar.selectbox("Session Type", ["P", "Q", "R"], index=2)

st.write(f"Loading {gp} {year} {session_type} session...")

# ---- Load session and laps (cached) ----
@st.cache_data(show_spinner=True)
def load_laps(year, gp, session_type):
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    laps_df = session.laps.copy()
    return laps_df, session

laps, session_obj = load_laps(year, gp, session_type)

# ---- Filter laps (accurate only initially) ----
laps_filtered = laps_filtered = filter_race_laps(laps)

# ---- Driver selection above plot ----
st.subheader("Select Drivers")
all_drivers = laps_filtered['Driver'].unique()
selected_drivers = st.multiselect("Drivers", options=all_drivers, default=list(all_drivers))

# Filter laps by selected drivers
laps_filtered = get_filtered_laps(laps_filtered, drivers=selected_drivers, accurate_only=False)

# ---- Lap Time Boxplot ----
fig = lap_time_boxplot_dark_by_finish(laps_filtered, session_obj)
st.plotly_chart(fig, use_container_width=True)

# ---- Top 10 fastest laps ----
st.subheader("Top 10 Fastest Laps (Overall)")
top_laps = get_top_fastest_laps(laps_filtered, top_n=10)
st.dataframe(style_top_fastest_laps_table(top_laps))

# ---- Lap time Consistency ----
st.subheader("Lap Time Consistency")
consistency_fig = lap_time_consistency_bar(
    laps_filtered,
    drivers=selected_drivers,
    accurate_only=True,
)
st.plotly_chart(consistency_fig, use_container_width=True)

# ---- Laps Spent in traffic ----
st.subheader("Laps Spent in Traffic (<1.5s behind car ahead)")
traffic_stats = get_laps_in_traffic(laps_filtered)
traffic_fig = laps_in_traffic_bar(traffic_stats, session_obj)
st.plotly_chart(traffic_fig, use_container_width=True)

# ---- Tire Degradation ----
st.subheader("Tire Degradation Analysis")

normalize_tires = st.checkbox("Normalize lap times by stint (show pure degradation trend)")
selected_drivers_degr = st.multiselect("Select Drivers for Tire Degradation", options=all_drivers, default=list(all_drivers))

tire_fig = tire_degradation_plot(laps_filtered, drivers=selected_drivers_degr, normalize=normalize_tires)
st.plotly_chart(tire_fig, use_container_width=True)

# ---- Team Performance Quadrant ----
st.subheader("Team Performance Quadrant")

fig_top_speed, team_top_df = team_top_speed_delta_chart(laps_filtered)
fig_mean_speed, team_mean_df = team_mean_speed_delta_chart(laps_filtered, session_obj)
fig_scatter = team_top_vs_mean_speed_chart_from_aggregates(team_top_df, team_mean_df)

# Generate bottom-right fastest-team track map
@st.cache_data
def get_fastest_team_track(_laps_df, _session_obj, grid_size=50):
    return track_segments_fastest_team_optimized(_laps_df, _session_obj, grid_size=grid_size)

from plots import track_segments_fastest_team_optimized

fig_fastest_team = get_fastest_team_track(laps_filtered, session_obj)

# Combine into quadrant
fig_quadrant = team_performance_quadrant(
    top_speed_fig=fig_top_speed,
    mean_speed_fig=fig_mean_speed,
    scatter_fig=fig_scatter,
    bottom_right_fig=fig_fastest_team
)

# Display
st.plotly_chart(fig_quadrant, use_container_width=True)

# Sidebar inputs
st.sidebar.subheader("Telemetry Comparison")

selected_drivers_telemetry = st.sidebar.multiselect(
    "Select Drivers", options=all_drivers, default=all_drivers[:2]
)

# Determine max lap number across selected drivers
max_lap_number = int(laps_filtered[laps_filtered['Driver'].isin(selected_drivers_telemetry)]['LapNumber'].max())
selected_lap = st.sidebar.slider(
    "Select Lap Number", min_value=1, max_value=max_lap_number, value=1
)

st.subheader(f"Telemetry Comparison - Lap {selected_lap}")

telemetry_df = get_telemetry_for_drivers(laps_filtered, session_obj, selected_drivers_telemetry, selected_lap)

if telemetry_df.empty:
    st.warning("No telemetry available for selected drivers/lap.")
else:
    for col in ['Speed', 'Throttle', 'Brake', 'nGear', 'RPM']:
        fig = plot_telemetry(telemetry_df, col, f"{col} vs Distance")
        st.plotly_chart(fig, use_container_width=True)




