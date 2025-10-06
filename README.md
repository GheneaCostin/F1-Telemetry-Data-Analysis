F1 Telemetry Dashboard

A Python + Streamlit application for analyzing Formula 1 telemetry and race data. This dashboard allows users to explore driver performance, tire degradation, lap times, and visualize track maps colored by fastest teams.

Features
1. Race & Session Selection

Select Year, Grand Prix, and Session Type (Practice, Qualifying, Race) from the sidebar.

Automatically loads session data and caches it for fast retrieval.

2. Driver & Lap Filtering

Filter laps by drivers.

Choose a specific lap number to analyze telemetry data.

3. Lap Time Analysis

Boxplots for lap times by finishing position.

Top 10 fastest laps table (overall).

Lap time consistency bar chart.

4. Traffic Analysis

Visualizes laps spent in traffic (<1.5s behind another car).

Insights into how drivers handle congestion on track.

5. Tire Degradation

Shows stint-normalized lap times to highlight degradation trends.

Multi-driver tire degradation comparison.

6. Team Performance Quadrant

Combines top speed, mean speed, and a fastest-team track map.

Track map shows which team was fastest on each segment of the circuit.

Uses optimized 2D binning for performance.

7. Telemetry Visualization

Compare Speed, Throttle (%), Brake (%), Gear , RPM for multiple drivers.

Interactive plots with Streamlit sliders and multi-select driver options.
