F1 Telemetry Dashboard

A Python + Streamlit application for analyzing Formula 1 telemetry and race data. This dashboard allows users to explore driver performance, tire degradation, lap times, and visualize track maps colored by fastest teams.

Features
1. Race & Session Selection

Select Year, Grand Prix, and Session Type (Practice, Qualifying, Race) from the sidebar.

Automatically loads session data and caches it for fast retrieval.

<img width="1919" height="898" alt="image" src="https://github.com/user-attachments/assets/d6a6af42-86d8-4666-8334-a9b7c7eab9b8" />

2. Driver & Lap Filtering

Filter laps by drivers.

Choose a specific lap number to analyze telemetry data.

3. Lap Time Analysis

Boxplots for lap times by finishing position.

Top 10 fastest laps table (overall).

Lap time consistency bar chart.

<img width="758" height="808" alt="image" src="https://github.com/user-attachments/assets/638d2d8e-12f8-46bf-b6c2-0ebe425e30fb" />


4. Traffic Analysis

Visualizes laps spent in traffic (<1.5s behind another car).

Insights into how drivers handle congestion on track.

<img width="963" height="423" alt="image" src="https://github.com/user-attachments/assets/34699828-0864-4cd6-83d1-901cfa0f8087" />


5. Tire Degradation

Shows stint-normalized lap times to highlight degradation trends.

Multi-driver tire degradation comparison.

<img width="730" height="525" alt="image" src="https://github.com/user-attachments/assets/91872461-6b53-4dd9-acc6-fe44330e28bc" />


6. Team Performance Quadrant

Combines top speed, mean speed, and a fastest-team track map.

Track map shows which team was fastest on each segment of the circuit.

Uses optimized 2D binning for performance.

<img width="682" height="716" alt="image" src="https://github.com/user-attachments/assets/35be0430-d3bd-4686-878e-93f9cd4f108c" />


7. Telemetry Visualization

Compare Speed, Throttle (%), Brake (%), Gear , RPM for multiple drivers.

Interactive plots with Streamlit sliders and multi-select driver options.


<img width="1012" height="868" alt="image" src="https://github.com/user-attachments/assets/fd232160-0bf6-4f84-8ae8-c7a54d847b92" />

<img width="1074" height="827" alt="image" src="https://github.com/user-attachments/assets/11c2154d-775f-4094-97d8-a6b51cf28e1a" />

