import xarray as xr
import dask.array as da
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px

# Step 1: Load the Dataset with Dask
ds = xr.open_mfdataset(
    "Z:/NARR/air/air.200112.nc",
    combine="by_coords",
    parallel=True,
    chunks={"time": 1, "level": 1, "y": 277,
            "x": 349}  # Adjust chunks as needed
)

# Select the 'air' variable
air = ds['air']

# Step 2: Preprocess the Data

# For Isolation Forest, we need a 2D array where each row is a sample and each column is a feature.
# Here, we'll create features from 'air', 'level', 'time', 'lat', and 'lon'.

# Flatten the spatial dimensions
air_reshaped = air.stack(sample=('time', 'level', 'y', 'x'))

# Extract coordinates
time = ds['time'].values
level = ds['level'].values
lat = ds['lat'].values.flatten()
lon = ds['lon'].values.flatten()

# Create a pandas DataFrame for features
# To manage memory, we may need to sample the data
# Here, we'll sample 10% of the data for demonstration
sampling_fraction = 0.1
total_samples = air_reshaped.size
sample_size = int(total_samples * sampling_fraction)

# Randomly sample indices
np.random.seed(42)  # For reproducibility
sample_indices = np.random.choice(
    total_samples, size=sample_size, replace=False)

# Extract sampled data
# Compute Dask array
air_sampled = air_reshaped.values[sample_indices].compute()
time_sampled = np.unravel_index(sample_indices, air_reshaped.shape)[0]
level_sampled = np.unravel_index(sample_indices, air_reshaped.shape)[1]
y_sampled = np.unravel_index(sample_indices, air_reshaped.shape)[2]
x_sampled = np.unravel_index(sample_indices, air_reshaped.shape)[3]

# Create DataFrame
df = pd.DataFrame({
    'air': air_sampled,
    'time': pd.to_datetime(ds['time'].values[time_sampled]),
    'level': ds['level'].values[level_sampled],
    'lat': lat[y_sampled],
    'lon': lon[x_sampled]
})

# Feature Engineering
# Convert time to numerical features (e.g., timestamp)
# Convert to seconds since epoch
df['time_num'] = df['time'].astype(np.int64) // 10**9

# Select features
features = df[['air', 'level', 'time_num', 'lat', 'lon']].values

# Handle missing values if any
# Isolation Forest in scikit-learn does not handle NaNs, so we need to remove them
mask = ~np.isnan(features).any(axis=1)
features = features[mask]
df = df[mask]

# Step 3: Fit Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    contamination='auto',
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(features)

# Step 4: Get Anomaly Scores
# The decision_function gives the anomaly score
scores = iso_forest.decision_function(features)
anomaly_scores = iso_forest.score_samples(features)

# Add scores back to the DataFrame
df['anomaly_score'] = anomaly_scores

# Step 5: Integrate Anomaly Scores Back to the Dataset
# Create a new DataArray for anomaly scores
# Initialize with NaNs
anomaly_da = xr.DataArray(
    data=np.nan,
    dims=("time", "level", "y", "x"),
    coords=ds.coords
)

# Assign anomaly scores to the sampled points
# Create a pandas MultiIndex for assignment
multi_index = pd.MultiIndex.from_arrays([
    df['time'],
    df['level'],
    df['y'],
    df['x']
], names=['time', 'level', 'y', 'x'])

# Create a DataArray from the scores
scores_da = xr.DataArray(
    data=df['anomaly_score'].values,
    dims=['sample'],
    coords={'sample': multi_index}
)

# Assign scores to the anomaly_da
anomaly_da = anomaly_da.assign_coords(sample=('sample', multi_index))
anomaly_da = anomaly_da.where(False)  # Initialize all as NaN
anomaly_da = anomaly_da.combine_first(
    scores_da.to_dataset(name='anomaly_score')['anomaly_score'])

# Optionally, compute the anomaly_da
# Be cautious with memory; consider saving to disk or using dask if needed
# Here, we'll compute a subset for visualization

# Step 6: Visualization with Plotly

# For visualization, we'll create a grid of anomaly scores for each time and level
# Due to the large number of levels and times, we'll use sliders to navigate

# Select a subset or ensure data is manageable
# For demonstration, we'll limit to a few time points and levels
# Adjust as needed

# Extract unique times and levels
unique_times = np.unique(df['time'])
unique_levels = np.unique(df['level'])

# Create a grid of anomaly scores
# Initialize a dictionary to hold frames
frames = []

# Create initial figure
initial_time = unique_times[0]
initial_level = unique_levels[0]

for i, current_time in enumerate(unique_times):
    for j, current_level in enumerate(unique_levels):
        # Filter data for current time and level
        mask = (df['time'] == current_time) & (df['level'] == current_level)
        df_subset = df[mask]

        if df_subset.empty:
            # If no data for this combination, skip
            continue

        fig = go.Scattergeo(
            lon=df_subset['lon'],
            lat=df_subset['lat'],
            marker=dict(
                size=5,
                color=df_subset['anomaly_score'],
                colorscale='RdBu',
                colorbar=dict(title="Anomaly Score"),
                reversescale=True
            ),
            mode='markers',
            name=f"Time: {current_time}, Level: {current_level}"
        )

        frames.append(go.Frame(data=[fig],
                               name=f"time_{i}_level_{j}",
                               traces=[0]))

# Create the initial data
mask = (df['time'] == initial_time) & (df['level'] == initial_level)
df_initial = df[mask]

scatter = go.Scattergeo(
    lon=df_initial['lon'],
    lat=df_initial['lat'],
    marker=dict(
        size=5,
        color=df_initial['anomaly_score'],
        colorscale='RdBu',
        colorbar=dict(title="Anomaly Score"),
        reversescale=True
    ),
    mode='markers'
)

# Define sliders
steps = []
for i, current_time in enumerate(unique_times):
    for j, current_level in enumerate(unique_levels):
        frame_name = f"time_{i}_level_{j}"
        step = dict(
            method="animate",
            args=[[frame_name],
                  {"mode": "immediate",
                   "frame": {"duration": 300, "redraw": True},
                   "transition": {"duration": 0}}],
            label=f"Time: {current_time.strftime(
                '%Y-%m-%d')}, Level: {current_level}"
        )
        steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Time & Level: "},
    pad={"t": 50},
    steps=steps
)]

# Create the figure
fig = go.Figure(
    data=[scatter],
    layout=go.Layout(
        title="Isolation Forest Anomaly Scores",
        geo=dict(
            scope='north america',
            projection=go.layout.geo.Projection(type='lambert'),
            showland=True,
            landcolor="lightgray",
            coastlinecolor="black"
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 300, "redraw": True},
                                  "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}])
            ],
            pad={"r": 10, "t": 87},
            showactive=False,
            x=0.1,
            y=0
        )],
        sliders=sliders
    ),
    frames=frames
)

fig.show()
