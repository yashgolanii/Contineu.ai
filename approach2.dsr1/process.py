import numpy as np
import pandas as pd
import plotly.graph_objects as go

def extract_float(s):
    """
    Given a string like "GX:-12.11" or "2:-2.79", split by colon
    and return the last part as a float.
    If s is None or conversion fails, return NaN.
    """
    try:
        return float(str(s).split(":")[-1])
    except:
        return np.nan

# -----------------------
# 1. Load Lidar Data File
# -----------------------
# We assume the lidar file has a header with columns:
# timestamp, quality, angle, distance
lidar_data = pd.read_csv('lidar_data2.txt', sep=',', header=0)
print("Lidar data head:")
print(lidar_data.head())

# Convert columns to numeric
lidar_data['timestamp'] = pd.to_numeric(lidar_data['timestamp'], errors='coerce')
lidar_data['angle'] = pd.to_numeric(lidar_data['angle'], errors='coerce')
lidar_data['distance'] = pd.to_numeric(lidar_data['distance'], errors='coerce')

# -----------------------
# 2. Load IMU Data File Manually
# -----------------------
imu_rows = []
with open('imu_data2.txt', 'r') as f:
    lines = f.readlines()
    # Skip the header (first line)
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        # If there are 3 parts, assume format: timestamp, pitch, roll (and no yaw)
        if len(parts) == 3:
            timestamp_str, pitch_str, roll_str = parts
            yaw_str = None
        elif len(parts) >= 4:
            timestamp_str, pitch_str, roll_str, yaw_str = parts[:4]
        else:
            continue
        try:
            ts = float(timestamp_str)
        except:
            ts = np.nan
        pitch_val = extract_float(pitch_str)
        roll_val  = extract_float(roll_str)
        yaw_val   = extract_float(yaw_str) if yaw_str is not None else np.nan
        imu_rows.append([ts, pitch_val, roll_val, yaw_val])
imu_data = pd.DataFrame(imu_rows, columns=['timestamp', 'pitch', 'roll', 'yaw'])
print("IMU data head:")
print(imu_data.head())

# -----------------------
# 3. (Optional) Convert IMU timestamps
# -----------------------
# Check whether the IMU timestamps are in milliseconds or already in seconds.
# Your sample shows values around 1.741259e+09, which looks like Unix time in seconds.
# If they were in ms, you would convert by dividing by 1000.
# For now, we assume they are in seconds.
imu_data['timestamp'] = pd.to_numeric(imu_data['timestamp'], errors='coerce')

# -----------------------
# 4. Sort and Normalize Time Stamps
# -----------------------
# Sort by timestamp
lidar_data.sort_values('timestamp', inplace=True)
lidar_data.reset_index(drop=True, inplace=True)
imu_data.sort_values('timestamp', inplace=True)
imu_data.reset_index(drop=True, inplace=True)

# Print original time ranges:
print("Original lidar time range:", lidar_data['timestamp'].min(), "to", lidar_data['timestamp'].max())
print("Original IMU time range (s):", imu_data['timestamp'].min(), "to", imu_data['timestamp'].max())

# We align the two by subtracting each file's minimum timestamp.
lidar_data['timestamp'] = lidar_data['timestamp'] - lidar_data['timestamp'].min()
imu_data['timestamp']   = imu_data['timestamp'] - imu_data['timestamp'].min()

print("Normalized lidar time range:", lidar_data['timestamp'].min(), "to", lidar_data['timestamp'].max())
print("Normalized IMU time range:", imu_data['timestamp'].min(), "to", imu_data['timestamp'].max())

# -----------------------
# 5. Determine Common Time Window and Filter Data
# -----------------------
common_start = max(lidar_data['timestamp'].min(), imu_data['timestamp'].min())
common_end   = min(lidar_data['timestamp'].max(), imu_data['timestamp'].max())

if common_start >= common_end:
    raise ValueError("No overlapping time window between lidar and IMU data.")

print("Common time window:", common_start, "to", common_end)

lidar_data = lidar_data[(lidar_data['timestamp'] >= common_start) & (lidar_data['timestamp'] <= common_end)]
imu_data   = imu_data[(imu_data['timestamp'] >= common_start) & (imu_data['timestamp'] <= common_end)]

# -----------------------
# 6. Synchronize Data: For each lidar record, find the nearest IMU record.
# -----------------------
def get_nearest_imu(ts, imu_df):
    diff = abs(imu_df['timestamp'] - ts)
    diff_clean = diff.dropna()
    if diff_clean.empty:
        return None
    idx = diff_clean.idxmin()
    return imu_df.loc[idx]

pitch_list = []
roll_list = []
for ts in lidar_data['timestamp']:
    imu_record = get_nearest_imu(ts, imu_data)
    if imu_record is None:
        pitch_list.append(np.nan)
        roll_list.append(np.nan)
    else:
        pitch_list.append(imu_record['pitch'])
        roll_list.append(imu_record['roll'])
lidar_data['pitch'] = pitch_list
lidar_data['roll']  = roll_list

# -----------------------
# 7. Convert 2D Lidar Data to Cartesian Coordinates
# -----------------------
theta = np.deg2rad(lidar_data['angle'].values)
r = lidar_data['distance'].values
x_lidar = r * np.cos(theta)
y_lidar = r * np.sin(theta)
z_lidar = np.zeros_like(x_lidar)

# -----------------------
# 8. "Lift" 2D Points into 3D Using IMU Data
# -----------------------
points = []
for xi, yi, zi, pitch_angle, roll_angle in zip(
        x_lidar, y_lidar, z_lidar,
        lidar_data['pitch'].values,
        lidar_data['roll'].values):
    # Convert IMU angles (in degrees) to radians.
    pitch_rad = np.deg2rad(pitch_angle)
    roll_rad  = np.deg2rad(roll_angle)
    # Rotation matrix for pitch (rotation about X-axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ])
    # Rotation matrix for roll (rotation about Y-axis)
    R_roll = np.array([
        [np.cos(roll_rad), 0, np.sin(roll_rad)],
        [0, 1, 0],
        [-np.sin(roll_rad), 0, np.cos(roll_rad)]
    ])
    R = R_roll @ R_pitch  # Combined rotation: pitch then roll.
    point = R @ np.array([xi, yi, zi])
    points.append(point)
points = np.array(points)
print("Points shape:", points.shape)

if points.ndim != 2 or points.shape[1] != 3:
    print("No valid 3D points were generated. Check your data overlap and processing steps.")
    exit(1)

# -----------------------
# 9. Visualize with Plotly
# -----------------------
# --- Visualize the 3D Point Cloud with Plotly (Colorful) ---
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(
        size=3,
        color=lidar_data['angle'],  # Map color to the lidar's angle values
        colorscale='Viridis',       # You can try other colorscales like 'Viridis' or 'Jet'
        colorbar=dict(title='Angle (deg)'),
        opacity=0.6
    )
)])
fig.update_layout(
    title='3D Point Cloud from 2D Lidar and IMU Data',
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)
fig.show()

