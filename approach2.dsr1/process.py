import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Load lidar data
# Expected columns: timestamp, quality, angle (in degrees), distance
lidar_cols = ['timestamp', 'quality', 'angle', 'distance']
lidar_data = pd.read_csv('lidar_data.txt', sep='\t', header=None, names=lidar_cols)

# Load IMU data
# Expected columns: timestamp, pitch (from gx), roll (from gy)
imu_cols = ['timestamp', 'pitch', 'roll']
imu_data = pd.read_csv('imu_data.txt', sep='\t', header=None, names=imu_cols)

# Function to get the nearest IMU record for a given timestamp
def get_nearest_imu(ts, imu_df):
    diff = abs(imu_df['timestamp'] - ts)
    return imu_df.iloc[diff.idxmin()]

# For each lidar measurement, add the corresponding pitch and roll
pitch_list = []
roll_list = []
for ts in lidar_data['timestamp']:
    imu_record = get_nearest_imu(ts, imu_data)
    pitch_list.append(imu_record['pitch'])
    roll_list.append(imu_record['roll'])
lidar_data['pitch'] = pitch_list
lidar_data['roll'] = roll_list

# Convert lidar polar coordinates to Cartesian in the lidar frame
theta = np.deg2rad(lidar_data['angle'].values)  # convert angle to radians
r = lidar_data['distance'].values
x_lidar = r * np.cos(theta)
y_lidar = r * np.sin(theta)
z_lidar = np.zeros_like(x_lidar)

# Compute rotated coordinates using the IMU pitch and roll.
points = []
for xi, yi, zi, pitch_angle, roll_angle in zip(x_lidar, y_lidar, z_lidar,
                                                 lidar_data['pitch'].values,
                                                 lidar_data['roll'].values):
    # Create rotation matrices (angles assumed in radians; if your IMU data is in degrees, convert them)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
                    [0, np.sin(pitch_angle),  np.cos(pitch_angle)]])
    R_y = np.array([[np.cos(roll_angle), 0, np.sin(roll_angle)],
                    [0, 1, 0],
                    [-np.sin(roll_angle), 0, np.cos(roll_angle)]])
    # Combined rotation: apply pitch then roll
    R = R_y.dot(R_x)
    point = R.dot(np.array([xi, yi, zi]))
    points.append(point)
points = np.array(points)

# Create an interactive 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=2)
)])
fig.update_layout(title='3D Point Cloud from 2D Lidar and IMU Data',
                  scene=dict(
                      xaxis_title='X',
                      yaxis_title='Y',
                      zaxis_title='Z'
                  ))
fig.show()
