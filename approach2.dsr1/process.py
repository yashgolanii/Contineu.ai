import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- Load Lidar Data ---
# Expected columns: timestamp, quality, angle, distance (tab-separated)
lidar_cols = ['timestamp', 'quality', 'angle', 'distance']
lidar_data = pd.read_csv('lidar_data.txt', sep='\t', header=None, names=lidar_cols)

# --- Load IMU Data ---
# Expected format per line: <timestamp>,PX:<pitchAngle>,PY:<rollAngle>,PZ:<yawAngle>
imu_rows = []
with open('imu_data.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        # First part is timestamp in ms; convert to seconds.
        timestamp = float(parts[0]) / 1000.0
        pitch = float(parts[1].split(':')[1])
        roll = float(parts[2].split(':')[1])
        yaw = float(parts[3].split(':')[1])
        imu_rows.append([timestamp, pitch, roll, yaw])
imu_data = pd.DataFrame(imu_rows, columns=['timestamp', 'pitch', 'roll', 'yaw'])

# --- Align Timestamps ---
# For best results, both logging sessions should start at the same time.
# Here, we normalize both files to start at time zero.
lidar_data['timestamp'] -= lidar_data['timestamp'].iloc[0]
imu_data['timestamp'] -= imu_data['timestamp'].iloc[0]

# --- Synchronize Data ---
# For each lidar measurement, find the nearest IMU reading by timestamp.
def get_nearest_imu(ts, imu_df):
    diff = abs(imu_df['timestamp'] - ts)
    return imu_df.iloc[diff.idxmin()]

pitch_list = []
roll_list = []
for ts in lidar_data['timestamp']:
    imu_record = get_nearest_imu(ts, imu_data)
    pitch_list.append(imu_record['pitch'])
    roll_list.append(imu_record['roll'])
lidar_data['pitch'] = pitch_list
lidar_data['roll'] = roll_list

# --- Convert Lidar 2D Data to Cartesian Coordinates ---
# Polar-to-Cartesian conversion (in the lidar’s own 2D plane)
theta = np.deg2rad(lidar_data['angle'].values)  # Convert angle from degrees to radians
r = lidar_data['distance'].values
x_lidar = r * np.cos(theta)
y_lidar = r * np.sin(theta)
z_lidar = np.zeros_like(x_lidar)  # Initially, all points lie on a 2D plane (z=0)

# --- Transform 2D Points into 3D Using IMU Data ---
# Apply rotations based on the IMU's pitch (rotation about X) and roll (rotation about Y).
points = []
for xi, yi, zi, pitch_angle, roll_angle in zip(x_lidar, y_lidar, z_lidar, 
                                                 lidar_data['pitch'].values, 
                                                 lidar_data['roll'].values):
    # Convert pitch and roll from degrees to radians.
    pitch_rad = np.deg2rad(pitch_angle)
    roll_rad = np.deg2rad(roll_angle)
    
    # Rotation matrix for pitch (rotation about X axis)
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
                        [0, np.sin(pitch_rad), np.cos(pitch_rad)]])
    # Rotation matrix for roll (rotation about Y axis)
    R_roll = np.array([[np.cos(roll_rad), 0, np.sin(roll_rad)],
                       [0, 1, 0],
                       [-np.sin(roll_rad), 0, np.cos(roll_rad)]])
    # Combined rotation: here we apply pitch first, then roll.
    R = R_roll @ R_pitch
    point = R @ np.array([xi, yi, zi])
    points.append(point)
points = np.array(points)

# --- Visualize the 3D Point Cloud ---
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
