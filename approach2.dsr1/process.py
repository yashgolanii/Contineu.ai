import numpy as np
import math
import plotly.graph_objects as go

def load_data(lidar_file, imu_file):
    # Load LiDAR data
    lidar = []
    with open(lidar_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 5: continue
            lidar.append((
                float(parts[0]),         # timestamp
                parts[1] == 'True',      # new_scan
                int(parts[2]),           # quality
                float(parts[3]),         # angle (deg)
                float(parts[4])          # distance (mm)
            ))
    
    # Load IMU data
    imu = []
    with open(imu_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3: continue
            imu.append((
                float(parts[0]),         # timestamp
                float(parts[1]),         # gx (deg/s)
                float(parts[2])          # gy (deg/s)
            ))
    
    return lidar, imu

def process_data(lidar, imu):
    # Integrate IMU angles
    angles = []
    if not imu:
        return []
    pitch, roll = 0.0, 0.0
    prev_time = imu[0][0]
    for t, gx, gy in imu:
        dt = t - prev_time
        pitch += gx * dt
        roll += gy * dt
        angles.append((t, pitch, roll))
        prev_time = t
    
    # Transform LiDAR points
    points = []
    for lidar_entry in lidar:
        t_lidar = lidar_entry[0]
        
        # Find nearest IMU angles
        imu_times = [a[0] for a in angles]
        idx = np.abs(np.array(imu_times) - t_lidar).argmin()
        _, pitch, roll = angles[idx]
        
        # Convert to radians
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        
        # LiDAR polar to cartesian
        angle_deg = lidar_entry[3]
        distance = lidar_entry[4] / 1000.0  # mm to meters
        angle_rad = math.radians(angle_deg)
        x = distance * math.cos(angle_rad)
        y = distance * math.sin(angle_rad)
        z = 0
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)]
        ])
        
        Ry = np.array([
            [math.cos(roll_rad), 0, math.sin(roll_rad)],
            [0, 1, 0],
            [-math.sin(roll_rad), 0, math.cos(roll_rad)]
        ])
        
        # Apply rotations
        rotated = np.dot(Ry, np.dot(Rx, [x, y, z]))
        points.append(rotated)
    
    return np.array(points)  # Convert to NumPy array

# Main processing
lidar_data, imu_data = load_data('lidar_data.txt', 'imu_data.txt')
point_cloud = process_data(lidar_data, imu_data)

# Visualization
if len(point_cloud) > 0:  # Ensure point cloud is not empty
    fig = go.Figure(data=[go.Scatter3d(
        x=point_cloud[:, 0],  # Now works because point_cloud is a NumPy array
        y=point_cloud[:, 1],
        z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=1, opacity=0.5)
    )])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()
else:
    print("No point cloud data to visualize.")