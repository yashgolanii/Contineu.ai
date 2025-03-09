import math
import bisect
import plotly.graph_objects as go

#############################
# 1. USER CONFIGURATION
#############################

# Paths to data files (adjust paths as needed)
IMU_FILE   = "imu_data.txt"
LIDAR_FILE = "lidar_data.txt"

#############################
# 2. Parsing IMU Data
#############################

def parse_imu_data(imu_file):
    """
    Reads IMU data lines of the form:
      timestamp,PX:<pitch_angle>,PY:<roll_angle>,PZ:<yaw_angle>
    Returns a sorted list of tuples:
      (timestamp, pitch_deg, roll_deg, yaw_deg)
    """
    imu_records = []
    with open(imu_file, 'r') as f:
        header = f.readline()  # skip header if any
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                timestamp = float(parts[0])
            except ValueError:
                continue

            pitch_val = None
            roll_val = None
            yaw_val = None
            for field in parts[1:]:
                field = field.strip()
                if field.startswith("PX:"):
                    try:
                        pitch_val = float(field.replace("PX:", "").strip())
                    except:
                        pass
                elif field.startswith("PY:"):
                    try:
                        roll_val = float(field.replace("PY:", "").strip())
                    except:
                        pass
                elif field.startswith("PZ:"):
                    try:
                        yaw_val = float(field.replace("PZ:", "").strip())
                    except:
                        pass
            if pitch_val is None or roll_val is None or yaw_val is None:
                continue
            imu_records.append((timestamp, pitch_val, roll_val, yaw_val))
    imu_records.sort(key=lambda r: r[0])
    return imu_records

#############################
# 3. Parsing LiDAR Data
#############################

def parse_lidar_data(lidar_file):
    """
    Reads LiDAR data lines of the form:
      timestamp,quality,angle,distance
    Returns a list of tuples: (timestamp, angle_deg, distance_mm)
    """
    records = []
    with open(lidar_file, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                ts = float(parts[0])
                angle = float(parts[2])
                distance = float(parts[3])
                records.append((ts, angle, distance))
            except ValueError:
                pass
    return records

#############################
# 4. Synchronization
#############################

def find_nearest_imu(imu_records, target_ts):
    """
    Given sorted imu_records = [(timestamp, pitch, roll, yaw), ...],
    returns (pitch_deg, roll_deg, yaw_deg) for the record closest to target_ts.
    """
    timestamps = [r[0] for r in imu_records]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return imu_records[0][1], imu_records[0][2], imu_records[0][3]
    if idx >= len(imu_records):
        return imu_records[-1][1], imu_records[-1][2], imu_records[-1][3]
    before = imu_records[idx - 1]
    after  = imu_records[idx]
    if target_ts - before[0] < after[0] - target_ts:
        return before[1], before[2], before[3]
    else:
        return after[1], after[2], after[3]

#############################
# 5. 3D Transformation
#############################

def polar_to_cartesian(angle_deg, distance_mm):
    """
    Convert LiDAR polar coordinates to 2D Cartesian coordinates (assuming Z=0).
    """
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d_all(x, y, pitch_deg, roll_deg, yaw_deg):
    """
    Apply the composite rotation:
      R = Rz(yaw) * Ry(roll) * Rx(pitch)
    to the point (x, y, 0).
    
    Rotations are applied in this order:
      1. Rotate about X by pitch_deg.
      2. Rotate about Y by roll_deg.
      3. Rotate about Z by yaw_deg.
    
    Returns (X, Y, Z).
    """
    # Convert degrees to radians.
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)
    yawr = math.radians(yaw_deg)
    
    # Compute sines and cosines.
    cp = math.cos(p); sp = math.sin(p)
    cr = math.cos(r); sr = math.sin(r)
    cy = math.cos(yawr); sy = math.sin(yawr)
    
    # The composite rotation matrix R = Rz * Ry * Rx.
    # Applied to vector [x, y, 0]:
    # X = (cy * cp) * x + (-sy * cr + cy * sp * sr) * y
    # Y = (sy * cp) * x + (cy * cr + sy * sp * sr) * y
    # Z = (-sp) * x + (cp * sr) * y
    X = (cy * cp) * x + (-sy * cr + cy * sp * sr) * y
    Y = (sy * cp) * x + (cy * cr + sy * sp * sr) * y
    Z = (-sp) * x + (cp * sr) * y
    return X, Y, Z

#############################
# 6. Building the 3D Point Cloud
#############################

def build_point_cloud(imu_records, lidar_records):
    """
    For each LiDAR record, find the nearest IMU record (pitch, roll, yaw),
    convert LiDAR polar coordinates to XY, and apply the composite 3D rotation.
    Returns a list of (X, Y, Z) points.
    """
    cloud = []
    for (ts, angle, dist) in lidar_records:
        pitch_deg, roll_deg, yaw_deg = find_nearest_imu(imu_records, ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d_all(x, y, pitch_deg, roll_deg, yaw_deg)
        cloud.append((X, Y, Z))
    return cloud

#############################
# 7. Visualization
#############################

def visualize_point_cloud(cloud):
    x_vals = [pt[0] for pt in cloud]
    y_vals = [pt[1] for pt in cloud]
    z_vals = [pt[2] for pt in cloud]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=2,
                    color=z_vals,  # Color points by Z value for visual effect
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ]
    )
    fig.update_layout(
        title="3D Point Cloud (X=Pitch, Y=Roll, Z=Yaw)",
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

#############################
# 8. Main Execution
#############################

if __name__ == "__main__":
    # 1) Parse IMU data.
    imu_records = parse_imu_data(IMU_FILE)
    print(f"IMU records: {len(imu_records)}")

    # 2) Parse LiDAR data.
    lidar_records = parse_lidar_data(LIDAR_FILE)
    print(f"LIDAR records: {len(lidar_records)}")

    # 3) Build the 3D point cloud.
    cloud = build_point_cloud(imu_records, lidar_records)
    print(f"Generated {len(cloud)} points in the 3D cloud.")

    # 4) Visualize the point cloud.
    visualize_point_cloud(cloud)
