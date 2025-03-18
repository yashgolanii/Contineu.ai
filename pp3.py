import math
import bisect
import plotly.graph_objects as go

#############################
# 1. USER CONFIGURATION
#############################

IMU_FILE   = "imu_data2.txt"
LIDAR_FILE = "lidar_data2.txt"
ALPHA = 0.98  # Complementary filter parameter (adjust between 0.9–0.99 based on noise/drift)

#############################
# 2. Parsing & Filtering IMU
#############################

def parse_and_fuse_imu(imu_file, alpha=ALPHA):
    """
    Reads IMU data, applies complementary filter to fuse gyro and accelerometer data.
    Returns sorted list of (timestamp, pitch_deg, roll_deg, yaw_deg).
    - PITCH: rotation about y-axis (from gyro/accel)
    - ROLL: rotation about x-axis (from gyro/accel)
    - YAW: rotation about z-axis (from gyro only)
    """
    imu_records = []
    with open(imu_file, 'r') as f:
        prev_t = None
        pitch_deg = 0.0
        roll_deg = 0.0
        yaw_deg = 0.0

        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                timestamp = float(parts[0])
            except ValueError:
                continue

            fields = {field.split(':')[0]: field.split(':')[1] for field in parts[1:] if ':' in field}
            required = ['PITCH', 'ROLL', 'YAW', 'ACC_PITCH', 'ACC_ROLL']
            if not all(k in fields for k in required):
                continue

            try:
                pitch_raw = float(fields['PITCH'])    # Gyro rate (deg/s) about y-axis
                roll_raw = float(fields['ROLL'])      # Gyro rate (deg/s) about x-axis
                yaw_raw = float(fields['YAW'])        # Gyro rate (deg/s) about z-axis
                acc_pitch = float(fields['ACC_PITCH'])  # Accel-derived pitch (deg)
                acc_roll = float(fields['ACC_ROLL'])    # Accel-derived roll (deg)
            except ValueError:
                continue

            dt = 0 if prev_t is None else (timestamp - prev_t)

            # Integrate gyro values
            pitch_gyro = pitch_deg + pitch_raw * dt
            roll_gyro = roll_deg + roll_raw * dt
            yaw_gyro = yaw_deg + yaw_raw * dt

            # Complementary filter
            pitch_deg = alpha * pitch_gyro + (1 - alpha) * acc_pitch
            roll_deg = alpha * roll_gyro + (1 - alpha) * acc_roll
            yaw_deg = yaw_gyro  # Yaw from gyro only

            imu_records.append((timestamp, pitch_deg, roll_deg, yaw_deg))
            prev_t = timestamp

    return sorted(imu_records, key=lambda x: x[0])

#############################
# 3. Parsing LiDAR Data
#############################

def parse_lidar_data(lidar_file):
    """
    Reads LiDAR data: timestamp, quality, angle (deg), distance (mm).
    Returns list of (timestamp, angle_deg, distance_mm).
    """
    records = []
    with open(lidar_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
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
# 4. Synchronization with Interpolation
#############################

def find_nearest_imu(imu_records, target_ts):
    """
    Interpolates pitch, roll, yaw from IMU records at target_ts.
    Returns (pitch_deg, roll_deg, yaw_deg).
    """
    timestamps = [r[0] for r in imu_records]
    if target_ts <= timestamps[0]:
        return imu_records[0][1], imu_records[0][2], imu_records[0][3]
    if target_ts >= timestamps[-1]:
        return imu_records[-1][1], imu_records[-1][2], imu_records[-1][3]

    idx = bisect.bisect_left(timestamps, target_ts)
    t0, pitch0, roll0, yaw0 = imu_records[idx - 1]
    t1, pitch1, roll1, yaw1 = imu_records[idx]

    if t1 == t0:
        return pitch0, roll0, yaw0

    ratio = (target_ts - t0) / (t1 - t0)
    pitch = pitch0 + ratio * (pitch1 - pitch0)
    roll = roll0 + ratio * (roll1 - roll0)
    yaw = yaw0 + ratio * (yaw1 - yaw0)
    return pitch, roll, yaw

#############################
# 5. Transforming to 3D
#############################

def polar_to_cartesian(angle_deg, distance_mm):
    """Convert LiDAR polar coordinates to local XY (Z=0)."""
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d(x, y, roll_deg, pitch_deg, yaw_deg):
    """
    Apply 3D rotation to (x, y, 0) using Euler angles in order:
    1. Roll (x-axis)
    2. Pitch (y-axis)
    3. Yaw (z-axis)
    Matches standard IMU convention (ROLL=x, PITCH=y, YAW=z).
    Returns (X, Y, Z).
    """
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    # Roll (x-axis)
    x1 = x
    y1 = y * math.cos(roll)
    z1 = y * math.sin(roll)

    # Pitch (y-axis)
    x2 = x1 * math.cos(pitch) + z1 * math.sin(pitch)
    y2 = y1
    z2 = -x1 * math.sin(pitch) + z1 * math.cos(pitch)

    # Yaw (z-axis)
    X = x2 * math.cos(yaw) - y2 * math.sin(yaw)
    Y = x2 * math.sin(yaw) + y2 * math.cos(yaw)
    Z = z2

    return X, Y, Z

#############################
# 6. Building the 3D Cloud
#############################

def build_point_cloud(imu_records, lidar_records):
    """
    Builds 3D point cloud using relative timestamps for synchronization.
    Returns list of (X, Y, Z) points.
    """
    imu_ts_start = imu_records[0][0]
    lidar_ts_start = lidar_records[0][0]
    cloud = []

    for ts, angle, dist in lidar_records:
        relative_ts = ts - lidar_ts_start
        target_ts = imu_ts_start + relative_ts
        pitch_deg, roll_deg, yaw_deg = find_nearest_imu(imu_records, target_ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d(x, y, roll_deg, pitch_deg, yaw_deg)
        cloud.append((X, Y, Z))
    return cloud

#############################
# 7. Visualization
#############################

def visualize_point_cloud(cloud):
    x_vals = [p[0] for p in cloud]
    y_vals = [p[1] for p in cloud]
    z_vals = [p[2] for p in cloud]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='markers',
            marker=dict(size=2, color=z_vals, colorscale='Viridis', opacity=0.8)
        )]
    )
    fig.update_layout(
        title="3D Point Cloud with Corrected Transformations",
        scene=dict(xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

#############################
# 8. Main Execution
#############################

if __name__ == "__main__":
    # Parse data
    imu_records = parse_and_fuse_imu(IMU_FILE, alpha=ALPHA)
    lidar_records = parse_lidar_data(LIDAR_FILE)

    # Diagnostic: Check timestamp ranges
    imu_ts = [r[0] for r in imu_records]
    lidar_ts = [r[0] for r in lidar_records]
    print(f"IMU timestamps: {min(imu_ts)} to {max(imu_ts)} (duration: {max(imu_ts) - min(imu_ts):.3f} s)")
    print(f"LiDAR timestamps: {min(lidar_ts)} to {max(lidar_ts)} (duration: {max(lidar_ts) - min(lidar_ts):.3f} s)")
    print(f"Timestamp offset (LiDAR - IMU): {lidar_ts[0] - imu_ts[0]:.3f} s")
    print(f"IMU records: {len(imu_records)}, LiDAR records: {len(lidar_records)}")

    # Build and visualize point cloud
    cloud = build_point_cloud(imu_records, lidar_records)
    print(f"Generated {len(cloud)} points in the 3D cloud.")
    visualize_point_cloud(cloud)