import math
import bisect
import plotly.graph_objects as go

#############################
# 1. USER CONFIGURATION
#############################

IMU_FILE   = "imu_data2.txt"
LIDAR_FILE = "lidar_data2.txt"
ALPHA = 0.98  # Complementary filter parameter (typically between 0.95 and 0.99)

#############################
# 2. Parsing & Filtering IMU
#############################

def parse_and_fuse_imu(imu_file, alpha=ALPHA):
    """
    Reads lines in the form:
      timestamp,PITCH:<pitch_raw>,ROLL:<roll_raw>,YAW:<yaw_raw>,ACC_PITCH:<acc_pitch>,ACC_ROLL:<acc_roll>
    Applies complementary filtering for both pitch and roll using gyro integration and accelerometer data.
    Yaw is integrated using gyro data only.
    Returns a sorted list of (timestamp, pitch_deg, roll_deg, yaw_deg).
    """
    imu_records = []
    with open(imu_file, 'r') as f:
        header = f.readline()  # Skip header line.
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

            data_str = ','.join(parts[1:])
            pitch_raw = None
            roll_raw = None
            yaw_raw = None
            acc_pitch = None
            acc_roll = None

            for field in data_str.split(','):
                field = field.strip()
                if field.startswith("PITCH:"):
                    try:
                        pitch_raw = float(field.replace("PITCH:", "").strip())
                    except:
                        pass
                elif field.startswith("ROLL:"):
                    try:
                        roll_raw = float(field.replace("ROLL:", "").strip())
                    except:
                        pass
                elif field.startswith("YAW:"):
                    try:
                        yaw_raw = float(field.replace("YAW:", "").strip())
                    except:
                        pass
                elif field.startswith("ACC_PITCH:"):
                    try:
                        acc_pitch = float(field.replace("ACC_PITCH:", "").strip())
                    except:
                        pass
                elif field.startswith("ACC_ROLL:"):
                    try:
                        acc_roll = float(field.replace("ACC_ROLL:", "").strip())
                    except:
                        pass

            if pitch_raw is None or roll_raw is None or yaw_raw is None:
                continue

            dt = 0 if prev_t is None else (timestamp - prev_t)

            # Integrate gyro values (values are already in deg/s from Arduino).
            pitch_gyro = pitch_deg + pitch_raw * dt
            roll_gyro  = roll_deg  + roll_raw * dt
            yaw_gyro   = yaw_deg   + yaw_raw * dt

            # Use accelerometer data if available.
            if acc_pitch is None:
                acc_pitch = pitch_deg
            if acc_roll is None:
                acc_roll = roll_deg

            # Complementary filter for pitch and roll.
            pitch_deg = alpha * pitch_gyro + (1 - alpha) * acc_pitch
            roll_deg  = alpha * roll_gyro  + (1 - alpha) * acc_roll
            yaw_deg   = yaw_gyro  # Yaw integration only

            imu_records.append((timestamp, pitch_deg, roll_deg, yaw_deg))
            prev_t = timestamp

    return imu_records

#############################
# 3. Parsing LiDAR Data
#############################

def parse_lidar_data(lidar_file):
    """
    Reads lines in the form: timestamp,quality,angle,distance
    Returns a list of (timestamp, angle_deg, distance_mm).
    """
    records = []
    with open(lidar_file, 'r') as f:
        header = f.readline()  # Skip header.
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
    returns (pitch_deg, roll_deg, yaw_deg) for the closest timestamp.
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
# 5. Transforming to 3D
#############################

def polar_to_cartesian(angle_deg, distance_mm):
    """Convert LiDAR polar coordinates to local XY (Z=0)."""
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d(x, y, pitch_deg, roll_deg, yaw_deg):
    """
    Apply a 3D rotation to the point (x, y, 0) using Euler angles.
    The applied rotations (in order) are:
      1. Rotation about the x-axis by 'pitch'
      2. Rotation about the y-axis by 'roll'
      3. Rotation about the z-axis by 'yaw'
    Returns the transformed (X, Y, Z) coordinates.
    """
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)
    yaw = math.radians(yaw_deg)

    # Rotate about x-axis (pitch).
    x1 = x
    y1 = y * math.cos(pitch)
    z1 = y * math.sin(pitch)

    # Rotate about y-axis (roll).
    x2 = x1 * math.cos(roll) + z1 * math.sin(roll)
    y2 = y1
    z2 = -x1 * math.sin(roll) + z1 * math.cos(roll)

    # Rotate about z-axis (yaw).
    X = x2 * math.cos(yaw) - y2 * math.sin(yaw)
    Y = x2 * math.sin(yaw) + y2 * math.cos(yaw)
    Z = z2

    return X, Y, Z

#############################
# 6. Building the 3D Cloud
#############################

def build_point_cloud(imu_records, lidar_records):
    """
    For each LiDAR record, find the nearest IMU (pitch, roll, yaw) reading,
    convert the LiDAR polar coordinates to Cartesian, and apply the 3D rotation.
    Returns a list of (X, Y, Z) points.
    """
    cloud = []
    for (ts, angle, dist) in lidar_records:
        pitch_deg, roll_deg, yaw_deg = find_nearest_imu(imu_records, ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d(x, y, pitch_deg, roll_deg, yaw_deg)
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
        data=[
            go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=2,
                    color=z_vals,  # Color by Z value
                    colorscale='Viridis',
                    opacity=0.8
                )
            )
        ]
    )
    fig.update_layout(
        title="3D Point Cloud with Pitch, Roll, and Yaw",
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
    imu_records = parse_and_fuse_imu(IMU_FILE, alpha=ALPHA)
    print(f"IMU records: {len(imu_records)}")

    lidar_records = parse_lidar_data(LIDAR_FILE)
    print(f"LIDAR records: {len(lidar_records)}")

    cloud = build_point_cloud(imu_records, lidar_records)
    print(f"Generated {len(cloud)} points in the 3D cloud.")

    visualize_point_cloud(cloud)
