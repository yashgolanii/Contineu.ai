import math
import bisect
import plotly.graph_objects as go

#############################
# 1. USER CONFIGURATION
#############################

# Paths to data files
IMU_FILE   = "imu_datarpi.txt"
LIDAR_FILE = "lidar_datarpi.txt"

# Gyro scale factor (LSB/°/s). 
# For ±250°/s on MPU9250, often 131.0. 
# Adjust if your sensor is set to ±500, ±1000, or ±2000°/s.
GYRO_SCALE = 131.0  

# Complementary filter parameter (0.95–0.99 typical)
ALPHA = 0.98

#############################
# 2. Parsing & Filtering IMU
#############################

def parse_and_fuse_imu(imu_file, alpha=ALPHA, gyro_scale=GYRO_SCALE):
    """
    Reads lines of the form:
      timestamp,GX:<gx_raw>,GZ:<gz_raw>,ACC:<...>
    Expected ACC field:
      - Either "ACC:<ax>:<ay>:<az>" (three values) or
      - "ACC:<value>" (one value), in which case we assume that value is ay.
    1) Converts raw gyro values to °/s.
    2) Applies a complementary filter for pitch using accelerometer + gyro (using GX for pitch).
    3) Integrates yaw from GZ.
    Returns a sorted list of (timestamp, pitch_deg, yaw_deg).
    """
    imu_records = []
    with open(imu_file, 'r') as f:
        header = f.readline()  # Skip header line.
        prev_t = None
        pitch_deg = 0.0
        yaw_deg   = 0.0

        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            try:
                timestamp = float(parts[0])
            except ValueError:
                continue

            # Join the remaining parts into a single string.
            data_str = ','.join(parts[1:])
            gx_raw, gz_raw = None, None
            ax_g, ay_g, az_g = None, None, None

            for field in data_str.split(','):
                field = field.strip()
                if field.startswith("GX:"):
                    try:
                        gx_raw = float(field.replace("GX:", "").strip())
                    except:
                        pass
                elif field.startswith("GZ:"):
                    try:
                        gz_raw = float(field.replace("GZ:", "").strip())
                    except:
                        pass
                elif field.startswith("ACC:"):
                    acc_str = field.replace("ACC:", "").strip()
                    acc_parts = acc_str.split(':')
                    if len(acc_parts) == 3:
                        try:
                            ax_g = float(acc_parts[0])
                            ay_g = float(acc_parts[1])
                            az_g = float(acc_parts[2])
                        except:
                            pass
                    elif len(acc_parts) == 1:
                        try:
                            # Fallback: assume the one ACC value represents the Y-axis
                            ay_g = float(acc_parts[0])
                            ax_g = 0.0
                            az_g = 9.81  # assume gravity if no Z measurement
                        except:
                            pass

            # Require the gyro fields; if ACC data is missing then skip complementary filtering.
            if gx_raw is None or gz_raw is None:
                continue

            # If accelerometer data is missing, then fallback to using previous pitch estimate.
            if ax_g is None or ay_g is None or az_g is None:
                pitch_acc_deg = pitch_deg
            else:
                try:
                    # Compute pitch from accelerometer.
                    # Using formula: pitch_acc = arctan2(-ay, sqrt(ax^2+az^2))
                    pitch_acc_rad = math.atan2(-ay_g, math.sqrt(ax_g**2 + az_g**2))
                    pitch_acc_deg = math.degrees(pitch_acc_rad)
                except:
                    pitch_acc_deg = pitch_deg

            # Convert raw gyro values to deg/s.
            gx_dps = gx_raw / gyro_scale
            gz_dps = gz_raw / gyro_scale

            dt = 0 if prev_t is None else (timestamp - prev_t)

            # Integrate gyro values.
            pitch_gyro = pitch_deg + gx_dps * dt
            yaw_gyro   = yaw_deg   + gz_dps * dt

            # Complementary filter for pitch.
            pitch_deg = alpha * pitch_gyro + (1 - alpha) * pitch_acc_deg

            # For yaw, use gyro integration only.
            yaw_deg = yaw_gyro

            imu_records.append((timestamp, pitch_deg, yaw_deg))
            prev_t = timestamp

    return imu_records

#############################
# 3. Parsing LiDAR Data
#############################

def parse_lidar_data(lidar_file):
    """
    Parses LiDAR data in the form:
    timestamp<TAB>invalid<TAB>quality<TAB>angle<TAB>distance

    Returns a list of (timestamp, angle_deg, distance_mm) tuples,
    filtering out invalid flags and zero distances.
    """
    records = []
    with open(lidar_file, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
            try:
                ts       = float(parts[0])
                invalid  = parts[1].strip().lower() == 'true'
                angle    = float(parts[3])
                distance = float(parts[4])

                if not invalid and distance > 0:
                    records.append((ts, angle, distance))
            except ValueError:
                continue
    return records

#############################
# 4. Synchronization
#############################

def find_nearest_imu(imu_records, target_ts):
    """
    Given sorted imu_records = [(timestamp, pitch, yaw), ...],
    returns (pitch_deg, yaw_deg) for the closest timestamp.
    """
    timestamps = [r[0] for r in imu_records]
    idx = bisect.bisect_left(timestamps, target_ts)
    if idx == 0:
        return imu_records[0][1], imu_records[0][2]
    if idx >= len(imu_records):
        return imu_records[-1][1], imu_records[-1][2]

    before = imu_records[idx - 1]
    after  = imu_records[idx]
    if target_ts - before[0] < after[0] - target_ts:
        return before[1], before[2]
    else:
        return after[1], after[2]

#############################
# 5. Transforming to 3D
#############################

def polar_to_cartesian(angle_deg, distance_mm):
    """Convert LiDAR polar coordinates to local XY (Z=0)."""
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d(x, y, pitch_deg, yaw_deg):
    """
    Apply a rotation Rz(yaw)*Rx(pitch) to point (x, y, 0).
    You can try swapping the order if necessary.
    """
    pitch_rad = math.radians(pitch_deg)
    yaw_rad   = math.radians(yaw_deg)

    # First, rotate around X-axis (pitch).
    x1 = x
    y1 = y * math.cos(pitch_rad)
    z1 = y * math.sin(pitch_rad)

    # Then, rotate around Z-axis (yaw).
    X = x1 * math.cos(yaw_rad) - y1 * math.sin(yaw_rad)
    Y = x1 * math.sin(yaw_rad) + y1 * math.cos(yaw_rad)
    Z = z1

    return X, Y, Z

#############################
# 6. Building the 3D Cloud
#############################

def build_point_cloud(imu_records, lidar_records):
    """
    For each LiDAR record, find the nearest IMU (pitch, yaw), convert the LiDAR polar coordinates to XY,
    and apply the 3D rotation. Returns a list of (X, Y, Z) points.
    """
    cloud = []
    for (ts, angle, dist) in lidar_records:
        pitch_deg, yaw_deg = find_nearest_imu(imu_records, ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d(x, y, pitch_deg, yaw_deg)
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
        title="Improved 3D Point Cloud (Pitch + Yaw)",
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
    # 1) Parse and fuse IMU data.
    imu_records = parse_and_fuse_imu(IMU_FILE, alpha=ALPHA, gyro_scale=GYRO_SCALE)
    print(f"IMU records: {len(imu_records)}")

    # 2) Parse LiDAR data.
    lidar_records = parse_lidar_data(LIDAR_FILE)
    print(f"LIDAR records: {len(lidar_records)}")

    # 3) Build point cloud.
    cloud = build_point_cloud(imu_records, lidar_records)
    print(f"Generated {len(cloud)} points in the 3D cloud.")

    # 4) Visualize.
    visualize_point_cloud(cloud)
