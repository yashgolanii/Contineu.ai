import math
import bisect
import plotly.graph_objects as go

#############################
# 1. USER CONFIGURATION
#############################

# Paths to data files
IMU_FILE   = "imu_data.txt"
LIDAR_FILE = "lidar_data.txt"

# Gyro scale factor (LSB/°/s). 
# For ±250°/s on MPU9250, often 131.0. 
# Adjust if your sensor is set to ±500, ±1000, or ±2000°/s.
GYRO_SCALE = 131.0  

# Complementary filter parameter (0.95–0.99 typical)
ALPHA = 0.98

# If your LiDAR’s plane is actually x-y with 0° = +x, 90° = +y, 
# you likely don't need to modify angle usage. If reversed, invert angles.

# If pitch is around the X-axis and yaw is around the Z-axis, 
# define the rotation order as Rz(yaw)*Rx(pitch). 
# If you suspect the real motion is reversed, you can switch the order below.

#############################
# 2. Parsing & Filtering IMU
#############################

def parse_and_fuse_imu(imu_file, alpha=ALPHA, gyro_scale=GYRO_SCALE):
    """
    Reads lines of the form:
      timestamp,GX:<gx_raw>,GZ:<gz_raw>,ACC:<ax>:<ay>:<az>
    1) Converts raw gyro values (gx_raw, gz_raw) to °/s by dividing by gyro_scale.
    2) Applies a complementary filter for pitch using accelerometer + gyro (GX).
    3) Integrates yaw from GZ.
    Returns a sorted list of (timestamp, pitch_deg, yaw_deg).
    """

    imu_records = []
    with open(imu_file, 'r') as f:
        header = f.readline()  
        prev_t = None
        pitch_deg = 0.0
        yaw_deg   = 0.0


        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            # Parse timestamp
            try:
                timestamp = float(parts[0])
            except ValueError:
                continue

            data_str = ','.join(parts[1:])

            gx_raw, gz_raw = None, None
            ax_g, ay_g, az_g = None, None, None

            for field in data_str.split(','):
                field = field.strip()
                if field.startswith("GX:"):
                    gx_raw = float(field.replace("GX:", "").strip())
                elif field.startswith("GZ:"):
                    gz_raw = float(field.replace("GZ:", "").strip())
                elif field.startswith("ACC:"):
                    try:
                        ax_str, ay_str, az_str = field.replace("ACC:", "").split(':')
                        ax_g = float(ax_str)
                        ay_g = float(ay_str)
                        az_g = float(az_str)
                    except:
                        pass

            if gx_raw is None or gz_raw is None or ax_g is None:
                continue

            gx_dps = gx_raw / gyro_scale
            gz_dps = gz_raw / gyro_scale


            dt = 0 if prev_t is None else (timestamp - prev_t)

            pitch_gyro = pitch_deg + gx_dps * dt
            yaw_gyro   = yaw_deg   + gz_dps * dt

            try:
                pitch_acc_rad = math.atan2(-ay_g, math.sqrt(ax_g**2 + az_g**2))
                pitch_acc_deg = math.degrees(pitch_acc_rad)
            except:
                pitch_acc_deg = pitch_deg  # fallback

            pitch_deg = alpha * pitch_gyro + (1 - alpha) * pitch_acc_deg


            yaw_deg = yaw_gyro

            imu_records.append((timestamp, pitch_deg, yaw_deg))
            prev_t = timestamp

    return imu_records



# Parsing LiDAR Data


def parse_lidar_data(lidar_file):
    """
    Reads lines: timestamp,quality,angle,distance
    Returns (timestamp, angle_deg, distance_mm).
    """
    records = []
    with open(lidar_file, 'r') as f:
        header = f.readline()  
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue
            try:
                ts       = float(parts[0])

                angle    = float(parts[2])
                distance = float(parts[3])
                records.append((ts, angle, distance))
            except ValueError:
                pass
    return records



#Synchronization


def find_nearest_imu(imu_records, target_ts):
    """
    Given sorted imu_records = [(timestamp, pitch, yaw), ...],
    return (pitch_deg, yaw_deg) for the closest timestamp.
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


# ransforming to 3D


def polar_to_cartesian(angle_deg, distance_mm):
    """Convert LiDAR polar to local XY (Z=0)."""
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d(x, y, pitch_deg, yaw_deg):
    """
    Apply Rz(yaw)*Rx(pitch) to (x, y, 0).
    If your actual mechanical motion is reversed, 
    you might try Rx(pitch)*Rz(yaw) or invert signs.
    """
    pitch_rad = math.radians(pitch_deg)
    yaw_rad   = math.radians(yaw_deg)

    # 1) Rx(pitch)
    x1 = x
    y1 = y * math.cos(pitch_rad)
    z1 = y * math.sin(pitch_rad)

    # 2) Rz(yaw)
    X = x1*math.cos(yaw_rad) - y1*math.sin(yaw_rad)
    Y = x1*math.sin(yaw_rad) + y1*math.cos(yaw_rad)
    Z = z1

    return X, Y, Z



# Building the 3D Cloud


def build_point_cloud(imu_records, lidar_records):
    """
    For each LiDAR record, find nearest IMU pitch & yaw, 
    convert polar->XY, then apply 3D rotation.
    Returns list of (X, Y, Z).
    """
    cloud = []
    for (ts, angle, dist) in lidar_records:
        pitch_deg, yaw_deg = find_nearest_imu(imu_records, ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d(x, y, pitch_deg, yaw_deg)
        cloud.append((X, Y, Z))
    return cloud


# Visualization


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
                    color=z_vals,  # color by Z
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





if __name__ == "__main__":
    # 1) Parse and fuse IMU data
    imu_records = parse_and_fuse_imu(IMU_FILE, alpha=ALPHA, gyro_scale=GYRO_SCALE)
    print(f"IMU records: {len(imu_records)}")

    # 2) Parse LiDAR data
    lidar_records = parse_lidar_data(LIDAR_FILE)
    print(f"LIDAR records: {len(lidar_records)}")

    # 3) Build point cloud
    cloud = build_point_cloud(imu_records, lidar_records)
    print(f"Generated {len(cloud)} points in the 3D cloud.")

    # 4) Visualize
    visualize_point_cloud(cloud)
