import math
import bisect
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation

# USER CONFIGURATION
IMU_FILE = "imu_data2.txt"
LIDAR_FILE = "lidar_data2.txt"
ALPHA = 0.98  # Complementary filter parameter
TIME_OFFSET = 355.0  # Adjust based on synchronization (e.g., IMU starts 355s earlier)

def parse_and_fuse_imu(imu_file, alpha=ALPHA):
    imu_records = []
    prev_t = None
    pitch, roll, yaw = 0.0, 0.0, 0.0

    with open(imu_file, 'r', errors='replace') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue

            try:
                timestamp = float(parts[0])
                data = {k:v for p in parts[1:] for k,v in [p.split(':') if ':' in p else ('','')]}
                
                pitch_raw = float(data.get('PITCH', 0))
                roll_raw = float(data.get('ROLL', 0))
                yaw_raw = float(data.get('YAW', 0))
                acc_pitch = float(data.get('ACC_PITCH', 0))
                acc_roll = float(data.get('ACC_ROLL', 0))
            except:
                continue

            dt = 0 if prev_t is None else (timestamp - prev_t)
            
            # Gyro integration
            pitch_gyro = pitch + pitch_raw * dt
            roll_gyro = roll + roll_raw * dt
            yaw += yaw_raw * dt

            # Complementary filter
            pitch = alpha * pitch_gyro + (1 - alpha) * acc_pitch
            roll = alpha * roll_gyro + (1 - alpha) * acc_roll

            imu_records.append((timestamp, pitch, roll, yaw))
            prev_t = timestamp

    return imu_records

def parse_lidar_data(lidar_file):
    records = []
    with open(lidar_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            try:
                ts = float(parts[0]) + TIME_OFFSET  # Apply time offset
                angle = float(parts[2])
                dist = float(parts[3])
                records.append((ts, angle, dist))
            except:
                pass
    return records

def find_interpolated_imu(imu_records, target_ts):
    timestamps = [r[0] for r in imu_records]
    idx = bisect.bisect_left(timestamps, target_ts)
    
    if idx == 0 or idx >= len(timestamps):
        return imu_records[idx][1:] if idx < len(imu_records) else imu_records[-1][1:]

    before = imu_records[idx-1]
    after = imu_records[idx]
    ratio = (target_ts - before[0]) / (after[0] - before[0])
    
    # Linear interpolation for angles
    pitch = before[1] + ratio*(after[1]-before[1])
    roll = before[2] + ratio*(after[2]-before[2])
    yaw = before[3] + ratio*(after[3]-before[3])
    
    return (pitch, roll, yaw)

def polar_to_cartesian(angle_deg, distance_mm):
    rad = math.radians(angle_deg)
    x = distance_mm * math.cos(rad)
    y = distance_mm * math.sin(rad)
    return x, y

def rotate_3d(x, y, pitch, roll, yaw):
    # Convert Euler angles to rotation matrix (Z-Y-X order)
    r = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    rotated = r.apply([x, y, 0])
    return rotated

def build_point_cloud(imu_records, lidar_records):
    cloud = []
    for ts, angle, dist in lidar_records:
        pitch, roll, yaw = find_interpolated_imu(imu_records, ts)
        x, y = polar_to_cartesian(angle, dist)
        X, Y, Z = rotate_3d(x, y, pitch, roll, yaw)
        cloud.append((X, Y, Z))
    return cloud

# Visualization remains the same as before

if __name__ == "__main__":
    imu_records = parse_and_fuse_imu(IMU_FILE)
    lidar_records = parse_lidar_data(LIDAR_FILE)
    
    print(f"IMU records: {len(imu_records)}, LIDAR: {len(lidar_records)}")
    
    cloud = build_point_cloud(imu_records, lidar_records)
    visualize_point_cloud(cloud)