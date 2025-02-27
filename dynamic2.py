import math
import time
import serial
import threading
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rplidar import RPLidar


LIDAR_PORT = '/dev/ttyUSB0'
IMU_PORT = '/dev/ttyACM0'
IMU_BAUDRATE = 9600
DMAX = 4000  # Maximum distance in mm
POINT_SIZE = 1


lidar = RPLidar(LIDAR_PORT)
imu_serial = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=1)
global_map = []


global_pitch = 0.0
global_yaw = 0.0
imu_lock = threading.Lock()
last_imu_update = time.time()

def polar_to_cartesian(angle_deg, distance_mm):
    """Convert polar coordinates to Cartesian with LIDAR's native orientation."""
    
    adjusted_angle = angle_deg + 90 
    angle_rad = math.radians(adjusted_angle)
    x = distance_mm * math.cos(angle_rad)
    y = distance_mm * math.sin(angle_rad)
    return x, y

def read_imu_data():
    """Read IMU data with proper time delta calculation."""
    global global_pitch, global_yaw, last_imu_update
    while True:
        try:
            line = imu_serial.readline().decode('utf-8').strip()
            current_time = time.time()
            dt = current_time - last_imu_update
            last_imu_update = current_time

            if line.startswith("GY:"):  
                gy_raw = float(line.replace("GY:", "").strip())
                with imu_lock:
                    global_pitch += gy_raw * dt  
            elif line.startswith("GZ:"):  
                gz_raw = float(line.replace("GZ:", "").strip())
                with imu_lock:
                    global_yaw += gz_raw * dt
        except Exception as e:
            print(f"IMU read error: {e}")

def transform_to_3d(local_points, current_yaw, current_pitch):
    """Apply rotations with proper order: yaw (Z) then pitch (Y)."""
    global_points = []
    
    
    yaw_rad = math.radians(current_yaw)
    pitch_rad = math.radians(current_pitch)


    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    cos_pitch = math.cos(pitch_rad)
    sin_pitch = math.sin(pitch_rad)

    for x_local, y_local in local_points:
        # Apply yaw rotation (around Z-axis)
        x_rot = x_local * cos_yaw - y_local * sin_yaw
        y_rot = x_local * sin_yaw + y_local * cos_yaw
        
        # Apply pitch rotation (around Y-axis)
        x_global = x_rot * cos_pitch
        z_global = -x_rot * sin_pitch 
        y_global = y_rot  
        
        global_points.append((x_global, y_global, z_global))
    
    return global_points

def update_map(lidar, ax, scatter):
    global global_map
    scan_count = 0

    for scan in lidar.iter_scans():
        
        with imu_lock:
            current_pitch = global_pitch
            current_yaw = global_yaw

        local_points = []
        for _, angle, distance in scan:
            if 0 < distance <= DMAX:
                x, y = polar_to_cartesian(angle, distance)
                local_points.append((x, y))

        global_points = transform_to_3d(local_points, current_yaw, current_pitch)
        global_map.extend(global_points)

        
        scan_count += 1
        if scan_count % 5 == 0:  # Update every 5 scans for responsiveness
            xs = [p[0] for p in global_map]
            ys = [p[1] for p in global_map]
            zs = [p[2] for p in global_map]
            scatter._offsets3d = (xs, ys, zs)
            plt.pause(0.01)

def main():
    try:
        print("Initializing sensors...")
        lidar_status, _ = lidar.get_health()
        print(f"LIDAR status: {lidar_status}")


        imu_thread = threading.Thread(target=read_imu_data, daemon=True)
        imu_thread.start()

        plt.ion()
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Real-Time 3D Mapping")
        ax.set_xlim3d(-DMAX, DMAX)
        ax.set_ylim3d(-DMAX, DMAX)
        ax.set_zlim3d(-DMAX, DMAX)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        scatter = ax.scatter([], [], [], s=POINT_SIZE, c='b', marker='o')

        update_map(lidar, ax, scatter)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        lidar.stop()
        lidar.disconnect()
        imu_serial.close()
        plt.close()

if __name__ == "__main__":
    main()