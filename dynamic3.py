import math
import time
import serial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rplidar import RPLidar
from matplotlib.widgets import Slider, Button

# Configuration
LIDAR_PORT = '/dev/ttyUSB0'
IMU_PORT = '/dev/ttyACM0'
IMU_BAUDRATE = 9600
DMAX = 4000  # Maximum distance in mm
POINT_SIZE = 1

# Initialize LIDAR and IMU
lidar = RPLidar(LIDAR_PORT)
imu_serial = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=2)
global_map = []  # Global list of mapped 3D points

# -------------------------------
# Complementary Filter for IMU with Yaw Filtering
# -------------------------------
class IMUFilter:
    def __init__(self, alpha=0.98, yaw_threshold=0.5, yaw_filter_alpha=0.1):
        """
        alpha: complementary filter factor for tilt (accelerometer fusion)
        yaw_threshold: ignore yaw rates below this value (deg/sec)
        yaw_filter_alpha: low-pass filter factor for yaw rate (0 < alpha <= 1)
                          Lower values produce more smoothing.
        """
        self.alpha = alpha
        self.tilt_angle = 0.0  # in degrees
        self.yaw_angle = 0.0   # in degrees
        self.last_time = time.time()
        self.yaw_threshold = yaw_threshold
        self.yaw_filter_alpha = yaw_filter_alpha
        self.filtered_yaw_rate = 0.0  # initial filtered yaw rate

    def update(self, tilt_rate, yaw_rate, acc_tilt=None):
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time

        # Filter out very small yaw rates to reduce noise
        if abs(yaw_rate) < self.yaw_threshold:
            yaw_rate = 0.0

        # Apply an exponential low-pass filter to the yaw rate
        self.filtered_yaw_rate = (self.yaw_filter_alpha * yaw_rate +
                                  (1 - self.yaw_filter_alpha) * self.filtered_yaw_rate)

        # Integrate the tilt rate and filtered yaw rate
        self.tilt_angle += tilt_rate * dt
        self.yaw_angle += self.filtered_yaw_rate * dt

        # Apply complementary filter for tilt if accelerometer data is available.
        if acc_tilt is not None:
            self.tilt_angle = self.alpha * self.tilt_angle + (1 - self.alpha) * acc_tilt

        return self.tilt_angle, self.yaw_angle

# Create a global IMU filter instance with default parameters
imu_filter = IMUFilter(alpha=0.98, yaw_threshold=0.5, yaw_filter_alpha=0.1)

# -------------------------------
# Helper Functions
# -------------------------------
def polar_to_cartesian(angle_deg, distance_mm):
    """Convert polar coordinates to Cartesian (in the sensor's plane)."""
    angle_rad = math.radians(angle_deg)
    x = distance_mm * math.cos(angle_rad)
    y = distance_mm * math.sin(angle_rad)
    return x, y

def parse_imu_line(line):
    """
    Expected IMU serial format: 
      "GY: <tilt_rate>,GZ: <yaw_rate>,ACC: <acc_tilt>"
    """
    data = {}
    parts = line.split(',')
    for part in parts:
        try:
            key, value = part.split(':')
            data[key.strip()] = float(value.strip())
        except Exception as e:
            print(f"Error parsing IMU data from part '{part}': {e}")
    return data

def transform_to_3d_with_yaw(local_points, tilt_angle, yaw_angle):
    """
    Transform 2D LIDAR points (from polar conversion) into 3D points.
    1. Apply tilt rotation (assumed about the X-axis):
         [ x, y, 0 ]  -> [ x, y*cos(tilt), y*sin(tilt) ]
    2. Apply yaw rotation (about the Z-axis) to orient the scan in the global frame.
    """
    global_points = []
    tilt_rad = math.radians(tilt_angle)
    yaw_rad = math.radians(yaw_angle)
    for x, y in local_points:
        # Tilt: rotate about the X-axis
        y_tilted = y * math.cos(tilt_rad)
        z = y * math.sin(tilt_rad)
        x_tilted = x

        # Yaw: rotate about the Z-axis
        x_global = x_tilted * math.cos(yaw_rad) - y_tilted * math.sin(yaw_rad)
        y_global = x_tilted * math.sin(yaw_rad) + y_tilted * math.cos(yaw_rad)
        global_points.append((x_global, y_global, z))
    return global_points

# -------------------------------
# Mapping Loop
# -------------------------------
def update_map(lidar, ax, scatter, yaw_slider):
    global global_map
    while True:
        # Update the IMU filter's yaw threshold from the slider value.
        imu_filter.yaw_threshold = yaw_slider.val

        # Get one scan from the lidar
        for scan in lidar.iter_scans():
            local_points = []
            for _, angle, distance in scan:
                if 0 < distance <= DMAX:
                    x, y = polar_to_cartesian(angle, distance)
                    local_points.append((x, y))

            # Read one line from IMU and update the filter.
            try:
                line = imu_serial.readline().decode('utf-8').strip()
                imu_data = parse_imu_line(line)
                if 'GY' in imu_data and 'GZ' in imu_data:
                    # Use accelerometer tilt if provided; else, None.
                    acc_tilt = imu_data.get('ACC', None)
                    tilt_angle, yaw_angle = imu_filter.update(imu_data['GY'], imu_data['GZ'], acc_tilt=acc_tilt)
                else:
                    tilt_angle, yaw_angle = imu_filter.tilt_angle, imu_filter.yaw_angle
            except Exception as e:
                print(f"IMU reading error: {e}")
                tilt_angle, yaw_angle = imu_filter.tilt_angle, imu_filter.yaw_angle

            # Transform the local 2D points into 3D using the current tilt and yaw angles.
            global_points = transform_to_3d_with_yaw(local_points, tilt_angle, yaw_angle)
            global_map.extend(global_points)

            # Update the 3D scatter plot.
            if global_map:
                x_coords = [p[0] for p in global_map]
                y_coords = [p[1] for p in global_map]
                z_coords = [p[2] for p in global_map]
                scatter._offsets3d = (x_coords, y_coords, z_coords)
            plt.pause(0.01)

# -------------------------------
# UI Callback Functions
# -------------------------------
def clear_map(event):
    """Clear the accumulated 3D map."""
    global global_map
    global_map = []
    print("Map cleared.")

def update_yaw_threshold(val):
    """Callback to update the yaw threshold (the IMU filter will use this value)."""
    imu_filter.yaw_threshold = val

# -------------------------------
# Main Function
# -------------------------------
def main():
    try:
        print("Starting LIDAR and IMU...")
        print(lidar.get_info())
        status, error_code = lidar.get_health()
        print(f"LIDAR health status: {status}, Error code: {error_code}")

        # Restart LIDAR motor for consistent data
        lidar.stop()
        lidar.disconnect()
        lidar.connect()
        lidar.clean_input()
        lidar.start_motor()
        lidar.motor_speed = 60

        imu_serial.reset_input_buffer()

        # Set up 3D plot with interactive widgets
        plt.ion()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Mapping with Tilt & Yaw")
        ax.set_xlim(-DMAX, DMAX)
        ax.set_ylim(-DMAX, DMAX)
        ax.set_zlim(-DMAX, DMAX)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        scatter = ax.scatter([], [], [], s=POINT_SIZE)

        # Create a slider for yaw threshold adjustment.
        ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
        yaw_slider = Slider(ax_slider, 'Yaw Thresh (deg/s)', 0.0, 5.0, valinit=imu_filter.yaw_threshold, valstep=0.1)
        yaw_slider.on_changed(update_yaw_threshold)

        # Create a button to clear the map.
        ax_button = plt.axes([0.81, 0.02, 0.1, 0.04])
        button = Button(ax_button, 'Clear Map')
        button.on_clicked(clear_map)

        update_map(lidar, ax, scatter, yaw_slider)

    except KeyboardInterrupt:
        print("Stopping LIDAR and IMU...")

    finally:
        lidar.stop()
        lidar.disconnect()
        imu_serial.close()
        print("LIDAR and IMU stopped")

if __name__ == "__main__":
    main()
