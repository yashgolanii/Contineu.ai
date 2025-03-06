import time
import serial
import threading
from rplidar import RPLidar

# Configuration
LIDAR_PORT = '/dev/ttyUSB0'
IMU_PORT = '/dev/ttyACM0'
IMU_BAUDRATE = 9600

# Output file names
LIDAR_FILE = "lidar_data.txt"
IMU_FILE = "imu_data.txt"

# Initialize LiDAR and IMU
lidar = RPLidar(LIDAR_PORT)
imu_serial = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=1)

def lidar_logging():
    """
    Logs LiDAR data to LIDAR_FILE.
    Each measurement is written with a timestamp, quality, angle, and distance.
    """
    with open(LIDAR_FILE, "w") as f:
        # Optionally, write a header
        f.write("timestamp,quality,angle,distance\n")
        try:
            for scan in lidar.iter_scans():
                timestamp = time.time()  # Current time in seconds (float)
                for measurement in scan:
                    quality, angle, distance = measurement
                    # Write a CSV-formatted line
                    f.write(f"{timestamp},{quality},{angle},{distance}\n")
                f.flush()
        except Exception as e:
            print(f"LiDAR logging error: {e}")

def imu_logging():
    """
    Logs IMU data to IMU_FILE.
    Each line read from the IMU (e.g., "GY: <value>") is prepended with a timestamp.
    """
    with open(IMU_FILE, "w") as f:
        # Optionally, write a header
        f.write("timestamp,imu_data\n")
        try:
            while True:
                # Decode using 'replace' to handle decoding errors gracefully
                line = imu_serial.readline().decode("utf-8", errors="replace").strip()
                if line:  # Only process non-empty lines
                    timestamp = time.time()
                    f.write(f"{timestamp},{line}\n")
                    f.flush()
        except Exception as e:
            print(f"IMU logging error: {e}")

def main():
    try:
        print("Starting data logging from LiDAR and IMU...")
        # Start separate threads for LiDAR and IMU logging
        lidar_thread = threading.Thread(target=lidar_logging, daemon=True)
        imu_thread = threading.Thread(target=imu_logging, daemon=True)
        lidar_thread.start()
        imu_thread.start()

        # Keep the main thread alive while data is being logged.
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping data logging...")

    finally:
        # Safely stop and disconnect sensors
        lidar.stop()
        lidar.disconnect()
        imu_serial.close()
        print("Sensors disconnected. Logging stopped.")

if __name__ == "__main__":
    main()
