import time
import serial
import threading
from rplidar import RPLidar

# Configuration
LIDAR_PORT = '/dev/ttyUSB0'
IMU_PORT = '/dev/ttyACM0'
IMU_BAUDRATE = 9600

# Output file names
LIDAR_FILE = "lidar_data2.txt"
IMU_FILE = "imu_data2.txt"

def lidar_logging():
    """
    Logs LiDAR data to LIDAR_FILE.
    Each measurement is written with a timestamp, quality, angle, and distance.
    """
    with open(LIDAR_FILE, "w") as f:
        f.write("timestamp,quality,angle,distance\n")
        try:
            for scan in lidar.iter_scans():
                timestamp = time.time()
                for measurement in scan:
                    quality, angle, distance = measurement
                    f.write(f"{timestamp},{quality},{angle},{distance}\n")
                f.flush()
        except Exception as e:
            print(f"LiDAR logging error: {e}")

def imu_logging():
    """
    Logs IMU data to IMU_FILE.
    Each line read from the IMU is prepended with a timestamp.
    Expected serial format:
      "PITCH:<value>,ROLL:<value>,YAW:<value>,ACC_PITCH:<value>,ACC_ROLL:<value>"
    """
    with open(IMU_FILE, "w") as f:
        f.write("timestamp,imu_data\n")
        try:
            while True:
                line = imu_serial.readline().decode("utf-8", errors="replace").strip()
                if line:
                    timestamp = time.time()
                    f.write(f"{timestamp},{line}\n")
                    f.flush()
        except Exception as e:
            print(f"IMU logging error: {e}")

def main():
    try:
        print("Starting data logging from LiDAR and IMU...")
        lidar_thread = threading.Thread(target=lidar_logging, daemon=True)
        imu_thread = threading.Thread(target=imu_logging, daemon=True)
        lidar_thread.start()
        imu_thread.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Stopping data logging...")
    finally:
        lidar.stop()
        lidar.disconnect()
        imu_serial.close()
        print("Sensors disconnected. Logging stopped.")

if __name__ == "__main__":
    lidar = RPLidar(LIDAR_PORT)
    imu_serial = serial.Serial(IMU_PORT, IMU_BAUDRATE, timeout=1)
    main()
