from rplidar import RPLidar

# Specify your LiDAR port (adjust if necessary)
LIDAR_PORT = '/dev/ttyUSB0'

def view_raw_data():
    """View raw data from the RPLidar."""
    lidar = RPLidar(LIDAR_PORT)
    try:
        print("Starting LiDAR and printing raw packets...")
        for new_scan, quality, angle, distance in lidar.iter_measures():
            # Print each raw packet's details
            print(f"New Scan: {new_scan}, Quality: {quality}, Angle: {angle}, Distance: {distance}")
    except KeyboardInterrupt:
        print("Stopping LiDAR.....")
    finally:
        lidar.stop()
        lidar.disconnect()
        print("LiDAR stopped.")

if __name__ == "__main__":
    view_raw_data()
