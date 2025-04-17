#!/usr/bin/env python3
import sys
import time
from rplidar import RPLidar

PORT_NAME = '/dev/ttyUSB0'  # Change as necessary for your system

def run(output_file):
    lidar = RPLidar(PORT_NAME)
    with open(output_file, 'w') as outfile:
        print("Recording lidar measurements... Press Ctrl+C to stop.")
        try:
            for measurement in lidar.iter_measures():
                # Each measurement is typically a list: [quality, angle (deg), distance (mm)]
                timestamp = time.time()  # Seconds since the Epoch
                line = f"{timestamp}\t" + "\t".join(str(v) for v in measurement)
                outfile.write(line + "\n")
        except KeyboardInterrupt:
            print("Stopping lidar recording.")
    lidar.stop()
    lidar.disconnect()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 lidar_logging.py <output_file>")
        sys.exit(1)
    run(sys.argv[1])
