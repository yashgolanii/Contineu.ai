#!/usr/bin/env python3
import sys
import time
from rplidar import RPLidar

PORT_NAME = '/dev/ttyUSB0'

def run(output_file):
    lidar = RPLidar(PORT_NAME)
    with open(output_file, 'w') as outfile:
        try:
            print("Recording lidar measurements... Press Ctrl+C to stop.")
            for measurement in lidar.iter_measures():
                # Prepend a timestamp (seconds since epoch)
                timestamp = time.time()
                # measurement is usually [quality, angle, distance] (check your documentation)
                line = f"{timestamp}\t" + "\t".join(str(v) for v in measurement)
                outfile.write(line + "\n")
        except KeyboardInterrupt:
            print("Stopping lidar recording.")
    lidar.stop()
    lidar.disconnect()

if __name__ == '__main__':
    run(sys.argv[1])
