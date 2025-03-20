#!/usr/bin/env python3
import serial
import time
import sys

def run_imu(path):
    ser = serial.Serial('/dev/ttyACM0', 115200)  # Check your port
    with open(path, 'w') as outfile:
        try:
            print('Recording IMU... Ctrl+C to stop')
            while True:
                line = ser.readline().decode().strip()
                if line:
                    timestamp = time.time()
                    try:
                        gx, gy = map(float, line.split(','))
                        outfile.write(f"{timestamp}\t{gx}\t{gy}\n")
                    except:
                        continue
        except KeyboardInterrupt:
            ser.close()

if __name__ == '__main__':
    run_imu(sys.argv[1])