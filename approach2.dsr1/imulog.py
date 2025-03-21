#!/usr/bin/env python3
import serial
import time

SERIAL_PORT = '/dev/ttyACM0'  # Adjust this to your Arduino's port
BAUD_RATE = 9600

def run(output_file):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time for Arduino to reset and start sending data
    with open(output_file, 'w') as outfile:
        print("Recording IMU data... Press Ctrl+C to stop.")
        try:
            while True:
                try:
                    # Decode with errors='ignore' to bypass any decoding issues
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except Exception as decode_err:
                    print("Decoding error:", decode_err)
                    continue

                if not line:
                    continue

                # Expected format: "timestamp,PX:<pitchAngle>,PY:<rollAngle>,PZ:<yawAngle>"
                parts = line.split(',')
                if len(parts) < 4:
                    print("Could not parse line:", line, "Error: insufficient parts")
                    continue
                try:
                    timestamp = float(parts[0])
                    pitch = float(parts[1].split(':')[1])
                    roll = float(parts[2].split(':')[1])
                    yaw = float(parts[3].split(':')[1])
                    # Write timestamp (ms) and angles separated by tabs
                    outfile.write(f"{timestamp}\t{pitch}\t{roll}\t{yaw}\n")
                except Exception as e:
                    print("Could not parse line:", line, "Error:", e)
        except KeyboardInterrupt:
            print("Stopping IMU recording.")
    ser.close()

if __name__ == '__main__':
    run('imu_data.txt')
