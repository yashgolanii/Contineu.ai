#!/usr/bin/env python3
import serial
import time

SERIAL_PORT = '/dev/ttyACM0'  # Adjust this to your Arduino's port
BAUD_RATE = 9600

def run(output_file):
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give time for Arduino to reset and start sending data
    with open(output_file, 'w') as outfile:
        print("Recording IMU data... Press Ctrl+C to stop.")
        try:
            while True:
                line = ser.readline().decode('utf-8').strip()
                if line:
                    # Example expected format: "gx: 131, gy: 65" (raw values)
                    try:
                        parts = line.split(',')
                        gx_raw = float(parts[0].split(':')[1])
                        gy_raw = float(parts[1].split(':')[1])
                        # Convert raw values to degrees per second (or another unit) by dividing by sensitivity factor
                        gx = gx_raw / 131.0
                        gy = gy_raw / 131.0
                        timestamp = time.time()
                        # Write the timestamp and the computed values
                        outfile.write(f"{timestamp}\t{gx}\t{gy}\n")
                    except Exception as e:
                        print("Could not parse line:", line, "Error:", e)
        except KeyboardInterrupt:
            print("Stopping IMU recording.")
    ser.close()

if __name__ == '__main__':
    run('imu_data.txt')
