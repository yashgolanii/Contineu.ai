#!/usr/bin/env python3
"""
dual_logger.py          ©2025  Khyati • GPL‑3.0

Usage
-----
    python3 dual_logger.py lidar_data.txt imu_data.txt

• LiDAR   →  <lidar_data.txt>  (same columns your existing script produced)
• IMU     →  <imu_data.txt>    (timestamp,GX:‑‑,GZ:‑‑,ACC:‑‑ format)

Stop with Ctrl‑C – both threads shut down gracefully.
"""

import sys, time, threading, serial
from rplidar import RPLidar

# ----------  EDIT ONLY IF YOUR PORTS ARE DIFFERENT  ----------
PORT_LIDAR = '/dev/ttyUSB0'     # RPLiDAR USB‑to‑UART adapter
PORT_IMU   = '/dev/ttyACM0'     # Arduino CDC/USB port
BAUD_IMU   = 115200
# --------------------------------------------------------------

def lidar_worker(fname: str, stop_event: threading.Event):
    lidar = RPLidar(PORT_LIDAR)
    with open(fname, 'w') as f:
        print(f"[LiDAR] logging → {fname}")
        try:
            for m in lidar.iter_measures():
                if stop_event.is_set():
                    break
                ts = time.time()                       # epoch seconds
                # m = [new_scan?,quality,angle°,dist_mm]
                f.write(f"{ts}\t" + "\t".join(str(v) for v in m) + "\n")
        finally:
            lidar.stop();  lidar.disconnect()
            print("[LiDAR] stopped.")

def imu_worker(fname: str, stop_event: threading.Event):
    with serial.Serial(PORT_IMU, BAUD_IMU, timeout=1) as ser, open(fname, 'w') as f:
        ser.reset_input_buffer()
        print(f"[IMU ] logging → {fname}")
        while not stop_event.is_set():
            raw = ser.readline().decode(errors='ignore').strip()
            if not raw:                               # nothing / empty line
                continue
            ts = time.time()
            f.write(f"{ts},{raw}\n")
        print("[IMU ] stopped.")

def main(lidar_file: str, imu_file: str):
    stop_event = threading.Event()
    t_lidar = threading.Thread(target=lidar_worker, args=(lidar_file, stop_event), daemon=True)
    t_imu   = threading.Thread(target=imu_worker,   args=(imu_file,   stop_event), daemon=True)

    t_lidar.start();  t_imu.start()
    print("Logging LiDAR + IMU …  Press Ctrl‑C to end.\n")

    try:
        while t_lidar.is_alive() and t_imu.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Main] stopping …")
        stop_event.set()
        t_lidar.join();  t_imu.join()
    print("[Main] done.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 dual_logger.py <lidar_out.txt> <imu_out.txt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
