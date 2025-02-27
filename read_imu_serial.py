import serial

def read_imu_data(port='/dev/ttyACM0', baudrate=9600):
    """Reads IMU data from Arduino via Serial."""
    ser = serial.Serial(port, baudrate, timeout=1)
    try:
        while True:
            line = ser.readline().decode('utf-8').strip()  
            print(f"Raw data: {line}")  
            if line.startswith("Accel:"):  
                line = line.replace("Accel: ", "").strip()

            if line.count(",") == 2:  
                try:
                    ax, ay, gz = map(float, line.split(","))
                    print(f"AccelX: {ax}, AccelY: {ay}, GyroZ: {gz}")
                except ValueError:
                    print("Malformed data, skipping...")
            else:
                print("Incomplete or malformed data, skipping...")
    except KeyboardInterrupt:
        print("Stopping IMU read...")
    finally:
        ser.close()



if __name__ == "__main__":
    read_imu_data()
