import socket
import struct
import time
import math
import random

DEST_IP = "127.0.0.1"
DEST_PORT = 5005
UDP_FREQ = 100
DT = 1.0 / UDP_FREQ

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ---------------------------------------------------------------------
# Realistic Sensor Imperfections (Based on MPU6050 Hardware Specs)
# ---------------------------------------------------------------------
GYRO_NOISE_STD     = 0.45    # Deg/s white noise standard deviation
ACCEL_NOISE_STD    = 0.02    # G-force noise standard deviation
GYRO_BIAS_DRIFT    = 0.08    # Deg/s gradual thermal zero-rate drift

# Persistent hardware bias state (simulating uncalibrated offset)
gyro_bias_x = 2.5   # +2.5 deg/s constant offset
gyro_bias_y = -1.8  # -1.8 deg/s constant offset
gyro_bias_z = 0.9   # +0.9 deg/s constant offset

start_time = time.time()
print(f"Streaming REALISTIC noisy IMU to {DEST_IP}:{DEST_PORT} at {UDP_FREQ}Hz...")

try:
    while True:
        now = time.time()
        elapsed = now - start_time
        
        # 1. Ground Truth Physics (Simulating continuous 3D air strokes)
        true_pitch_rate = math.cos(elapsed * 2.5) * 60.0  # Deg/s pitch
        true_yaw_rate   = math.sin(elapsed * 1.8) * 40.0  # Deg/s yaw
        true_roll_rate  = math.sin(elapsed * 0.9) * 25.0  # Deg/s roll

        # Simulated absolute orientation angles (for calculating true gravity)
        pitch_rad = math.sin(elapsed * 2.5) * 0.4
        roll_rad  = math.cos(elapsed * 0.9) * 0.2

        # 2. Real-world Accelerometer Dynamics (Gravity + Linear Motion + Centripetal Force)
        true_ax = -math.sin(pitch_rad) + (math.cos(elapsed * 2.5) * 0.15)
        true_ay =  math.sin(roll_rad) * math.cos(pitch_rad) + (math.sin(elapsed * 1.8) * 0.15)
        true_az =  math.cos(roll_rad) * math.cos(pitch_rad)

        # 3. Add Sensor Imperfections
        # Slow random thermal drift over time
        gyro_bias_x += random.gauss(0, GYRO_BIAS_DRIFT * DT)
        gyro_bias_y += random.gauss(0, GYRO_BIAS_DRIFT * DT)
        gyro_bias_z += random.gauss(0, GYRO_BIAS_DRIFT * DT)

        # Corrupt Gyroscope measurements (True signal + Bias + White Noise)
        gx = true_pitch_rate + gyro_bias_x + random.gauss(0, GYRO_NOISE_STD)
        gy = true_yaw_rate   + gyro_bias_y + random.gauss(0, GYRO_NOISE_STD)
        gz = true_roll_rate  + gyro_bias_z + random.gauss(0, GYRO_NOISE_STD)

        # Corrupt Accelerometer measurements (True signal + White Noise)
        ax = true_ax + random.gauss(0, ACCEL_NOISE_STD)
        ay = true_ay + random.gauss(0, ACCEL_NOISE_STD)
        az = true_az + random.gauss(0, ACCEL_NOISE_STD)

        timestamp = int(elapsed * 1_000_000) & 0xFFFFFFFF
        touch = 0

        # Pack exact 32-byte struct layout
        packet = struct.pack("<I i f f f f f f", timestamp, touch, ax, ay, az, gx, gy, gz)
        
        sock.sendto(packet, (DEST_IP, DEST_PORT))
        time.sleep(DT)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
