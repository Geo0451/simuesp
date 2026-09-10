import socket
import struct
import time

DEST_IP = "127.0.0.1"
DEST_PORT = 5005
UDP_FREQ = 100
DT = 1.0 / UDP_FREQ
DATA_FILE = "data.txt"

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Streaming replayed data to {DEST_IP}:{DEST_PORT} at {UDP_FREQ}Hz...")

try:
    while True:
        with open(DATA_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                # Strip the socket header "('192.168.1.8', 4210): "
                _, payload = line.split(":", 1)
                parts = [p.strip() for p in payload.split(",")]

                if len(parts) < 8:
                    continue

                # Parse based on data.txt sequence
                timestamp = int(parts[0]) & 0xFFFFFFFF
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                gx = float(parts[4])
                gy = float(parts[5])
                gz = float(parts[6])
                touch = int(parts[7])

                # Pack using simuesp.py exact format and field order
                # sequence: timestamp, touch, ax, ay, az, gx, gy, gz
                packet = struct.pack("<I i f f f f f f", timestamp, touch, ax, ay, az, gx, gy, gz)
                
                sock.sendto(packet, (DEST_IP, DEST_PORT))
                time.sleep(DT)

except KeyboardInterrupt:
    print("\nSimulation stopped.")
