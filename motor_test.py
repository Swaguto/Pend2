"""
motor_test.py
=============
Sends a direct velocity command to the Arduino stepper.
Run this to confirm the motor wiring works before running the RL.

Usage:
    python motor_test.py

The cart should move RIGHT for 2 seconds, stop for 1 second,
then move LEFT for 2 seconds, then stop.
"""

import serial
import time

PORT = "COM5"
BAUD = 115200

print(f"Connecting to {PORT}...")
ser = serial.Serial(PORT, BAUD, timeout=0.1)
time.sleep(2.0)
ser.reset_input_buffer()
print("Connected.\n")

def send(cmd):
    ser.write(cmd.encode())
    print(f"  Sent: {cmd.strip()}")

try:
    print("Step 1: Ramping RIGHT to 8000 steps/sec...")
    for v in range(0, 8000, 200):
        send(f"V{v}.0\n")
        time.sleep(0.01)
    time.sleep(1.0)

    print("Step 2: Ramping down to stop...")
    for v in range(8000, -1, -200):
        send(f"V{v}.0\n")
        time.sleep(0.01)
    time.sleep(0.5)

    print("Step 3: Ramping LEFT to 8000 steps/sec...")
    for v in range(0, -8000, -200):
        send(f"V{v}.0\n")
        time.sleep(0.01)
    time.sleep(1.0)

    print("Step 4: Ramping down to stop...")
    for v in range(-8000, 1, 200):
        send(f"V{v}.0\n")
        time.sleep(0.01)

    print("\nDone. Did the cart move?")
    print("  YES → Motor wiring is correct. Run: python run_inference.py")
    print("  NO  → Check EN_PIN wiring and TB6600 power supply.")

finally:
    send("V0.0\n")
    ser.close()
    print("Serial closed.")
