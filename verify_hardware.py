import time
import numpy as np
from Hardware_Env import HardwarePendulumEnv

def verify():
    print("--- RoboPend Hardware Verification ---")
    port = input("Enter Arduino COM port (e.g. COM3): ")
    
    try:
        env = HardwarePendulumEnv(serial_port=port)
        print(f"\nConnected to {port} successfully.")
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    print("\n--- Testing Encoders ---")
    print("Move the cart and pendulum manually. Press Ctrl+C to stop.")
    try:
        while True:
            obs, _ = env.reset() # This gets current state
            # obs = [x, x_dot, theta_cos, theta_sin, theta_dot]
            x = obs[0]
            theta = np.arctan2(obs[3], obs[2])
            print(f"\rCart Position: {x:6.3f}m | Pendulum Angle: {np.degrees(theta):7.2f}°", end="")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nEncoder test stopped.")

    print("\n--- Testing Motor Nudge ---")
    val = input("Ready to nudge motor? (y/n): ")
    if val.lower() == 'y':
        print("Sending 0.1 torque for 0.5 seconds...")
        # Step takes action in range [-1, 1]
        env.step(np.array([0.2])) 
        time.sleep(0.5)
        env.step(np.array([0.0]))
        print("Nudge complete.")

    env.close()
    print("\nVerification Finished.")

if __name__ == "__main__":
    verify()
