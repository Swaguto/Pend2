import time
import numpy as np
from stable_baselines3 import SAC
from Hardware_Env import HardwarePendulumEnv

def run_inference():
    MODEL_PATH = "pendulum_final.zip"
    SERIAL_PORT = "COM5"  # CHANGE THIS to your port
    
    print(f"Loading model: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
    
    print(f"Connecting to hardware on {SERIAL_PORT}...")
    try:
        env = HardwarePendulumEnv(port=SERIAL_PORT)
    except Exception as e:
        print(f"Error: {e}")
        return

    print("\n--- RL CONTROL ACTIVE ---")
    print("Keep hand on the power switch!")
    
    obs, _ = env.reset()
    
    try:
        while True:
            # Get action from the brain
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply to hardware
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("\nBoundary hit or episode ended. Resetting...")
                env.step(np.array([0.0])) # Stop motor
                time.sleep(1.0)
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.step(np.array([0.0])) # Safety stop
        env.close()

if __name__ == "__main__":
    run_inference()
