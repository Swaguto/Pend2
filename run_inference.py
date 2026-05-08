import time
import numpy as np
from stable_baselines3 import SAC
from Hardware_Env import HardwarePendulumEnv

def run_inference():
    BASE_MODEL   = "pendulum_final.zip"       # clean architecture
    HW_MODEL     = "pendulum_hardware_tuned.zip"  # fine-tuned weights
    SERIAL_PORT  = "COM3"

    # Load the clean base model first (avoids ent_coef corruption crash)
    print(f"Loading base model: {BASE_MODEL}")
    model = SAC.load(BASE_MODEL)

    # If a hardware-tuned model exists, transplant its policy weights
    import os, zipfile, torch, io
    if os.path.exists(HW_MODEL):
        print(f"Applying hardware-tuned weights from: {HW_MODEL}")
        try:
            with zipfile.ZipFile(HW_MODEL, 'r') as zf:
                with zf.open('policy.pth') as f:
                    weights = torch.load(io.BytesIO(f.read()), map_location='cpu')
            model.policy.load_state_dict(weights, strict=False)
            print("  Hardware weights applied OK!")
        except Exception as e:
            print(f"  Could not apply hardware weights ({e}) — using base model.")
    else:
        print(f"  No hardware model found — using base model.")
    
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
