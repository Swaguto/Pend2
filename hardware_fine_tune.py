"""
Hardware Fine-Tuning Script
===========================
This script takes the pre-trained PyBullet model (`pendulum_final.zip`)
and trains it directly on the physical hardware to bridge the final Sim-to-Real gap.

WARNING: RL exploration on physical hardware is violent. The motor will make loud noises
and jerk unpredictably. Keep your hands clear of the rail!
"""

import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from Hardware_Env import HardwarePendulumEnv

def main():
    print("=" * 60)
    print("  PHYSICAL HARDWARE FINE-TUNING")
    print("=" * 60)
    print("  This will take the PyBullet model and train it ON YOUR DESK.")
    print("  Expect violent movements. Do not touch the cart while training!")
    print("\n  If the pendulum falls, you MUST manually pick it back up.")
    
    # 1. Initialize Hardware Environment
    env = HardwarePendulumEnv()
    
    # 2. Load the PyBullet-trained model
    model_path = "pendulum_final.zip"
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Could not find {model_path}!")
        print("Please ensure the PyBullet model is in this directory.")
        env.close()
        return

    print(f"\nLoading pre-trained model: {model_path}...")
    model = SAC.load(model_path, env=env)
    
    # 3. Setup Checkpoints (in case you have to emergency stop)
    os.makedirs("checkpoints_hw", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path="./checkpoints_hw/",
        name_prefix="hw_tune"
    )

    # 4. Train!
    TIMESTEPS = 10_000
    print(f"\nStarting {TIMESTEPS} steps of physical training...")
    try:
        model.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
        
        # Save the finalized model
        model.save("pendulum_hardware_tuned.zip")
        print("\nTraining Complete! Saved as pendulum_hardware_tuned.zip")
        
    except KeyboardInterrupt:
        print("\nEmergency Stop Triggered! Saving current progress...")
        model.save("pendulum_hardware_tuned_partial.zip")
        print("Saved partial progress as pendulum_hardware_tuned_partial.zip")
    finally:
        env.close()

if __name__ == "__main__":
    main()
