"""
Hardware Fine-Tuning Script  — Upright-Start Mode
===================================================
This script fine-tunes the pre-trained SAC policy ONLY in the near-upright
zone, so the model learns to actively hold the pendulum there on the real
hardware.

HOW IT WORKS
------------
1. Auto-homes the cart to centre.
2. You hold the pendulum manually near vertical.
3. Press Enter — the RL loop takes over immediately.
4. If the pendulum falls, it stops the motor and waits for you to lift it
   back up, then resumes from the upright position again.
5. After TOTAL_TIMESTEPS steps, the fine-tuned model is saved.

WHY THIS IS SAFE
----------------
- ent_coef is fixed at a very small value (0.001) so the policy stays
  nearly deterministic — no violent random thrashing.
- Exploration noise is tiny because we're only adjusting fine corrections
  that the pre-trained policy already almost gets right.
- The cart stops automatically the moment the pendulum falls below 60° from
  vertical (cos_theta < 0.5), so it won't shoot into the rail.
"""

import os
import time
import numpy as np
from stable_baselines3 import SAC
from Hardware_Env import HardwarePendulumEnv, ACCEL_SCALE, MAX_SPEED, BALANCE_BOOST

# ── Tuning ─────────────────────────────────────────────────────────────────────
TOTAL_TIMESTEPS     = 5_000      # total env steps to collect & learn from
GRADIENT_STEPS      = 4          # SAC gradient updates per env step (aggressive)
LEARNING_RATE       = 3e-5       # very small — we're fine-tuning, not retraining
# NOTE: ent_coef is intentionally NOT overridden here — setting it to a float
# corrupts the saved model and causes a crash on reload. The base model's
# entropy coefficient is already low enough after 23M steps of training.
FALL_THRESHOLD      = 0.50       # cos_theta below this → pendulum has fallen (≈ 60°)
SAVE_EVERY          = 500        # save a checkpoint every N steps
# ──────────────────────────────────────────────────────────────────────────────


class UprightStartEnv(HardwarePendulumEnv):
    """
    Subclass of HardwarePendulumEnv that:
      - Only runs episodes while the pendulum is near vertical.
      - Terminates the episode immediately when it falls past FALL_THRESHOLD.
      - On reset(), waits for you to lift the pendulum back to vertical
        before handing control back to the agent.
    """

    def reset(self, seed=None, options=None):
        # Call parent to home (first call) or re-centre (subsequent calls)
        obs, info = super().reset(seed=seed, options=options)

        # Now override: wait until the user lifts the pendulum to near-upright
        print("\n  Lift the pendulum to VERTICAL and hold it there.")
        print("  Press Enter when it is near the top ...")
        input()

        # Drain stale serial data accumulated while waiting
        self.ser.reset_input_buffer()
        self._speed = 0.0
        pend, cart = self._read_latest()
        self._prev_pend = pend
        self._prev_cart = cart
        self._prev_t    = time.perf_counter()
        obs = self._build_obs(pend, cart)

        print(f"  cos(θ) = {obs[2]:+.3f}  (should be ≥ +0.85 for good upright start)")
        if obs[2] < 0.5:
            print("  [WARNING] Pendulum looks far from vertical — try lifting it higher!")
        print("  RL is now balancing. Let go!\n")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        # Terminate early if the pendulum has fallen
        cos_theta = float(obs[2])
        if cos_theta < FALL_THRESHOLD:
            terminated = True
            self._speed = 0.0
            self._send("V0.0\n")

        return obs, reward, terminated, truncated, info


def main():
    print("=" * 62)
    print("  HARDWARE FINE-TUNING — UPRIGHT START MODE")
    print("=" * 62)
    print(f"  ACCEL_SCALE   = {ACCEL_SCALE:,.0f}")
    print(f"  MAX_SPEED     = {MAX_SPEED:,.0f}")
    print(f"  BALANCE_BOOST = {BALANCE_BOOST}")
    print(f"  ENT_COEF      = {ENT_COEF}  (near-deterministic)")
    print(f"  Total steps   = {TOTAL_TIMESTEPS}")
    print()

    # 1. Upright-start environment
    env = UprightStartEnv()

    # 2. Load pre-trained model
    model_path = "pendulum_final.zip"
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Could not find {model_path}!")
        env.close()
        return

    print(f"Loading pre-trained model: {model_path} ...")
    model = SAC.load(model_path, env=env)

    # Override hyper-params for gentle hardware fine-tuning
    # (must be set AFTER load, not via custom_objects)
    model.learning_rate  = LEARNING_RATE
    model.gradient_steps = GRADIENT_STEPS
    # ent_coef is intentionally left unchanged — overriding it corrupts save/load
    print(f"  learning_rate={LEARNING_RATE},  gradient_steps={GRADIENT_STEPS}")

    # 3. Run the fine-tuning loop
    os.makedirs("checkpoints_hw", exist_ok=True)
    total_steps = 0
    episode     = 0

    print("\nStarting fine-tuning loop.  Ctrl-C to stop and save.\n")
    try:
        obs, _ = env.reset()

        while total_steps < TOTAL_TIMESTEPS:
            action, _ = model.predict(obs, deterministic=False)  # tiny noise
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1

            # Log every 50 steps
            if total_steps % 50 == 0:
                cos_t = float(obs[2])
                print(f"  step {total_steps:5d}/{TOTAL_TIMESTEPS}  |  "
                      f"cos(θ)={cos_t:+.3f}  reward={reward:+.2f}")

            # Perform gradient updates
            if model.replay_buffer.size() >= model.batch_size:
                model.train(gradient_steps=GRADIENT_STEPS, batch_size=model.batch_size)

            # Checkpoint
            if total_steps % SAVE_EVERY == 0:
                ckpt = f"checkpoints_hw/hw_tune_{total_steps}.zip"
                model.save(ckpt)
                print(f"  [Checkpoint saved: {ckpt}]")

            if terminated or truncated:
                episode += 1
                print(f"\n  Episode {episode} ended at step {total_steps}. "
                      f"Pendulum fell — pick it back up!\n")
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n  Emergency stop! Saving...")

    # 4. Save final model
    model.save("pendulum_hardware_tuned.zip")
    print(f"\nDone! Saved fine-tuned model as pendulum_hardware_tuned.zip")
    print(f"Run with:  py run_inference.py   (and update MODEL_PATH to pendulum_hardware_tuned.zip)")
    env.close()


if __name__ == "__main__":
    main()
