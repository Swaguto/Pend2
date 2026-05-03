"""
demo_best.py — loads the best saved checkpoint and renders it in PyBullet.

Usage:
    python demo_best.py                          # auto-finds best model
    python demo_best.py pendulum_final           # specific file
    python demo_best.py checkpoints/best/best_model
"""

import sys
import io
import time
import os
import numpy as np

# Force UTF-8 so unicode chars never crash on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import SAC


class PendulumEnv(gym.Env):
    # ── Must match training exactly ───────────────────────────────────────────
    CART_LOWER  = -0.325
    CART_UPPER  =  0.215
    CART_CENTRE = (CART_LOWER + CART_UPPER) / 2.0
    CART_RANGE  = (CART_UPPER - CART_LOWER) / 2.0
    CART_MARGIN =  0.02
    MAX_FORCE   = 20.0
    MAX_STEPS   = 1000
    SIM_HZ      = 480
    CTRL_SKIP   = 8

    # ── Confirmed joint indices ───────────────────────────────────────────────
    CART_IDX = 3   # PRISMATIC  dof_cart_slider_0
    PEND_IDX = 7   # REVOLUTE   dof_pendulum_pivot_0

    def __init__(self, render: bool = True):
        super().__init__()
        self._cid = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self._cid)
        p.setRealTimeSimulation(0, physicsClientId=self._cid)
        p.setPhysicsEngineParameter(enableFileCaching=0,
                                    physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.SIM_HZ, physicsClientId=self._cid)

        # Side-on camera pointing at robot (z=1.5)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=90, cameraPitch=-15,
            cameraTargetPosition=[0, 0, 1.5],
            physicsClientId=self._cid)

        self.cart_idx  = self.CART_IDX
        self.pend_idx  = self.PEND_IDX
        self.urdf_path = "Pend_assem_stl/pend_assem/urdf/pend_assem.urdf"
        self.robot     = None
        self._step_count = 0

        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (5,), dtype=np.float32)

        self._build_world()

    def _build_world(self):
        p.resetSimulation(physicsClientId=self._cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.SIM_HZ, physicsClientId=self._cid)
        p.setRealTimeSimulation(0, physicsClientId=self._cid)
        p.loadURDF("plane.urdf", physicsClientId=self._cid)
        self.robot = p.loadURDF(
            self.urdf_path, basePosition=[0, 0, 1.5],
            useFixedBase=True, physicsClientId=self._cid)
        self._disable_motors()

    def _disable_motors(self):
        for joint in (self.cart_idx, self.pend_idx):
            p.setJointMotorControl2(self.robot, joint,
                                    p.VELOCITY_CONTROL, force=0,
                                    physicsClientId=self._cid)

    def reset(self, seed=None, options=None):
        self._step_count = 0
        rng = np.random.default_rng(seed)   # reproducible from seed
        p.resetJointState(
            self.robot, self.cart_idx,
            targetValue=self.CART_CENTRE + rng.uniform(-0.03, 0.03),
            targetVelocity=0.0, physicsClientId=self._cid)
        # Start hanging (theta~0)
        p.resetJointState(
            self.robot, self.pend_idx,
            targetValue=rng.uniform(-0.15, 0.15),
            targetVelocity=rng.uniform(-0.3, 0.3),
            physicsClientId=self._cid)
        self._disable_motors()
        return self._get_obs(), {}

    def _get_obs(self):
        states     = p.getJointStates(self.robot,
                                      [self.cart_idx, self.pend_idx],
                                      physicsClientId=self._cid)
        cart_pos   = states[0][0]
        cart_vel   = states[0][1]
        pole_angle = states[1][0]
        pole_vel   = states[1][1]
        shifted    = pole_angle - np.pi   # shift: 0=hanging → -π, upright → 0
        return np.array([
            (cart_pos - self.CART_CENTRE) / self.CART_RANGE,
            cart_vel  / 2.0,
            np.cos(shifted),   # +1 upright, -1 hanging
            np.sin(shifted),
            pole_vel  / 10.0,
        ], dtype=np.float32)

    def step(self, action):
        self._step_count += 1
        cart_force = float(action[0]) * self.MAX_FORCE
        p.setJointMotorControl2(self.robot, self.cart_idx,
                                p.TORQUE_CONTROL, force=cart_force,
                                physicsClientId=self._cid)
        for _ in range(self.CTRL_SKIP):
            p.stepSimulation(physicsClientId=self._cid)
        obs = self._get_obs()
        cart_pos_norm, _, cos_theta, sin_theta, pole_vel = obs
        actual_cart_pos = cart_pos_norm * self.CART_RANGE + self.CART_CENTRE
        angle_from_top  = abs(np.arctan2(sin_theta, cos_theta))
        reward = (cos_theta
                  + (3.0 * (1.0 - angle_from_top / 0.3)
                     if angle_from_top < 0.3 else 0.0)
                  + (-0.2 * pole_vel ** 2 if angle_from_top < 0.4 else 0.0)
                  - 1.5 * cart_pos_norm ** 2
                  - 0.001 * (cart_force / self.MAX_FORCE) ** 2)
        terminated = bool(
            actual_cart_pos < self.CART_LOWER + self.CART_MARGIN or
            actual_cart_pos > self.CART_UPPER - self.CART_MARGIN)
        if terminated:
            reward -= 100.0  # Huge penalty for hitting the rail to stop suicide
        truncated = bool(self._step_count >= self.MAX_STEPS)
        return obs, reward, terminated, truncated, {}

    def get_angle_deg(self):
        """Return actual angle from upright in degrees for display."""
        states = p.getJointStates(self.robot, [self.pend_idx],
                                  physicsClientId=self._cid)
        pole_angle = states[0][0]
        # Distance from π (upright) in degrees
        return abs(np.degrees(pole_angle - np.pi))

    def close(self):
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None


# ── Find best model ───────────────────────────────────────────────────────────

def find_model(arg):
    if arg:
        path = arg if arg.endswith(".zip") else arg + ".zip"
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Model not found: {path}")

    for candidate in ["pendulum_final.zip", "checkpoints/best/best_model.zip"]:
        if os.path.exists(candidate):
            return candidate

    ckpt_dir = "checkpoints"
    if os.path.isdir(ckpt_dir):
        zips = sorted(
            [f for f in os.listdir(ckpt_dir) if f.endswith(".zip")],
            key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
            reverse=True,
        )
        if zips:
            return os.path.join(ckpt_dir, zips[0])

    raise FileNotFoundError(
        "No model found. Train first:\n"
        "  python pendulum_ludicrous_speed.py\n"
        "Expected: checkpoints/best/best_model.zip or pendulum_final.zip"
    )


# ── Demo loop ─────────────────────────────────────────────────────────────────

N_EVAL = 100   # silent episodes to screen before picking the best


def evaluate_silent(model, n=N_EVAL):
    """Run n episodes headless; return (best_seed, best_reward, best_steps)."""
    env = PendulumEnv(render=False)
    best_seed   = 0
    best_reward = -np.inf
    best_steps  = 0

    print(f"  Screening {n} episodes silently to find the best...")
    for i in range(n):
        seed = i
        obs, _ = env.reset(seed=seed)
        cum_rew = 0.0
        steps   = 0
        done    = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            cum_rew += reward
            steps   += 1
            done = terminated or truncated

        flag = "  <- BEST" if cum_rew > best_reward else ""
        status = "BALANCED [OK]" if not terminated else "HIT RAIL  [X]"
        print(f"    seed={seed:3d}  steps={steps:4d}  reward={cum_rew:8.1f}  {status}{flag}")

        if cum_rew > best_reward:
            best_reward = cum_rew
            best_seed   = seed
            best_steps  = steps

    env.close()
    return best_seed, best_reward, best_steps


def replay_best(model, best_seed, best_reward):
    """Replay the best episode in the GUI at real-time speed."""
    print(f"\n  Replaying best episode (seed={best_seed}, reward={best_reward:.1f}) in GUI...")
    print("  Press Ctrl-C to quit.\n")

    env = PendulumEnv(render=True)
    obs, _ = env.reset(seed=best_seed)

    txt = p.addUserDebugText("Best episode replay", [0, 0, 2.3],
                             textColorRGB=[1, 1, 0], textSize=1.5,
                             physicsClientId=env._cid)
    try:
        while True:   # loop the best episode forever
            obs, _ = env.reset(seed=best_seed)
            steps   = 0
            cum_rew = 0.0
            done    = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                cum_rew += reward
                steps   += 1
                done = terminated or truncated

                angle_deg = env.get_angle_deg()
                near_top  = angle_deg < 10
                overlay   = (f"BEST EPISODE (seed={best_seed})  |  "
                             f"Step {steps}  |  "
                             f"Angle from top: {angle_deg:.1f} deg")
                txt = p.addUserDebugText(
                    overlay, [0, 0, 2.3],
                    textColorRGB=[0.2, 1.0, 0.2] if near_top else [1.0, 1.0, 0.2],
                    textSize=1.5,
                    replaceItemUniqueId=txt,
                    physicsClientId=env._cid)

                time.sleep(1.0 / 60.0)

            status = "BALANCED [OK]" if not terminated else "HIT RAIL [X]"
            print(f"  Ep done — steps={steps}  reward={cum_rew:.1f}  {status}  (looping...)")

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        env.close()


def run_demo(model_path):
    print(f"\n  Loading: {model_path}")
    model = SAC.load(model_path)
    print("  Loaded.\n")

    best_seed, best_reward, best_steps = evaluate_silent(model)
    print(f"\n  Best: seed={best_seed}  steps={best_steps}  reward={best_reward:.1f}\n")
    replay_best(model, best_seed, best_reward)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        model_path = find_model(arg)
    except FileNotFoundError as e:
        print(f"\n  ERROR: {e}")
        sys.exit(1)
    run_demo(model_path)