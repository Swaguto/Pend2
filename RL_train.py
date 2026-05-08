"""
==============================================================================
              PENDULUM SWING-UP - MAXIMUM SPEED EDITION                      

  Fixes applied vs previous version:                                          
    * cart_idx=3, pend_idx=7  (confirmed by find_joints.py)                  
    * URDF angle convention: theta=0 = hanging, theta=pi = upright                    
    * Observation shifted so cos=+1 always means upright                      
    * Upright bonus so agent learns to HOLD, not just reach                   
    * 2M timesteps - swing-up needs more exploration than balance alone       
                                                                              
  Speed optimisations:                                                        
  [1] Fast reset      - resetJointState only, zero world rebuilds             
  [2] SubprocVecEnv   - one PyBullet per CPU core, true parallelism           
  [3] SAC             - off-policy, 4-5x fewer steps than PPO                 
  [4] Dense sub-steps - 8 steps @ 480 Hz, half the Python<->C++ crossings      
  [5] RealTimeSim off - stops PyBullet fighting your step loop                
  [6] FileCaching off - no redundant URDF disk IO                             
  [7] getJointStates  - both joints in one C++ call, not two                  
  [8] Checkpointing   - resume from interruptions, never lose progress        
  [9] EvalCallback    - saves best model automatically                        
==============================================================================

Usage:
    pip install stable-baselines3 gymnasium pybullet torch
    python pendulum_ludicrous_speed.py

    # Delete old checkpoints before retraining from scratch:
    # rmdir /s /q checkpoints && del pendulum_final.zip
"""

import os
import io
import sys
import time
import multiprocessing

# Force UTF-8 so unicode chars never crash on Windows cp1252 terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════════════

class PendulumEnv(gym.Env):
    """
    Cart-pole swing-up and balance using pend_assem URDF.

    JOINT INDICES (confirmed by find_joints.py on this URDF):
        Idx 3  PRISMATIC  dof_cart_slider_0    <- cart slides on rail
        Idx 7  REVOLUTE   dof_pendulum_pivot_0 <- pendulum pivots on cart

    ANGLE CONVENTION (confirmed by visual inspection):
        theta = 0  -> pendulum hanging down  (start state)
        theta = pi  -> pendulum upright       (goal state)

    RAIL LIMITS (from URDF):
        lower = -0.325 m,  upper = +0.215 m  (asymmetric, centre ≈ -0.055 m)
    """

    # ── Rail geometry ─────────────────────────────────────────────────────────
    CART_LOWER  = -0.325
    CART_UPPER  =  0.215
    CART_CENTRE = (CART_LOWER + CART_UPPER) / 2.0
    CART_RANGE  = (CART_UPPER - CART_LOWER) / 2.0
    CART_MARGIN =  0.02

    # ── Actuation ─────────────────────────────────────────────────────────────
    # Matched to hardware settings
    ACCEL_SCALE      = 400_000.0   # steps / s²
    MAX_SPEED        = 4000.0      # steps/sec
    COUNTS_PER_METER = 40290.0
    EMA_ALPHA        = 0.35

    # ── Episode ───────────────────────────────────────────────────────────────
    MAX_STEPS   = 1000

    # ── Simulation ────────────────────────────────────────────────────────────
    SIM_HZ    = 480
    CTRL_SKIP = 8     # → 60 Hz effective control rate

    def __init__(self, render: bool = False):
        super().__init__()

        self._cid = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._cid)
        p.setRealTimeSimulation(0, physicsClientId=self._cid)
        p.setPhysicsEngineParameter(
            enableFileCaching=0, physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.SIM_HZ, physicsClientId=self._cid)

        # ── Confirmed joint indices ───────────────────────────────────────────
        self.cart_idx  = 3
        self.pend_idx  = 7
        self.urdf_path = "Pend_assem_stl/pend_assem/urdf/pend_assem.urdf"
        self.robot     = None
        self._step_count = 0

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self._speed = 0.0

        self._build_world()   # [1a] build once, never again in reset()

    def _build_world(self):
        p.resetSimulation(physicsClientId=self._cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.SIM_HZ, physicsClientId=self._cid)
        p.setRealTimeSimulation(0, physicsClientId=self._cid)
        p.loadURDF("plane.urdf", physicsClientId=self._cid)
        self.robot = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 1.5],
            useFixedBase=True,
            physicsClientId=self._cid,
        )
        
        # Fetch base masses for Domain Randomization
        cart_info = p.getDynamicsInfo(self.robot, self.cart_idx, physicsClientId=self._cid)
        pend_info = p.getDynamicsInfo(self.robot, self.pend_idx, physicsClientId=self._cid)
        self.base_cart_mass = cart_info[0]
        self.base_pend_mass = pend_info[0]

        # Preserve URDF damping — it's physically correct, don't zero it
        self._disable_motors()

    def _disable_motors(self):
        """
        Kill PyBullet's default velocity-control holding force on the pendulum joint.
        The cart joint will be explicitly controlled via velocity commands.
        """
        p.setJointMotorControl2(
            self.robot, self.pend_idx,
            p.VELOCITY_CONTROL, force=0,
            physicsClientId=self._cid,
        )

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        self._speed = 0.0

        # ── Domain Randomization ──────────────────────────────────────────────
        # Randomize mass by +/- 15%
        cart_mass = self.base_cart_mass * np.random.uniform(0.85, 1.15)
        pend_mass = self.base_pend_mass * np.random.uniform(0.85, 1.15)
        p.changeDynamics(self.robot, self.cart_idx, mass=cart_mass, physicsClientId=self._cid)
        p.changeDynamics(self.robot, self.pend_idx, mass=pend_mass, physicsClientId=self._cid)

        # Randomize friction/damping of the rail and pivot
        p.changeDynamics(self.robot, self.cart_idx, linearDamping=np.random.uniform(0.0, 0.1), physicsClientId=self._cid)
        p.changeDynamics(self.robot, self.pend_idx, jointDamping=np.random.uniform(0.0, 0.02), physicsClientId=self._cid)

        # Cart near the physical rail centre (rail is NOT symmetric around 0)
        p.resetJointState(
            self.robot, self.cart_idx,
            targetValue=self.CART_CENTRE + np.random.uniform(-0.03, 0.03),
            targetVelocity=0.0,
            physicsClientId=self._cid,
        )

        # ── Start HANGING: θ≈0 in this URDF ──────────────────────────────────
        # Small random kick breaks symmetry so the agent explores both directions
        p.resetJointState(
            self.robot, self.pend_idx,
            targetValue=np.random.uniform(-0.15, 0.15),
            targetVelocity=np.random.uniform(-0.3, 0.3),
            physicsClientId=self._cid,
        )

        self._disable_motors()   # resetJointState re-enables holds — kill again
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        # [7] Read both joints in one C++ call
        states     = p.getJointStates(
            self.robot,
            [self.cart_idx, self.pend_idx],
            physicsClientId=self._cid,
        )
        cart_pos   = states[0][0]
        cart_vel   = states[0][1]
        pole_angle = states[1][0]
        pole_vel   = states[1][1]

        # ── Angle convention correction ───────────────────────────────────────
        # This URDF: theta=0 = hanging, theta=pi = upright
        # Subtract pi so the network sees:
        #   cos(shifted) = +1 when upright  <- reward maximum
        #   cos(shifted) = -1 when hanging  <- reward minimum
        # Without this, the reward signal is inverted and the agent learns
        # to keep it hanging (the wrong goal).
        shifted = pole_angle - np.pi

        return np.array([
            (cart_pos - self.CART_CENTRE) / self.CART_RANGE,
            cart_vel  / 2.0,
            np.cos(shifted),   # +1 upright, -1 hanging
            np.sin(shifted),   # encodes which side it's falling toward
            pole_vel  / 10.0,
        ], dtype=np.float32)

    def step(self, action):
        self._step_count += 1

        # Integrate action into target stepper motor speed
        dt = float(self.CTRL_SKIP) / self.SIM_HZ
        self._speed -= float(action[0]) * self.ACCEL_SCALE * dt
        self._speed = float(np.clip(self._speed, -self.MAX_SPEED, self.MAX_SPEED))

        # Convert steps/sec to m/s for PyBullet
        target_vel_m_s = self._speed / self.COUNTS_PER_METER

        # Apply to PyBullet using VELOCITY_CONTROL (matching hardware stepper motor)
        p.setJointMotorControl2(
            self.robot, self.cart_idx,
            p.VELOCITY_CONTROL,
            targetVelocity=target_vel_m_s,
            force=200.0,  # Max force the simulated motor can use to hit target velocity
            physicsClientId=self._cid,
        )

        for _ in range(self.CTRL_SKIP):   # [4] tight sub-step loop
            p.stepSimulation(physicsClientId=self._cid)

        obs = self._get_obs()
        cart_pos_norm, _, cos_theta, sin_theta, pole_vel = obs
        actual_cart_pos = cart_pos_norm * self.CART_RANGE + self.CART_CENTRE
        angle_from_top  = abs(np.arctan2(sin_theta, cos_theta))  # 0 = upright

        # ── Reward ────────────────────────────────────────────────────────────

        # 1. Height: primary swing-up drive. cos_theta = +1 upright, -1 hanging
        height_reward  = cos_theta

        # 2. Upright bonus: sharp extra reward inside ±17° of vertical.
        #    Makes the target unmistakable — without it the agent hovers
        #    near the top but never commits to holding it.
        upright_bonus  = 3.0 * (1.0 - angle_from_top / 0.3) \
                         if angle_from_top < 0.3 else 0.0

        # 3. Velocity penalty near top: teaches the agent to decelerate
        #    and hold rather than swinging through and falling over.
        vel_penalty    = -0.2 * pole_vel ** 2 if angle_from_top < 0.4 else 0.0

        # 4. Cart centering: rail is only 0.54 m — stay near the middle
        cart_penalty   = -1.5 * cart_pos_norm ** 2

        # 5. Action penalty: discourages wasteful oscillation
        action_penalty = -0.001 * float(action[0]) ** 2

        reward = (height_reward + upright_bonus + vel_penalty
                  + cart_penalty + action_penalty)

        # ── Termination ───────────────────────────────────────────────────────
        terminated = bool(
            actual_cart_pos < self.CART_LOWER + self.CART_MARGIN or
            actual_cart_pos > self.CART_UPPER - self.CART_MARGIN
        )
        if terminated:
            reward -= 100.0  # Huge penalty for hitting the rail to stop suicide

        truncated = bool(self._step_count >= self.MAX_STEPS)

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def make_env():
    def _init():
        return PendulumEnv(render=False)
    return _init


def train():
    os.makedirs("checkpoints/best", exist_ok=True)

    n_envs = multiprocessing.cpu_count()
    print(f"\n  CPU cores  : {n_envs}")
    print(f"  Sim Hz     : {PendulumEnv.SIM_HZ}  ctrl skip: {PendulumEnv.CTRL_SKIP}")
    print(f"  Control Hz : {PendulumEnv.SIM_HZ // PendulumEnv.CTRL_SKIP}")
    print(f"  cart_idx=3, pend_idx=7  (PRISMATIC + REVOLUTE)")
    print(f"  Angle zero = hanging, pi = upright")
    print()

    train_env = SubprocVecEnv([make_env() for _ in range(n_envs)])  # [2]
    eval_env  = PendulumEnv(render=False)

    checkpoint_cb = CheckpointCallback(                              # [8]
        save_freq=max(50_000 // n_envs, 1),
        save_path="checkpoints/",
        name_prefix="pendulum",
        verbose=0,
    )
    eval_cb = EvalCallback(                                          # [9]
        eval_env,
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=20,
        best_model_save_path="checkpoints/best/",
        verbose=1,
    )

    latest_best = "checkpoints/best/best_model.zip"
    final_fallback = "pendulum_final.zip"
    
    if os.path.exists(latest_best):
        print(f"  Resuming from {latest_best}\n")
        model = SAC.load(latest_best, env=train_env)
        resuming = True
    elif os.path.exists(final_fallback):
        print(f"  Recovery mode: Loading from {final_fallback}\n")
        model = SAC.load(final_fallback, env=train_env)
        resuming = True
    else:
        resuming = False
        model = SAC(                                                 # [3]
            "MlpPolicy",
            train_env,
            verbose=1,
            device="cpu",
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[64, 64]),
        )

    print("  GO!\n")
    t0 = time.perf_counter()

    model.learn(
        total_timesteps=23_000_000,   # Recovery training
        callback=CallbackList([checkpoint_cb, eval_cb]),
        reset_num_timesteps=not resuming,
    )

    elapsed = time.perf_counter() - t0
    print(f"\n  Done in {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    model.save("pendulum_final")
    print("  Saved -> pendulum_final.zip")

    train_env.close()
    eval_env.close()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo(model):
    print("\n  Running demo - Ctrl-C to quit\n")
    env = PendulumEnv(render=True)

    # Point camera at the robot - it loads at z=1.5
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=90,
        cameraPitch=-15,
        cameraTargetPosition=[0, 0, 1.5],
        physicsClientId=env._cid,
    )

    obs, _  = env.reset()
    episode = 1
    steps   = 0
    cum_rew = 0.0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            cum_rew += reward
            steps   += 1
            time.sleep(1.0 / 60.0)

            if terminated or truncated:
                status = "BALANCED [OK]" if not terminated else "HIT RAIL  [X]"
                print(f"  Episode {episode:3d} | steps={steps:4d} | "
                      f"reward={cum_rew:8.1f} | {status}")
                obs, _ = env.reset()
                episode += 1
                steps   = 0
                cum_rew = 0.0

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        env.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
  ===========================================
  |   PENDULUM - LUDICROUS SPEED EDITION    |
  ===========================================
    """)
    trained_model = train()
    demo(trained_model)