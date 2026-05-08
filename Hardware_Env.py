"""
Hardware_Env.py
===============
Production hardware interface for RoboPend sim-to-real deployment.

This Gymnasium environment talks to the Arduino over Serial and produces
observations that are IDENTICAL to what PendulumEnv (RL_train.py) produced
during training. The pre-trained SAC policy can therefore run without
any modification.

USAGE — run inference:
    from Hardware_Env import HardwarePendulumEnv
    from stable_baselines3 import SAC

    model = SAC.load("pendulum_final.zip")
    env   = HardwarePendulumEnv()
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            obs, _ = env.reset()

USAGE — sensor sanity check (run before RL):
    python Hardware_Env.py

SERIAL PROTOCOL (firmware v4.0):
    Python → Arduino : "V<float>\\n"  set velocity (steps/sec)
                        "Z\\n"         zero both encoders in place
                        "H\\n"         stop + re-enable
    Arduino → Python : "P <pend> <cart>\\n"   100 Hz
                        "LIMIT\\n"             cart hit safety boundary
                        "READY\\n"             boot confirmation
"""

import sys
import time
import serial
import numpy as np
import gymnasium as gym


# ── Physical constants ────────────────────────────────────────────────────────

PEND_CPR         = 2400.0      # encoder counts per full revolution
                               # (600 PPR × 4 quadrature edges)
                               # hanging = 0, upright = 1200

COUNTS_PER_METER = 40290.0     # cart encoder counts per metre
                               # measured: 4029 counts / 10 cm

RAIL_HALF_COUNTS = 14500       # half of the 29,000-count rail
CART_RANGE_M     = RAIL_HALF_COUNTS / COUNTS_PER_METER  # ≈ 0.360 m
# CART_CENTRE = 0.0 by construction (we zero the encoder at the rail centre)

# ── Training env normalisation (MUST match RL_train.py exactly) ───────────────
# obs[1] = cart_vel  / 2.0
# obs[4] = pend_vel  / 10.0
CART_VEL_DENOM   = 2.0
PEND_VEL_DENOM   = 10.0

MAX_SPEED        = 4000.0      # steps/sec — must match firmware setMaxSpeed()

# ── Sensor Trim ───────────────────────────────────────────────────────────────
# If the cart balances but slowly drifts to the RIGHT, increase this (e.g., +0.02)
# If the cart balances but slowly drifts to the LEFT, decrease this  (e.g., -0.02)
ANGLE_TRIM       = 0.02         # radians

# ── Action → velocity integration ────────────────────────────────────────────
# The sim applies force (N) to a cart with inertia.
# On hardware we command velocity directly, so we integrate:
#   speed += action × ACCEL_SCALE × dt
# ACCEL_SCALE is the single tuning knob:
#   too low  → can't build swing momentum
#   too high → violent oscillation, can't balance
# Start here, increase by 50_000 if swing-up is too weak.
ACCEL_SCALE      = 400_000.0   # steps / s²

# ── Velocity smoothing ────────────────────────────────────────────────────────
# Exponential moving average applied to finite-differenced velocities.
# 0 = fully smoothed (very slow to respond), 1 = no smoothing (noisy).
EMA_ALPHA        = 0.35

# ── Serial ────────────────────────────────────────────────────────────────────
DEFAULT_PORT     = "COM5"
BAUD             = 115200


# ── Environment ──────────────────────────────────────────────────────────────

class HardwarePendulumEnv(gym.Env):
    """
    Gymnasium wrapper around the RoboPend hardware.

    Observation vector (identical to PendulumEnv in RL_train.py):
        [0]  cart_pos_norm   = cart_pos_m / CART_RANGE_M
        [1]  cart_vel_norm   = cart_vel_m_s / 2.0
        [2]  cos(θ − π)      = +1.0 upright,  −1.0 hanging
        [3]  sin(θ − π)      = direction of lean
        [4]  pend_vel_norm   = pend_vel_rad_s / 10.0

    Action:
        Scalar in [−1, 1].  Integrated into motor velocity each step.
    """

    metadata = {"render_modes": []}

    def __init__(self, port: str = DEFAULT_PORT, baud: int = BAUD):
        super().__init__()

        self.port = port
        self.baud = baud
        self.ser  = None

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        # Integrated motor velocity (steps/sec)
        self._speed        = 0.0

        # Previous encoder readings for finite-difference velocity
        self._prev_pend    = 0
        self._prev_cart    = 0
        self._prev_t       = time.perf_counter()

        # EMA-smoothed velocities
        self._cart_vel_ema = 0.0
        self._pend_vel_ema = 0.0

        # Last valid observation (returned on read timeout)
        self._last_obs     = np.zeros(5, dtype=np.float32)

        self._step_count   = 0

        self._connect()

    # ── Serial helpers ────────────────────────────────────────────────────────

    def _connect(self):
        print(f"Connecting to Arduino on {self.port} @ {self.baud} baud ...")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.05)
            time.sleep(2.0)          # wait for Arduino bootloader
            self.ser.reset_input_buffer()
            print("Connected.\n")
        except serial.SerialException as e:
            if "PermissionError" in str(e) or "Access is denied" in str(e):
                print(f"\n[ERROR] {self.port} is in use.")
                print("Close the Arduino Serial Monitor and any other script.\n")
            raise

    def _send(self, msg: str):
        """Write a command string to the Arduino."""
        self.ser.write(msg.encode())

    def _read_latest(self):
        """
        Return the freshest (pend_counts, cart_counts) from the serial buffer.

        Flushes stale packets so the policy always acts on the newest data.
        Returns ("LIMIT", None) when the Arduino reports a safety boundary hit.
        Falls back to the previous reading on timeout.
        """
        # Discard backed-up data if Python is slower than Arduino
        while self.ser.in_waiting > 150:
            self.ser.readline()

        deadline = time.perf_counter() + 0.08  # 80 ms timeout
        while time.perf_counter() < deadline:
            try:
                line = self.ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:
                continue

            if not line:
                continue
            if "LIMIT" in line:
                return "LIMIT", None
            if not line.startswith("P "):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pend = int(parts[1])   # pendulum — first value in stream
                cart = int(parts[2])   # cart     — second value in stream
                return pend, cart
            except ValueError:
                continue

        # Timeout — return last known values to avoid crashing the loop
        return self._prev_pend, self._prev_cart

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, pend_cnt: int, cart_cnt: int) -> np.ndarray:
        """
        Convert raw encoder counts to the 5-element obs vector.
        Every formula here must mirror PendulumEnv._get_obs() exactly.
        """
        now = time.perf_counter()
        dt  = max(now - self._prev_t, 1e-4)

        # ── Cart position (metres, zeroed at rail centre) ─────────────────
        cart_m        = cart_cnt / COUNTS_PER_METER
        cart_pos_norm = cart_m / CART_RANGE_M
        # CART_CENTRE = 0 by construction → no subtraction needed

        # ── Pendulum angle ────────────────────────────────────────────────
        # 0 counts = hanging (θ=0), 1200 counts = upright (θ=π)
        pend_angle    = pend_cnt * (2.0 * np.pi / PEND_CPR)   # radians
        shifted       = pend_angle - np.pi + ANGLE_TRIM       # 0=upright
        cos_theta     = float(np.cos(shifted))   # +1 upright, −1 hanging
        sin_theta     = float(np.sin(shifted))   # direction of lean

        # ── Velocities (finite difference → EMA smoothing) ───────────────
        d_cart = cart_cnt - self._prev_cart
        d_pend = pend_cnt - self._prev_pend

        raw_cart_vel  = (d_cart / COUNTS_PER_METER) / dt          # m/s
        raw_pend_vel  = (d_pend / PEND_CPR * 2.0 * np.pi) / dt   # rad/s

        self._cart_vel_ema = (EMA_ALPHA * raw_cart_vel
                              + (1.0 - EMA_ALPHA) * self._cart_vel_ema)
        self._pend_vel_ema = (EMA_ALPHA * raw_pend_vel
                              + (1.0 - EMA_ALPHA) * self._pend_vel_ema)

        # ── Update history ────────────────────────────────────────────────
        self._prev_cart = cart_cnt
        self._prev_pend = pend_cnt
        self._prev_t    = now

        obs = np.array([
            cart_pos_norm,
            self._cart_vel_ema / CART_VEL_DENOM,
            cos_theta,
            sin_theta,
            self._pend_vel_ema / PEND_VEL_DENOM,
        ], dtype=np.float32)

        self._last_obs = obs
        return obs

    def _auto_home(self):
        print("\n" + "=" * 52)
        print("  AUTO-HOMING INITIATED")
        print("  Keep hands clear of the rail!")
        print("=" * 52)

        # 1. Find Left Limit
        print("  [1/4] Finding left physical limit...")
        self._send("V-800.0\n")
        time.sleep(0.5)
        
        while True:
            _, c1 = self._read_latest()
            time.sleep(0.2)
            _, c2 = self._read_latest()
            if abs(c2 - c1) < 10:
                break # Stalled!
                
        left_limit = c2
        print(f"        Left limit found: {left_limit}")
        
        # 2. Find Right Limit
        print("  [2/4] Finding right physical limit...")
        self._send("V800.0\n")
        time.sleep(0.5)
        
        while True:
            _, c1 = self._read_latest()
            time.sleep(0.2)
            _, c2 = self._read_latest()
            if abs(c2 - c1) < 10:
                break # Stalled!
                
        right_limit = c2
        print(f"        Right limit found: {right_limit}")
        
        # 3. Calculate Center and Drive There
        center = (left_limit + right_limit) // 2
        print(f"  [3/4] Calculated center: {center}. Moving to center...")
        
        self._send("V-800.0\n")
        while True:
            _, c = self._read_latest()
            if c <= center:
                break
                
        self._send("V0.0\n")
        
        # 4. Prompt for Pendulum Zeroing
        print("  [4/4] Cart centered!")
        print("\n  Let the pendulum hang STRAIGHT DOWN.")
        input("  Press Enter when ready ...")

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """
        Auto calibration sequence:
          1. Drives cart left until physical stall.
          2. Drives cart right until physical stall.
          3. Calculates center and drives there.
          4. Prompts user to hang pendulum down.
          5. Arduino zeros both encoders here via 'Z' command.
          → RL takes control immediately.
        """
        super().reset(seed=seed)
        self._step_count   = 0
        self._speed        = 0.0
        self._cart_vel_ema = 0.0
        self._pend_vel_ema = 0.0

        self._send("V0.0\n")
        time.sleep(0.05)
        self.ser.reset_input_buffer()

        self._auto_home()

        # Zero both encoders at this exact position
        self._send("Z\n")
        time.sleep(0.1)
        self.ser.reset_input_buffer()

        # First reading (should be ≈ 0, 0 after Z)
        pend, cart = self._read_latest()
        self._prev_pend = pend
        self._prev_cart = cart
        self._prev_t    = time.perf_counter()
        obs = self._build_obs(pend, cart)

        print(f"\n  Calibration done.")
        print(f"  Hanging check → cos(θ): {obs[2]:+.3f}  (should be ≈ −1.0)")
        print(f"  Cart centre   → pos:    {obs[0]:+.3f}  (should be ≈  0.0)")
        print(f"\n  RL is in control. Ctrl-C to stop.\n")

        return obs, {}

    def step(self, action: np.ndarray):
        self._step_count += 1
        now = time.perf_counter()
        dt  = max(now - self._prev_t, 1e-4)

        # ── Integrate force → velocity ────────────────────────────────────
        # INVERTED: If the cart runs away to one side, the motor direction
        # is backwards relative to the encoder. We subtract instead of add.
        self._speed -= float(action[0]) * ACCEL_SCALE * dt
        self._speed  = float(np.clip(self._speed, -MAX_SPEED, MAX_SPEED))
        self._send(f"V{self._speed:.1f}\n")

        # ── Read sensors ──────────────────────────────────────────────────
        result = self._read_latest()
        if result[0] == "LIMIT":
            self._speed = 0.0
            self._send("V0.0\n")
            print("\n  !!! HARDWARE SAFETY LIMIT — stopping !!!")
            return self._last_obs, -100.0, True, False, {"reason": "hw_limit"}

        pend, cart = result
        obs = self._build_obs(pend, cart)
        cart_pos_norm, _, cos_theta, sin_theta, pend_vel_norm = obs

        # ── Reward (mirrors RL_train.py exactly) ──────────────────────────
        angle_from_top = abs(np.arctan2(sin_theta, cos_theta))   # 0 = upright

        height_reward  = cos_theta
        upright_bonus  = (3.0 * (1.0 - angle_from_top / 0.3)
                          if angle_from_top < 0.3 else 0.0)
        vel_penalty    = (-0.2 * pend_vel_norm ** 2
                          if angle_from_top < 0.4 else 0.0)
        cart_penalty   = -1.5 * cart_pos_norm ** 2
        action_penalty = -0.001 * float(action[0]) ** 2

        reward = (height_reward + upright_bonus + vel_penalty
                  + cart_penalty + action_penalty)

        # ── Termination ───────────────────────────────────────────────────
        # Match sim: only terminate when cart hits rail — never on pole angle.
        # The policy MUST be free to swing from hanging, just like in training.
        hit_rail   = abs(cart_pos_norm) > 1.0
        terminated = bool(hit_rail)
        truncated  = bool(self._step_count >= 1000)

        if terminated:
            self._speed = 0.0
            self._send("V0.0\n")
            reward -= 100.0

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.ser and self.ser.is_open:
            self._send("H\n")    # stop motor
            time.sleep(0.1)
            self.ser.close()
            print("Serial closed.")


# ── Standalone sanity check ──────────────────────────────────────────────────

def sanity_check(port: str = DEFAULT_PORT):
    """
    Run this BEFORE the RL policy to verify sensor wiring and scaling.

    Expected readings:
      Pendulum hanging  → obs[2] (cos) ≈ −1.0,  obs[3] (sin) ≈  0.0
      Pendulum upright  → obs[2] (cos) ≈ +1.0,  obs[3] (sin) ≈  0.0
      Cart at centre    → obs[0] ≈  0.0
      Cart at left end  → obs[0] ≈ −1.0
      Cart at right end → obs[0] ≈ +1.0

    If cos / sin are inverted, flip one encoder wire on the pendulum.
    If cart direction is wrong, flip one encoder wire on the cart.
    """
    env = HardwarePendulumEnv(port=port)
    env.reset()

    print("\nMove the pendulum and cart by hand. Ctrl-C to stop.\n")
    print(f"{'Step':>5}  {'cart':>8}  {'cart_vel':>9}  "
          f"{'cos(θ)':>8}  {'sin(θ)':>8}  {'pend_vel':>9}")
    print("─" * 60)

    step = 0
    try:
        while True:
            result = env._read_latest()
            if result[0] == "LIMIT":
                print("LIMIT hit — send H to reset.")
                continue
            pend, cart = result
            obs = env._build_obs(pend, cart)
            print(f"{step:5d}  {obs[0]:+8.3f}  {obs[1]:+9.3f}  "
                  f"{obs[2]:+8.3f}  {obs[3]:+8.3f}  {obs[4]:+9.3f}")
            step += 1
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    port = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    sanity_check(port)
