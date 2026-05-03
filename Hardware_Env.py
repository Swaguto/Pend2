import serial
import time
import numpy as np
import gymnasium as gym

class HardwarePendulumEnv(gym.Env):
    """
    Physical hardware bridge for the RoboPend system.
    Communicates with Arduino over Serial.
    """
    
    # --- Physical Scaling ---
    TICKS_PER_REV = 2400.0
    PULLEY_CIRC   = 0.040   # 40mm = 0.04m
    
    CART_CENTRE   = -0.055  # Midpoint in meters (adjust to your rail)
    CART_RANGE    = 0.27    # Half-width in meters
    
    MAX_STEPS     = 1000
    
    def __init__(self, port='COM3', baudrate=115200):
        super().__init__()
        
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            time.sleep(2) # Wait for Arduino reset
        except Exception as e:
            print(f"Error connecting to serial: {e}")
            self.ser = None

        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (5,), dtype=np.float32)
        
        self.last_cart_pos = 0.0
        self.last_pend_pos = 0.0
        self.last_time     = time.time()
        
        # State estimation filters
        self.cart_vel_smooth = 0.0
        self.pend_vel_smooth = 0.0
        self.alpha = 0.3 # Alpha filter weight
        
        self.step_count = 0

    def _read_sensors(self):
        """Read latest P <cart> <pend> from serial buffer."""
        if not self.ser: return 0, 0
        
        # Clear buffer to get freshest data
        while self.ser.in_waiting > 100:
            self.ser.readline()
            
        line = self.ser.readline().decode('ascii', errors='ignore').strip()
        if line.startswith("P"):
            parts = line.split()
            if len(parts) >= 3:
                return int(parts[1]), int(parts[2])
        return None, None

    def _get_obs(self):
        ticks_cart, ticks_pend = self._read_sensors()
        if ticks_cart is None: # Fallback to last known if read failed
            return self._last_obs
        
        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        
        # Convert ticks to meters/radians
        cart_pos = ticks_cart * (self.PULLEY_CIRC / self.TICKS_PER_REV)
        pend_pos = ticks_pend * (2 * np.pi / self.TICKS_PER_REV)
        
        # Calculate Velocities
        if dt > 0:
            v_cart = (cart_pos - self.last_cart_pos) / dt
            v_pend = (pend_pos - self.last_pend_pos) / dt
            
            # Smooth
            self.cart_vel_smooth = (1-self.alpha)*self.cart_vel_smooth + self.alpha*v_cart
            self.pend_vel_smooth = (1-self.alpha)*self.pend_vel_smooth + self.alpha*v_pend
            
        self.last_cart_pos = cart_pos
        self.last_pend_pos = pend_pos
        
        # Match training obs space: 
        # [cart_norm, cart_vel, cos, sin, pend_vel]
        shifted = pend_pos - np.pi # Assuming hanging is 0, upright is pi
        
        obs = np.array([
            (cart_pos - self.CART_CENTRE) / self.CART_RANGE,
            self.cart_vel_smooth / 2.0,
            np.cos(shifted),
            np.sin(shifted),
            self.pend_vel_smooth / 10.0
        ], dtype=np.float32)
        
        self._last_obs = obs
        return obs

    def step(self, action):
        self.step_count += 1
        
        # Map Force Action (-1 to 1) to Stepper Velocity (example mapping)
        # In a real system, you might want Force -> Acceleration
        target_vel = float(action[0]) * 3000.0 # Steps per second
        
        if self.ser:
            self.ser.write(f"V {target_vel}\n".encode())
            
        obs = self._get_obs()
        
        # Dummy reward/done for evaluation
        # Real training would need full reward logic
        done = self.step_count >= self.MAX_STEPS
        return obs, 0.0, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        print("\n[RESET] Manual Homing Sequence:")
        print("1. Center the cart.")
        print("2. Hold the pendulum at the bottom (hanging).")
        input("Press Enter when ready...")
        
        if self.ser:
            self.ser.write(b"H\n") # Home command
            time.sleep(0.5)
            self.ser.reset_input_buffer()
            
        self.last_time = time.time()
        return self._get_obs(), {}

    def close(self):
        if self.ser:
            self.ser.write(b"V 0\n") # Stop motor
            self.ser.close()
