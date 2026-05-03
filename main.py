import pybullet as p
import time
import pybullet_data

# Setup
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load model - useFixedBase=1 keeps it suspended
modelId = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf", [0, 0, 1.5], useFixedBase=1)

# --- VERIFY JOINT INDICES ---
cart_idx = -1
pend_idx = -1

for i in range(p.getNumJoints(modelId)):
    info = p.getJointInfo(modelId, i)
    joint_name = info[1].decode('utf-8')
    if joint_name == "dof_cart_slider_0":
        cart_idx = i
    elif joint_name == "dof_pendulum_pivot_0":
        pend_idx = i

# --- SETUP SLIDERS ---
# Rail limits set to 30cm (-0.3 to 0.3)
slider_cart = p.addUserDebugParameter("Cart Position", -0.3, 0.3, 0)
# This slider now acts as a toggle: 
# Left (< 0.5) = Motor holds the pendulum
# Right (> 0.5) = Gravity takes over
slider_gravity = p.addUserDebugParameter("0: Motor | 1: Gravity", 0, 1, 0)
slider_force = p.addUserDebugParameter("Cart Motor Strength", 0, 500, 200)

# Simulation Loop
while True:
    target_pos = p.readUserDebugParameter(slider_cart)
    gravity_toggle = p.readUserDebugParameter(slider_gravity)
    current_force = p.readUserDebugParameter(slider_force)

    # 1. Control the Cart (Always on POSITION_CONTROL)
    if cart_idx != -1:
        p.setJointMotorControl2(modelId, cart_idx, p.POSITION_CONTROL, 
                                targetPosition=target_pos, force=current_force)
    
    # 2. Control the Pendulum
    if pend_idx != -1:
        if gravity_toggle < 0.5:
            # MOTOR MODE: Holds the pendulum upright (0 rad)
            p.setJointMotorControl2(modelId, pend_idx, p.POSITION_CONTROL, 
                                    targetPosition=0, force=current_force)
        else:
            # GRAVITY MODE: Disables the motor (force=0) so it swings freely
            p.setJointMotorControl2(modelId, pend_idx, p.VELOCITY_CONTROL, 
                                    targetVelocity=0, force=0)

    p.stepSimulation()
    time.sleep(1./240.)