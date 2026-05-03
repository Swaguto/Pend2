# find_joints.py
import pybullet as p
import pybullet_data
import numpy as np
import time

cid = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(2.0, 0, -20, [0, 0, 1.5])
p.loadURDF("plane.urdf")
robot = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf",
                   [0, 0, 1.5], useFixedBase=True)

TYPES = {0:"REVOLUTE", 1:"PRISMATIC", 2:"SPHERICAL", 3:"PLANAR", 4:"FIXED"}

print(f"\nTotal joints: {p.getNumJoints(robot)}\n")
print(f"{'Idx':<5} {'Type':<12} {'Name'}")
print("-" * 50)

movable = []
for i in range(p.getNumJoints(robot)):
    info  = p.getJointInfo(robot, i)
    jtype = TYPES.get(info[2], "UNKNOWN")
    name  = info[1].decode()
    print(f"{i:<5} {jtype:<12} {name}")
    if info[2] != 4:   # not fixed
        movable.append(i)

print(f"\nMovable joint indices: {movable}")

# Disable ONLY the movable joints
for j in movable:
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

# Set pendulum hanging with a kick
cart_idx = movable[0]
pend_idx = movable[1]
p.resetJointState(robot, cart_idx, targetValue=-0.055, targetVelocity=0)
p.resetJointState(robot, pend_idx, targetValue=0 + 0.1, targetVelocity=0)
for j in movable:
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

print(f"\nUsing cart_idx={cart_idx}, pend_idx={pend_idx}")
print("Simulating — pendulum should swing freely...\n")

for i in range(300):
    p.stepSimulation()
    if i % 60 == 0:
        s = p.getJointStates(robot, [cart_idx, pend_idx])
        print(f"t={i/60:.1f}s | cart={s[0][0]:.4f} | "
              f"angle={np.degrees(s[1][0]):.2f}°")
    time.sleep(1/60)

p.disconnect()