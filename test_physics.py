import pybullet as p
import pybullet_data
import time
import math

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf", useFixedBase=True)

for j in range(p.getNumJoints(robot)):
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

p.setGravity(0, 0, -9.81)

# Push the cart
p.setJointMotorControl2(robot, 3, p.TORQUE_CONTROL, force=20)
for i in range(100):
    p.stepSimulation()
    angle = p.getJointState(robot, 7)[0]
    print(f"Step {i}: Cart Pos = {p.getJointState(robot, 3)[0]:.3f}, Pend Angle = {angle:.3f}")

p.disconnect()
