import pybullet as p
import pybullet_data
import time

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf", [0, 0, 1.5], useFixedBase=True)

# Try applying force to joint 3
print("Initial joint 3 pos:", p.getJointState(robot, 3)[0])
p.setJointMotorControl2(robot, 3, p.TORQUE_CONTROL, force=20)
for _ in range(100):
    p.stepSimulation()
print("After 100 steps joint 3 pos:", p.getJointState(robot, 3)[0])

# Try applying force to joint 0?
robot2 = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf", [2, 0, 1.5], useFixedBase=True)
print("Initial joint 0 pos:", p.getJointState(robot2, 0)[0])
p.setJointMotorControl2(robot2, 0, p.TORQUE_CONTROL, force=20)
for _ in range(100):
    p.stepSimulation()
print("After 100 steps joint 0 pos:", p.getJointState(robot2, 0)[0])

p.disconnect()
