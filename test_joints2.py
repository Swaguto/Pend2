import pybullet as p
import pybullet_data

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF("Pend_assem_stl/pend_assem/urdf/pend_assem.urdf", useFixedBase=True)

# Disable velocity control for all joints
for j in range(p.getNumJoints(robot)):
    p.setJointMotorControl2(robot, j, p.VELOCITY_CONTROL, force=0)

# Apply force to joint 3
p.setJointMotorControl2(robot, 3, p.TORQUE_CONTROL, force=20)
for _ in range(100):
    p.stepSimulation()
print("After 100 steps joint 3 pos:", p.getJointState(robot, 3)[0])

p.disconnect()
