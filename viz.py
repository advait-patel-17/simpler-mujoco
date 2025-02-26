import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

# Load model
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_step(model, data)

base_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'robot_right_base')

print("robot base xpos:", data.site_xpos[base_idx])
print("robot base xmat:", data.site_xmat[base_idx])
ee_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')
print("end effector xpos:", data.site_xpos[ee_idx])
print("end effector xmat:", data.site_xmat[ee_idx])

# Transformation matrix from robot base to world frame
T_base_world = np.array([
    [-1.0,  0.0,  0.0,  0.525],
    [ 0.0, -1.0,  0.0, -0.019],
    [ 0.0,  0.0,  1.0,  0.02 ],
    [ 0.0,  0.0,  0.0,  1.0  ]
])

# For reference:
# robot base xpos: [ 0.525 -0.019  0.02 ]
# robot base xmat: [-1.  0.  0.  0. -1.  0.  0.  0.  1.]
# end effector xpos: [ 0.24353877 -0.019       0.32524417]
# end effector xmat: [-0.99500417  0.          0.09983342  0.         -1.          0.
#   0.09983342  0.          0.99500417]

#ee pos in world coord
ee_pos = np.array([0.24353877,  -0.019 , 0.32524417, 1])
# ee pos in robot base
ee_pos_rob = np.linalg.inv(T_base_world) @ ee_pos
print("ee pos in robot:", ee_pos_rob)



thing = np.zeros((model.nu,))
for i in range(7):
    kp = model.actuator_gainprm[i, 0]
    print(f"Actuator {i} kp: {kp}")
    thing[i] = kp
print("thing:", thing)
# Visualization loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    count = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()


# TODO finish sysid -> collect trajectory data on real arm, record timesteps, joint command positions, and current position, and then
# tune PD parameters to match those in sim

# done: collect traj data, python file to replay it in simulation
# TODO do simulated annealing over kp, kd parameters 