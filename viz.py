import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

# Load model
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_step(model, data)
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