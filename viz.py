import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel

# Load model
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

mujoco.mj_step(model, data)

print("data ctrl",  data.ctrl)
# Visualization loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("HELLOOO)")
    count = 0
    while viewer.is_running():
        if count < 1000:
            data.ctrl[0] = 1
            print("data ctrl",  data.ctrl)
        else:
            data.ctrl[0] = 0
            print("data ctrl", data.ctrl)
            if count > 2000:
                count = 0

        mujoco.mj_step(model, data)
        viewer.sync()
        count += 1


# TODO finish sysid -> collect trajectory data on real arm, record timesteps, joint command positions, and current position, and then
# tune PD parameters to match those in sim

# done: collect traj data, python file to replay it in simulation
# TODO do grid search over kp, kd parameters 