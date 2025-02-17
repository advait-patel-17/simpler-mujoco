import mujoco
import mujoco.viewer
import numpy as np
from mujoco import MjData, MjModel
# Loading a specific model description as an imported module.
# model = mujoco.MjModel.from_xml_path(aloha_mj_description.MJCF_PATH)
import time
# # Directly loading an instance of MjModel.

import h5py
# model = load_robot_description("panda_mj_description")

# Loading a variant of the model, e.g. panda without a gripper.
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

TIMESTEP = 0.1

ds_filepath = "./data/episode_1.hdf5"
file = h5py.File(ds_filepath, 'r')
joint_actions = file["joint_action"]
obs_joint_pos = file["/observations/full_joint_pos"]
obs_ee_pos = file["/observations/ee_pos"]
timestamps = file["timestamp"]
print(obs_joint_pos[0])

data.qpos[:7] = obs_joint_pos[0][:7]
mujoco.mj_step(model, data)

dataset_len = len(joint_actions)
count = 0

start = time.time()
score = 0

start_err = obs_joint_pos[count][:7] - data.qpos[:7]
print("start err:", start_err)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and count < dataset_len:
        error = obs_joint_pos[count][:7] - data.qpos[:7]
        score += np.sum(np.abs(error))
        print("error:", error)
        print("score:", np.sum(np.abs(error)))
        print("data qpos", data.qpos[:7])
        print("ee pos:", obs_ee_pos[count])
        sim_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper_')]
        print("sim ee pos:", sim_ee_pos)
        print("EE ERROR:", np.linalg.norm(sim_ee_pos - obs_ee_pos[count][:3]))
        curr_sim_time = data.time
        data.ctrl[:7] = joint_actions[count]
        while data.time < curr_sim_time + TIMESTEP:
            mujoco.mj_step(model, data)
            viewer.sync()
        count += 1
    viewer.close()

end = time.time()
print("time elapsed:", end - start)
print("score", score)
"""
DIMENSIONS

table length x width x height (m): 1.21 x 0.76 x 0.75

"""

# mujoco actuator docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
