import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel
# Loading a specific model description as an imported module.
from robot_descriptions import aloha_mj_description
# model = mujoco.MjModel.from_xml_path(aloha_mj_description.MJCF_PATH)
import time
# # Directly loading an instance of MjModel.
from robot_descriptions.loaders.mujoco import load_robot_description

import h5py
# model = load_robot_description("panda_mj_description")

# Loading a variant of the model, e.g. panda without a gripper.
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

TIMESTEP = 0.1

ds_filepath = "./data/episode_1.hdf5"
file = h5py.File(ds_filepath, 'r')
joint_actions = file["joint_action"]
obs_joint_pos = file["/observations/full_joint_pos"]
timestamps = file["timestamp"]
print(joint_actions[0])

dataset_len = len(joint_actions)
count = 0

start = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and count < dataset_len:
        curr_sim_time = data.time
        data.ctrl[:7] = joint_actions[count]
        while data.time < curr_sim_time + TIMESTEP:
            mujoco.mj_step(model, data)
            viewer.sync()

        error = obs_joint_pos[count][:7] - data.qpos[:7]
        count += 1

        print("error:", error)

end = time.time()
print("time elapsed:", end - start)
"""
DIMENSIONS

table length x width x height (m): 1.21 x 0.76 x 0.75

"""

# mujoco actuator docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
