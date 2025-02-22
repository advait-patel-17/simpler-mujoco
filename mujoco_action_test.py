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

ds_filepath = "./data/episode_0.hdf5"
file = h5py.File(ds_filepath, 'r')
joint_actions = np.array(file["joint_action"])
obs_joint_pos = file["/observations/full_joint_pos"]
obs_ee_pos = file["/observations/ee_pos"]
timestamps = file["timestamp"]
print(obs_joint_pos[0])

# joint_actions[131] = (joint_actions[133] - joint_actions[130])/3 + joint_actions[130]
# joint_actions[132] = 2*(joint_actions[133] - joint_actions[130])/3 + joint_actions[130]

data.qpos[:7] = obs_joint_pos[0][:7]
mujoco.mj_step(model, data)

dataset_len = len(joint_actions)
count = 0

start = time.time()
score = 0

start_err = obs_joint_pos[count][:7] - data.qpos[:7]
print("start err:", start_err)
sim_curr_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')]
obs_curr_ee_pos = obs_joint_pos[count][:7]
sim_last_ee_pos = sim_curr_ee_pos
obs_last_ee_pos = obs_curr_ee_pos
last_qpos = data.qpos[:7]

print("initial action:", joint_actions[0])
print("initial pos:", data.qpos[:7])

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and count < dataset_len:
        error = obs_joint_pos[count][:7] - data.qpos[:7]
        score += np.sum(np.abs(error))
        # print("error:", error)
        # print("score:", np.sum(np.abs(error)))
        # print("data qpos", data.qpos[:7])
        # print("ee pos:", obs_ee_pos[count])
        sim_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')]
        # print("sim ee pos:", sim_ee_pos)
        # print("EE ERROR:", np.linalg.norm(sim_ee_pos - obs_ee_pos[count][:3]))
        curr_sim_time = data.time
        data.ctrl[:7] = joint_actions[count]
        last_qpos = data.qpos[:7]
        while data.time < curr_sim_time + TIMESTEP:
            mujoco.mj_step(model, data)
            
            viewer.sync()
            sim_last_ee_pos = sim_curr_ee_pos.copy()
            sim_curr_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')].copy()
            # print("error:", np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)))
            # if np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)) > 0.0025:
            #     print("================== BIG SIM DIFF ====================")
            #     print("sim last ee pos:", sim_last_ee_pos)
            #     print("sim curr ee pos:", sim_curr_ee_pos)
            #     print("diff:", np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)))
            #     print("obs ee pos:", obs_ee_pos[count])
            #     print("joint action", joint_actions[count])
            #     print("count:", count)

        obs_last_ee_pos = obs_curr_ee_pos
        obs_curr_ee_pos = obs_joint_pos[count][:7]
        if np.sum(np.abs(obs_curr_ee_pos[:3] - obs_last_ee_pos[:3])) > 0.2:
            print("================== BIG OBS DIFF ====================")
            print("obs last ee pos:", obs_last_ee_pos)
            print("obs curr ee pos:", obs_curr_ee_pos)
            print("diff:", np.sum(np.abs(obs_curr_ee_pos[:3] - obs_last_ee_pos[:3])))
            print("count:", count)
        if np.sum(np.abs(data.qpos[:7] - last_qpos[:7])) > 0.1:
            print("================== BIG JOINT DIFF ====================")
            print("last qpos:", last_qpos[:7])
            print("sim joint pos:", data.qpos[:7])
            print("diff:", np.sum(np.abs(data.qpos[:7] - obs_joint_pos[count][:7])))
            print("count:", count)
        if count > 1 and np.sum(np.abs(joint_actions[count] - joint_actions[count - 1])) > 0.5:
            print("================== BIG ACTION DIFF ====================")
            print("last action:", joint_actions[count - 1])
            print("curr action:", joint_actions[count]) 
            print("diff:", np.sum(np.abs(joint_actions[count] - joint_actions[count - 1])))
            print("count:", count)

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