import mujoco
import numpy as np
from mujoco import MjData, MjModel
import time
import h5py

model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

TIMESTEP = 0.1

ds_filepath = "./data/episode_5.hdf5"
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
sim_curr_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')]
obs_curr_ee_pos = obs_joint_pos[count][:7]
sim_last_ee_pos = sim_curr_ee_pos
obs_last_ee_pos = obs_curr_ee_pos
while count < dataset_len:
    error = obs_joint_pos[count][:7] - data.qpos[:7]
    score += np.sum(np.abs(error))
    # print("ee pos:", obs_ee_pos[count])
    sim_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')]
    # print("sim ee pos:", sim_ee_pos)
    # print("EE ERROR:", np.linalg.norm(sim_ee_pos - obs_ee_pos[count][:3]))
    # print("site xpos:", data.site_xpos)

    curr_sim_time = data.time
    data.ctrl[:7] = joint_actions[count]
    while data.time < curr_sim_time + TIMESTEP:
        mujoco.mj_step(model, data)
        sim_last_ee_pos = sim_curr_ee_pos.copy()
        sim_curr_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')].copy()
        # print("error:", np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)))
        if np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)) > 0.0025:
            print("================== BIG SIM DIFF ====================")
            print("sim last ee pos:", sim_last_ee_pos)
            print("sim curr ee pos:", sim_curr_ee_pos)
            print("diff:", np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)))
            print("obs ee pos:", obs_ee_pos[count])
            print("joint action", joint_actions[count])
            print("count:", count)


    sim_last_ee_pos = sim_curr_ee_pos
    obs_last_ee_pos = obs_curr_ee_pos
    sim_curr_ee_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')]
    obs_curr_ee_pos = obs_joint_pos[count][:7]
    if np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)) > 0.05:
        print("================== BIG SIM DIFF PT 2 ELECTRIC BOOGALOO ====================")
        print("sim last ee pos:", sim_last_ee_pos)
        print("sim curr ee pos:", sim_curr_ee_pos)
        print("diff:", np.sum(np.abs(sim_curr_ee_pos - sim_last_ee_pos)))
        print("count:", count)
    count += 1





print("avg err per state:", score / dataset_len / 6)

end = time.time()
print("time elapsed:", end - start)
print("score", score)
"""
DIMENSIONS

table length x width x height (m): 1.21 x 0.76 x 0.75

"""

# mujoco actuator docs: https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor
