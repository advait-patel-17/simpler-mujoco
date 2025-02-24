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

T_base_w = np.array([
    [1, 0, 0, 0.525],
    [0, 1, 0, -0.019],
    [0, 0, 1, 0.02],
    [0, 0, 0, 1]
])

# NOTE: INSTEAD OF ADDING A SITE, WHICH MAY OR MAY NOT BE ACCURATE, JUST PLUG THE JOINT ANGLES INTO THE GENDP FORWARD KINEMATICS CALCULATOR
def ee_to_base(sim_ee_pos, sim_ee_mat):
    # Convert sim_ee_pos and sim_ee_mat to 4x4 pose matrix in base frame
    T_ee_world = np.eye(4)
    T_ee_world[:3,:3] = sim_ee_mat
    T_ee_world[:3,3] = sim_ee_pos
    
    # Transform from world to base frame
    T_ee_base = np.inv(T_base_w) @ T_ee_world
    
    return T_ee_base


def pose_vector_to_matrix(pose):
    """
    Converts a 6D pose vector (x, y, z, roll, pitch, yaw) into a 4x4 homogeneous transformation matrix.
    
    Parameters:
        pose: A tuple or list of six elements: (x, y, z, roll, pitch, yaw)
              where roll, pitch, yaw are in radians.
              
    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """
    x, y, z, roll, pitch, yaw = pose

    # Rotation about x-axis (roll)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Rotation about y-axis (pitch)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Rotation about z-axis (yaw)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: Note the order is important!
    # Here we assume yaw, then pitch, then roll: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx

    # Build the 4x4 homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    
    return T

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running() and count < dataset_len:
        error = obs_joint_pos[count][:7] - data.qpos[:7]
        score += np.sum(np.abs(error))
        # print("error:", error)
        # print("score:", np.sum(np.abs(error)))
        # print("data qpos", data.qpos[:7])
        # print("ee pos:", obs_ee_pos[count])
        ee_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'right/gripper')
        sim_ee_pos = data.site_xpos[ee_idx]
        sim_ee_mat = data.site_xmat[ee_idx]
        ee_pose_base = ee_to_base(sim_ee_pos, sim_ee_mat)
        # print("sim ee pos:", sim_ee_pos)
        # print("EE ERROR:", np.linalg.norm(sim_ee_pos - obs_ee_pos[count][:3]))
        curr_sim_time = data.time
        data.ctrl[:7] = joint_actions[count]
        last_qpos = data.qpos[:7]
        while data.time < curr_sim_time + TIMESTEP:
            mujoco.mj_step(model, data)
            
            viewer.sync()
            sim_last_ee_pos = sim_curr_ee_pos.copy()
            sim_curr_ee_pos = data.site_xpos[ee_idx].copy()
        obs_last_ee_pos = obs_curr_ee_pos
        obs_curr_ee_pos = obs_joint_pos[count][:7]

        # if np.sum(np.abs(obs_curr_ee_pos[:3] - obs_last_ee_pos[:3])) > 0.2:
        #     print("================== BIG OBS DIFF ====================")
        #     print("obs last ee pos:", obs_last_ee_pos)
        #     print("obs curr ee pos:", obs_curr_ee_pos)
        #     print("diff:", np.sum(np.abs(obs_curr_ee_pos[:3] - obs_last_ee_pos[:3])))
        #     print("count:", count)
        # if np.sum(np.abs(data.qpos[:7] - last_qpos[:7])) > 0.1:
        #     print("================== BIG JOINT DIFF ====================")
        #     print("last qpos:", last_qpos[:7])
        #     print("sim joint pos:", data.qpos[:7])
        #     print("diff:", np.sum(np.abs(data.qpos[:7] - obs_joint_pos[count][:7])))
        #     print("count:", count)
        # if count > 1 and np.sum(np.abs(joint_actions[count] - joint_actions[count - 1])) > 0.5:
        #     print("================== BIG ACTION DIFF ====================")
        #     print("last action:", joint_actions[count - 1])
        #     print("curr action:", joint_actions[count]) 
        #     print("diff:", np.sum(np.abs(joint_actions[count] - joint_actions[count - 1])))
        #     print("count:", count)

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