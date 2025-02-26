import mujoco
import mujoco.viewer
from mujoco import MjData, MjModel
import h5py
import numpy as np

# Loading a variant of the model, e.g. panda without a gripper.
model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
data = mujoco.MjData(model)

ds_filepath = "./data/episode_0.hdf5"
file = h5py.File(ds_filepath, 'r')
joint_actions = file["joint_action"]
obs_joint_pos = file["/observations/full_joint_pos"]
obs_ee_pos = file["/observations/ee_pos"]
timestamps = file["timestamp"]
print(obs_joint_pos[0])


T_base_world = np.array([
    [-1.0,  0.0,  0.0,  0.525],
    [ 0.0, -1.0,  0.0, -0.019],
    [ 0.0,  0.0,  1.0,  0.02 ],
    [ 0.0,  0.0,  0.0,  1.0  ]
])


data.qpos[:7] = obs_joint_pos[0][:7]
mujoco.mj_step(model, data)


#ee pos in world coord
ee_pos = np.array([0.24353877,  -0.019 , 0.32524417, 1])
# ee pos in robot base
ee_pos_rob = np.linalg.inv(T_base_world) @ ee_pos
print("ee pos in robot:", ee_pos_rob)


# Print the first 4 matrices from robot_pase_pose_in_world
robot_poses = file["/observations/ee_pos"]
for i in range(min(4, len(robot_poses))):
    print(f"Matrix {i}:")
    print(np.array(robot_poses[i]))
    print()