import mujoco
from mujoco import MjModel, MjData
import numpy as np

import h5py
from simulated_annealing import generate_neighbor_normal, generate_neighbor_random, simulated_annealing

TIMESTEP = 0.1

STIFFNESS_HIGH = np.array([600, 600, 600, 600, 600, 600, 4000])
STIFFNESS_LOW = np.ones((7,))
DAMPING_HIGH = np.array([200, 200, 200, 200, 200, 200, 200])
DAMPING_LOW = np.zeros((7,))
STIFFNESS_INIT = np.array([  43,   265,   227,    78,    37,    10.4, 2000 ])
DAMPING_INIT = np.array([[ 5.76, 20,   18.49,  6.78,  6.28,  1.2,  40  ]])
# note -> model.actuator_gainprm[i, 0] for stiffness, model.dof_damping[i] for damping

def loss(x, xp):
    err = np.linalg.norm(x - xp)
    return err

def map_vec_to_model(vec):



def main():
    model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
    data = MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:7] = obs_joint_pos[0][:7]
    mujoco.mj_step(model, data)


    ds_filepath = "./data/episode_1.hdf5"
    file = h5py.File(ds_filepath, 'r')
    joint_actions = file["joint_action"]
    obs_joint_pos = file["/observations/full_joint_pos"]
    timestamps = file["timestamp"]

    # write "evaluate trajectory" fn (return error between simulated trajectory and ground truth)
    # plug function into simulated_annealing fn

    def evaluate_trajectory(neighbor):
        score = 0
        #write kp/kd values to thing
        for i, act in enumerate(joint_actions):
            err = loss(obs_joint_pos[i][:7], data.qpos[:7])
            score += err
            curr_sim_time = data.time
            data.ctrl[:7] = act
            while data.time < curr_sim_time + TIMESTEP:
                mujoco.mj_step(model, data)
