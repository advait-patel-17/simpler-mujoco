from cma import CMA
import h5py
import os
import mujoco
import time
from mujoco import MjModel, MjData
import numpy as np

TIMESTEP = 0.1
ENERGY_SCALE = 0.002
# STIFFNESS_HIGH = np.array([600, 600, 600, 600, 600, 600, 4000])
# STIFFNESS_LOW = np.ones((7,))
# DAMPING_HIGH = np.array([200, 200, 200, 200, 200, 200, 200])
# DAMPING_LOW = np.zeros((7,))
STIFFNESS_HIGH = np.array([400, 400, 400, 400, 400, 400, 2500])
STIFFNESS_LOW = np.ones((7,))
DAMPING_HIGH = np.array([100, 100, 100, 100, 100, 100, 100])
DAMPING_LOW = np.zeros((7,))

STIFFNESS_INIT = np.array([  43,   265,   227,    78,    37,    10.4, 2000 ])
DAMPING_INIT = np.array([[ 5.76, 20,   18.49,  6.78,  6.28,  1.2,  40  ]])
# note -> model.actuator_gainprm[i, 0] for stiffness, model.dof_damping[i] for damping


def main():
    start = time.time()
    model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
    data = MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)


    def evaluate_trajectory(prms):
        def loss(x, xp):
            err = np.linalg.norm(x - xp)
            return err

        score = 0

        for filename in os.listdir('./data'):
            ds_filepath = os.path.join('./data', filename)
            file = h5py.File(ds_filepath, 'r')
            joint_actions = file["joint_action"]
            obs_joint_pos = file["/observations/full_joint_pos"]
            timestamps = file["timestamp"]


            mujoco.mj_resetDataKeyframe(model, data, 0)
            model.actuator_gainprm[:7, 0] = prms[0]
            model.dof_damping[:7] = prms[1]

            data.qpos[:7] = obs_joint_pos[0][:7]
            mujoco.mj_step(model, data)

            for i, act in enumerate(joint_actions):
                err = loss(obs_joint_pos[i][:7], data.qpos[:7])
                # print("data qpos:", data.qpos[:7])
                # print("Current observed joint position:", obs_joint_pos[i][:7])
                # print("error:",err)
                score += err
                curr_sim_time = data.time
                data.ctrl[:7] = act
                while data.time < curr_sim_time + TIMESTEP:
                    mujoco.mj_step(model, data)

        return score
