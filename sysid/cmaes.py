from cma import CMA
import h5py
import os
import mujoco
import time
from mujoco import MjModel, MjData
import numpy as np
from datetime import datetime

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
        L = 0
        for filename in os.listdir('./data'):
            ds_filepath = os.path.join('./data', filename)
            file = h5py.File(ds_filepath, 'r')
            joint_actions = file["joint_action"]
            obs_joint_pos = file["/observations/full_joint_pos"]
            timestamps = file["timestamp"]

            L += len(joint_actions)
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

        return score / L
    
    curr_stiffness = STIFFNESS_INIT
    curr_damping = DAMPING_INIT
    curr_state = np.vstack([curr_stiffness, curr_damping])
    lower_bound = np.vstack([STIFFNESS_LOW, DAMPING_LOW])
    upper_bound =  np.vstack([DAMPING_LOW, DAMPING_HIGH])

    cma = CMA(
        initial_solution=curr_state,
        initial_step_size=1.0,
        fitness_function=evaluate_trajectory,
        enforce_bounds=[lower_bound, upper_bound]
    )
    print("starting search")
    best_solution, best_fitness = cma.search()
    print("search done")
    # Create timestamp for unique logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)
    
    # Save results to txt file
    log_path = os.path.join("./logs", f"cma_results_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write(f"CMA-ES Optimization Results\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Best Fitness Score: {best_fitness}\n\n")
        f.write("Best Solution:\n")
        f.write(f"Stiffness: {best_solution[0].tolist()}\n")
        f.write(f"Damping: {best_solution[1].tolist()}\n")
    
    print(f"Results saved to: {log_path}")