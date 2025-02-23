from cma import CMA
import h5py
import os
import mujoco
import time
from mujoco import MjModel, MjData
import numpy as np
from datetime import datetime
import tensorflow as tf

import pickle

TIMESTEP = 0.1
# STIFFNESS_HIGH = np.array([600, 600, 600, 600, 600, 600, 4000])
# STIFFNESS_LOW = np.ones((7,))
# DAMPING_HIGH = np.array([200, 200, 200, 200, 200, 200, 200])
# DAMPING_LOW = np.zeros((7,))
STIFFNESS_HIGH = np.array([400, 400, 400, 400, 400, 400, 2500])
STIFFNESS_LOW = np.ones((7,))
DAMPING_HIGH = np.array([100, 100, 100, 100, 100, 100, 100])
DAMPING_LOW = np.zeros((7,))

STIFFNESS_INIT = np.array([  43,   265,   227,    78,    37,    10.4, 2000 ])
DAMPING_INIT = np.array([5.76, 20, 18.49, 6.78, 6.28, 1.2, 40])
# note -> model.actuator_gainprm[i, 0] for stiffness, model.dof_damping[i] for damping
def save_trace_numpy(trace, filename):
    """
    Save CMA trace to a .npz file using numpy.
    
    Args:
        trace: List of dictionaries containing CMA trace data
        filename: String, path to save the file (will append .npz if not present)
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    # Convert list of dicts to dict of lists for easier saving
    trace_arrays = {
        'm': np.array([t['m'] for t in trace]),
        'sigma': np.array([t['σ'] for t in trace]),
        'C': np.array([t['C'] for t in trace]),
        'p_sigma': np.array([t['p_σ'] for t in trace]),
        'p_C': np.array([t['p_C'] for t in trace]),
        'B': np.array([t['B'] for t in trace]),
        'D': np.array([t['D'] for t in trace]),
        'population': np.array([t['population'] for t in trace])
    }
    
    np.savez(filename, **trace_arrays)

def load_trace_numpy(filename):
    """
    Load CMA trace from a .npz file.
    
    Args:
        filename: String, path to the .npz file
        
    Returns:
        List of dictionaries containing CMA trace data
    """
    if not filename.endswith('.npz'):
        filename += '.npz'
    
    data = np.load(filename)
    
    # Convert back to list of dicts format
    trace = []
    for i in range(len(data['m'])):
        trace.append({
            'm': data['m'][i],
            'σ': data['sigma'][i],
            'C': data['C'][i],
            'p_σ': data['p_sigma'][i],
            'p_C': data['p_C'][i],
            'B': data['B'][i],
            'D': data['D'][i],
            'population': data['population'][i]
        })
    
    return trace

def main():
    start = time.time()
    model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
    data = MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)

    def evaluate_population(population):
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
                model.actuator_gainprm[:7, 0] = prms[:7]
                model.dof_damping[:7] = prms[7:]

                data.qpos[:7] = obs_joint_pos[0][:7]
                mujoco.mj_step(model, data)

                for i, act in enumerate(joint_actions):
                    err = loss(obs_joint_pos[i][:7], data.qpos[:7])
                    score += err
                    curr_sim_time = data.time
                    data.ctrl[:7] = act
                    while data.time < curr_sim_time + TIMESTEP:
                        mujoco.mj_step(model, data)

            return score / L
        res = np.zeros(len(population))
        for i, prm in enumerate(population):
            res[i] = evaluate_trajectory(prm)
        res = tf.convert_to_tensor(res, dtype=tf.float32)
        return res
    
    max_epochs = 1000

    def logging_function(cma, logger):
        if cma.generation % 10 == 0:
            fitness = cma.best_fitness()
            logger.info(f'Generation {cma.generation} - fitness {fitness}')

        if cma.termination_criterion_met or cma.generation == max_epochs:
            sol = cma.best_solution()
            fitness = cma.best_fitness()
            logger.info(f'Final solution at gen {cma.generation}: {sol} (fitness: {fitness})')
    
    curr_stiffness = STIFFNESS_INIT
    curr_damping = DAMPING_INIT
    curr_state = np.vstack([curr_stiffness, curr_damping]).flatten()
    bounds = np.array([[STIFFNESS_LOW[i], STIFFNESS_HIGH[i]] for i in range(7)] + 
                      [[DAMPING_LOW[i], DAMPING_HIGH[i]] for i in range(7)])
    print("Current state shape:", curr_state.shape)

    cma = CMA(
        initial_solution=curr_state,
        initial_step_size=50.0,
        fitness_function=evaluate_population,
        enforce_bounds=bounds,
        callback_function=logging_function,
        store_trace=True
    )
    
    print("STARTING SEARCH")

    best_solution, best_fitness = cma.search(max_epochs)

    print("search done")
    # Create timestamp for unique logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create logs directory if it doesn't exist
    os.makedirs("./logs_cmaes", exist_ok=True)
    
    # Save results to txt file
    log_path = os.path.join("./logs_cmaes", f"cma_results_{timestamp}.txt")
    with open(log_path, "w") as f:
        f.write(f"CMA-ES Optimization Results\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Best Fitness Score: {best_fitness}\n\n")
        f.write("Best Solution:\n")
        f.write(f"Stiffness: {best_solution[:7].tolist()}\n")
        f.write(f"Damping: {best_solution[7:].tolist()}\n")

        # To save the trace to a file:
    with open("cma_trace.pkl", "wb") as f:
        pickle.dump(cma.trace, f)

    print(f"Results saved to: {log_path}")

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Run the CMA-ES optimization
    logger.info("Starting CMA-ES optimization...")
    main()
    logger.info("CMA-ES optimization completed.")
