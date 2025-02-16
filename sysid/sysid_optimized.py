import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

def evaluate_trajectory_vectorized(prms, model, data, file_data):
    """Vectorized version of trajectory evaluation"""
    score = 0
    joint_actions = file_data["joint_action"]
    obs_joint_pos = file_data["observations/full_joint_pos"]
    
    # Reset state
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.actuator_gainprm[:7, 0] = prms[0]
    model.dof_damping[:7] = prms[1]
    data.qpos[:7] = obs_joint_pos[0][:7]
    mujoco.mj_step(model, data)
    
    # Pre-allocate arrays for batch processing
    batch_size = 100  # Adjust based on memory constraints
    for i in range(0, len(joint_actions), batch_size):
        batch_slice = slice(i, min(i + batch_size, len(joint_actions)))
        batch_actions = joint_actions[batch_slice]
        batch_obs = obs_joint_pos[batch_slice]
        
        for act, obs in zip(batch_actions, batch_obs):
            err = np.linalg.norm(obs[:7] - data.qpos[:7])
            score += err
            curr_sim_time = data.time
            data.ctrl[:7] = act
            while data.time < curr_sim_time + TIMESTEP:
                mujoco.mj_step(model, data)
    
    return score * ENERGY_SCALE

def evaluate_parallel(prms, model_path, file_paths):
    """Parallel evaluation across multiple files"""
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)
    
    total_score = 0
    for filepath in file_paths:
        with h5py.File(filepath, 'r') as file:
            score = evaluate_trajectory_vectorized(prms, model, data, file)
            total_score += score
    return total_score

def simulated_annealing(initial_state, evaluate_fn, neighbor_fn, max_iterations=1000, 
                       initial_temp=1.0, cooling_rate=0.995, num_workers=None):
    """
    Optimized simulated annealing implementation
    """
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print("============ NUM_WORKERS:", num_workers)
    
    current_state = initial_state.copy()
    current_energy = evaluate_fn(current_state)
    best_state = current_state.copy()
    best_energy = current_energy
    temperature = initial_temp
    
    # Use numpy arrays for faster operations
    energy_history = np.zeros(max_iterations + 1)
    energy_history[0] = current_energy
    
    # Pre-compute temperature schedule
    temp_schedule = initial_temp * (cooling_rate ** np.arange(max_iterations))
    
    # Batch processing of neighbors
    batch_size = min(100, max_iterations)  # Adjust based on memory
    
    for batch_start in range(0, max_iterations, batch_size):
        batch_end = min(batch_start + batch_size, max_iterations)
        batch_temps = temp_schedule[batch_start:batch_end]
        
        # Generate batch of neighbors
        neighbors = [neighbor_fn(current_state) for _ in range(batch_end - batch_start)]
        
        # Parallel evaluation of neighbors
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            neighbor_energies = list(executor.map(evaluate_fn, neighbors))
        
        for i, (neighbor, neighbor_energy, temp) in enumerate(zip(neighbors, neighbor_energies, batch_temps)):
            idx = batch_start + i
            delta_E = neighbor_energy - current_energy
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / temp):
                current_state = neighbor.copy()
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_state = current_state.copy()
                    best_energy = current_energy
            
            energy_history[idx + 1] = current_energy
            
            # Early stopping check
            if temp < 1e-10:
                energy_history = energy_history[:idx + 2]  # Truncate history
                break
    
    return best_state, best_energy, energy_history.tolist()

def main():
    model_path = "./aloha/robolab_setup.xml"
    data_dir = './data'
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    
    # Partial function for parallel evaluation
    evaluate_partial = partial(evaluate_parallel, model_path=model_path, file_paths=file_paths)
    
    curr_state = np.vstack([STIFFNESS_INIT, DAMPING_INIT])
    
    best_state, best_energy, energy_history = simulated_annealing(
        initial_state=curr_state,
        evaluate_fn=evaluate_partial,
        neighbor_fn=generate_gaussian_params,
    )
    
    log_paths = log_annealing_results(best_state, best_energy, energy_history)
    print(log_paths)

if __name__ == "__main__":
    main()