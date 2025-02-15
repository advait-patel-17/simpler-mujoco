import mujoco
from mujoco import MjModel, MjData
import numpy as np

import h5py
from simulated_annealing import simulated_annealing

import matplotlib.pyplot as plt
import json
import os
from datetime import datetime


TIMESTEP = 0.1
ENERGY_SCALE = 0.01
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

def generate_uniform_parameters():
    # Generate random values between 0 and 1
    random_values = np.random.random(7)
    
    # Scale and shift the values to be between low and high bounds
    stiffness = STIFFNESS_LOW + (STIFFNESS_HIGH - STIFFNESS_LOW) * random_values
    
    # Generate new random values for damping
    random_values = np.random.random(7)
    damping = DAMPING_LOW + (DAMPING_HIGH - DAMPING_LOW) * random_values
    result = np.vstack((stiffness, damping))
    return result

def generate_gaussian_params(current_state):
    """
    Generate random parameters using truncated normal distribution centered around current state.
    
    Args:
        current_state: Current parameter values, shape (2,7) where row 0 is stiffness, row 1 is damping
    
    Returns:
        New parameter values with same shape as current_state
    """
    std_fraction = 0.15  # Controls spread of distribution
    
    def truncated_normal(current_value, low, high):
        # Use current value as mean
        std = (high - low) * std_fraction
        
        while True:
            value = np.random.normal(current_value, std)
            if low <= value <= high:
                return value
    
    # Split current state into stiffness and damping
    current_stiffness = current_state[0]
    current_damping = current_state[1]
    
    # Generate new stiffness parameters
    stiffness = np.array([
        truncated_normal(current, low, high) 
        for current, low, high in zip(current_stiffness, STIFFNESS_LOW, STIFFNESS_HIGH)
    ])
    
    # Generate new damping parameters
    damping = np.array([
        truncated_normal(current, low, high)
        for current, low, high in zip(current_damping, DAMPING_LOW, DAMPING_HIGH)
    ])
    
    return np.vstack((stiffness, damping))

def log_annealing_results(best_state, best_energy, energy_history, output_dir="./logs"):
    """
    Log the results of simulated annealing optimization.
    
    Args:
        best_state (np.ndarray): Best parameters found [stiffness, damping]
        best_energy (float): Best score achieved
        energy_history (list): History of energy values throughout optimization
        output_dir (str): Directory to save logs
    """
    # Create timestamp for unique logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numerical results
    results = {
        "timestamp": timestamp,
        "best_stiffness": best_state[0].tolist(),
        "best_damping": best_state[1].tolist(),
        "best_energy": float(best_energy),
        "energy_history": energy_history
    }
    
    # Save JSON results
    json_path = os.path.join(output_dir, f"annealing_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # Create visualization of optimization progress
    plt.figure(figsize=(12, 6))
    plt.plot(energy_history)
    plt.title("Simulated Annealing Optimization Progress")
    plt.xlabel("Iteration")
    plt.ylabel("Energy (Loss)")
    plt.yscale('log')  # Using log scale as optimization values often span orders of magnitude
    plt.grid(True)
    
    # Save plot
    plot_path = os.path.join(output_dir, f"annealing_progress_{timestamp}.png")
    plt.savefig(plot_path)
    plt.close()
    
    # Create parameter visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot stiffness parameters
    joint_indices = range(len(best_state[0]))
    ax1.bar(joint_indices, best_state[0])
    ax1.set_title("Optimized Stiffness Parameters")
    ax1.set_xlabel("Joint Index")
    ax1.set_ylabel("Stiffness")
    ax1.grid(True)
    
    # Plot damping parameters
    ax2.bar(joint_indices, best_state[1])
    ax2.set_title("Optimized Damping Parameters")
    ax2.set_xlabel("Joint Index")
    ax2.set_ylabel("Damping")
    ax2.grid(True)
    
    # Save parameter plot
    params_plot_path = os.path.join(output_dir, f"annealing_parameters_{timestamp}.png")
    plt.tight_layout()
    plt.savefig(params_plot_path)
    plt.close()
    
    print(f"Results saved to {output_dir}:")
    print(f"- JSON results: annealing_results_{timestamp}.json")
    print(f"- Progress plot: annealing_progress_{timestamp}.png")
    print(f"- Parameter plot: annealing_parameters_{timestamp}.png")
    
    return {
        "json_path": json_path,
        "progress_plot_path": plot_path,
        "parameter_plot_path": params_plot_path
    }


def main():
    model = MjModel.from_xml_path("./aloha/robolab_setup.xml")
    data = MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)




    def evaluate_trajectory(prms):
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
                score += err
                curr_sim_time = data.time
                data.ctrl[:7] = act
                while data.time < curr_sim_time + TIMESTEP:
                    mujoco.mj_step(model, data)

        return score*ENERGY_SCALE
    
    curr_stiffness = STIFFNESS_INIT
    curr_damping = DAMPING_INIT
    curr_state = np.vstack([curr_stiffness, curr_damping])

    best_state, best_energy, energy_history = simulated_annealing(initial_state=curr_state, evaluate_fn=evaluate_trajectory, neighbor_fn=generate_gaussian_params)

    # After simulated annealing
    log_paths = log_annealing_results(best_state, best_energy, energy_history)
    print(log_paths)

    
if __name__ == "__main__":
    main()