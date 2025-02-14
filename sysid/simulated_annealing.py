import numpy as np
import time


def simulated_annealing(initial_state, evaluate_fn, neighbor_fn, max_iterations=10000, 
                       initial_temp=1.0, cooling_rate=0.995):
    """
    Simulated annealing optimization
    
    Parameters:
    - initial_state: 6x2 numpy array
    - evaluate_fn: function that takes a state and returns energy (lower is better)
    - max_iterations: maximum number of iterations
    - initial_temp: starting temperature
    - cooling_rate: rate at which temperature decreases
    - neighbor_type: 'random' or 'normal' for neighbor generation method
    """
    
    # Set up initial state
    current_state = initial_state.copy()    # write "evaluate trajectory" fn (takes in kp/kd state and return error between simulated trajectory and ground truth)
    # plug function into simulated_annealing fn

    current_energy = evaluate_fn(current_state)
    best_state = current_state.copy()
    best_energy = current_energy
    temperature = initial_temp

    
    # Select neighbor generation function
    if neighbor_type == 'random':
        generate_neighbor = generate_neighbor_random
    else:
        generate_neighbor = generate_neighbor_normal
    
    # Keep track of energies for visualization
    energy_history = [current_energy]
    
    for iteration in range(max_iterations):
        # Generate neighbor
        neighbor = generate_neighbor(current_state)
        neighbor_energy = evaluate_fn(neighbor)
        
        # Calculate energy difference
        delta_E = neighbor_energy - current_energy
        
        # Decide whether to accept neighbor
        if delta_E < 0 or np.random.random() < np.exp(-delta_E / temperature):
            current_state = neighbor.copy()
            current_energy = neighbor_energy
            
            # Update best solution if current is better
            if current_energy < best_energy:
                best_state = current_state.copy()
                best_energy = current_energy
        
        # Record energy
        energy_history.append(current_energy)
        
        # Cool temperature
        temperature *= cooling_rate
        # Optional early stopping
        if temperature < 1e-10:
            break

    
    return best_state, best_energy, energy_history
