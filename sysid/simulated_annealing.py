import numpy as np

def generate_neighbor_random(state):
    """Generate neighbor by random perturbation"""
    neighbor = state + np.random.uniform(-0.1, 0.1, size=state.shape)
    return np.clip(neighbor, 0, 1)

def generate_neighbor_normal(state):
    """Generate neighbor using normal distribution"""
    neighbor = state + np.random.normal(0, 0.1, size=state.shape)
    return np.clip(neighbor, 0, 1)

def simulated_annealing(initial_state, evaluate_fn, max_iterations=1000, 
                       initial_temp=1.0, cooling_rate=0.95, 
                       neighbor_type='random'):
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
    current_state = initial_state.copy()
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
