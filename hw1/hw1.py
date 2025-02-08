import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def create_transition_matrices() -> List[np.ndarray]:
    """
    Create transition matrices for each layer of the neural network.
    Returns a list of transition matrices for each layer transition.
    """
    # Transition from layer 1 (11,12) to layer 2 (21,22,23)
    P1 = np.array([
        [0.5, 0.5, 0.0],  # From 11 to 21,22,23
        [0.0, 0.5, 0.5]   # From 12 to 21,22,23
    ])
    
    # Transition from layer 2 (21,22,23) to layer 3 (31,32)
    P2 = np.array([
        [0.5, 0.5],  # From 21 to 31,32
        [0.5, 0.5],  # From 22 to 31,32
        [0.5, 0.5]   # From 23 to 31,32
    ])
    
    # Transition from layer 3 (31,32) to layer 4 (41,42,43)
    P3 = np.array([
        [1/3, 1/3, 1/3],  # From 31 to 41,42,43
        [1/3, 1/3, 1/3]   # From 32 to 41,42,43
    ])
    
    return [P1, P2, P3]

def calculate_final_probabilities(pi0: np.ndarray) -> np.ndarray:
    """
    Calculate the final probabilities given initial probabilities.
    
    Args:
        pi0: Initial probability distribution [P(11), P(12)]
    
    Returns:
        Final probability distribution [P(41), P(42), P(43)]
    """
    matrices = create_transition_matrices()
    
    # Calculate through each layer
    pi1 = pi0 @ matrices[0]  # Layer 1 to 2
    pi2 = pi1 @ matrices[1]  # Layer 2 to 3
    pi3 = pi2 @ matrices[2]  # Layer 3 to 4
    
    return pi3

def monte_carlo_simulation(
    pi0: np.ndarray,
    n_simulations: int = 100000
) -> np.ndarray:
    """
    Perform Monte Carlo simulation of the random walk.
    
    Args:
        pi0: Initial probability distribution [P(11), P(12)]
        n_simulations: Number of simulations to run
    
    Returns:
        Estimated probabilities [P(41), P(42), P(43)]
    """
    # Define possible states at each layer
    layer1 = np.array([11, 12])
    layer2 = np.array([21, 22, 23])
    layer3 = np.array([31, 32])
    layer4 = np.array([41, 42, 43])
    
    # Initialize counters for final states
    final_counts = {41: 0, 42: 0, 43: 0}
    
    for _ in range(n_simulations):
        # Choose initial state
        current = np.random.choice(layer1, p=pi0)
        
        # Layer 1 to 2
        if current == 11:
            current = np.random.choice(layer2[:2])  # Can only go to 21 or 22
        else:  # current == 12
            current = np.random.choice(layer2[1:])  # Can only go to 22 or 23
        
        # Layer 2 to 3
        current = np.random.choice(layer3)  # Can go to either 31 or 32
        
        # Layer 3 to 4
        current = np.random.choice(layer4)  # Can go to 41, 42, or 43
        
        final_counts[current] += 1
    
    # Convert counts to probabilities
    probs = np.array([final_counts[41], final_counts[42], final_counts[43]]) / n_simulations
    return probs

def calculate_l2_distance_vs_simulations(
    pi0: np.ndarray,
    true_pi4: np.ndarray,
    max_simulations: int = 100000,
    num_points: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate L2 distance between Monte Carlo estimates and true probabilities
    for different numbers of simulations.
    
    Returns:
        Tuple of (simulation_counts, l2_distances)
    """
    # Generate logarithmically spaced numbers of simulations
    simulation_counts = np.logspace(1, np.log10(max_simulations), num_points, dtype=int)
    l2_distances = []
    
    for n_sim in simulation_counts:
        estimated_pi4 = monte_carlo_simulation(pi0, n_sim)
        l2_distance = np.sqrt(np.sum((true_pi4 - estimated_pi4) ** 2))
        l2_distances.append(l2_distance)
    
    return simulation_counts, np.array(l2_distances)

def plot_l2_convergence(pi0: np.ndarray, title: str):
    """Plot L2 distance convergence for given initial distribution."""
    true_pi4 = calculate_final_probabilities(pi0)
    sim_counts, l2_distances = calculate_l2_distance_vs_simulations(pi0, true_pi4)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(sim_counts, l2_distances, 'b-', label='L2 Distance')
    plt.loglog(sim_counts, 1/np.sqrt(sim_counts), 'r--', 
              label='1/√n (Expected Rate)')
    
    plt.xlabel('Number of Simulations')
    plt.ylabel('L2 Distance')
    plt.title(f'L2 Distance Convergence\n{title}')
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    # Initial distributions to test
    initial_distributions = [
        (np.array([0.5, 0.5]), "π₀ = [0.5, 0.5]"),
        (np.array([0.4, 0.6]), "π₀ = [0.4, 0.6]"),
        (np.array([0.1, 0.9]), "π₀ = [0.1, 0.9]")
    ]
    
    print("Analytical Results:")
    print("-" * 50)
    for pi0, desc in initial_distributions:
        pi4 = calculate_final_probabilities(pi0)
        print(f"\n{desc}:")
        print(f"π₄ = [{', '.join(f'{p:.4f}' for p in pi4)}]")
        
        # Plot L2 convergence
        plot_l2_convergence(pi0, desc)
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    main()