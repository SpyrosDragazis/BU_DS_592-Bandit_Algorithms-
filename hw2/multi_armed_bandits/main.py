import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from policies import ExploreThenCommit, EpsilonGreedy, SuccessiveElimination
from environment import GaussianBandit
from game import simulate_policy

# Constants
HORIZON = 10000
N_SIMULATIONS = 100
DELTA_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEED = 42

# Create output directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')


def run_simulations():
    """
    Run all simulations and save results.
    """
    # Store results
    results = {
        'ETC': {'regrets': [], 'std_errors': [], 'theoretical': []},
        'SE': {'regrets': [], 'std_errors': [], 'theoretical': []},
        'EG': {'regrets': [], 'std_errors': [], 'theoretical': []}
    }
    
    # Run simulations for each delta value
    for delta in DELTA_VALUES:
        print(f"Running simulations for delta = {delta}")
        
        # Create environment
        env = GaussianBandit.create_two_armed_instance(delta)
        
        # Explore Then Commit
        etc_params = {}
        avg_regret, std_error, _ = simulate_policy(
            env, ExploreThenCommit, etc_params, HORIZON, N_SIMULATIONS, SEED)
        results['ETC']['regrets'].append(avg_regret)
        results['ETC']['std_errors'].append(std_error)
        
        # Calculate theoretical bound for ETC
        m = int(np.ceil(HORIZON**(2/3)))
        theoretical_etc = delta * m + delta * (HORIZON - m) * np.exp(-m * delta**2 / 4)
        results['ETC']['theoretical'].append(theoretical_etc)
        
        # Successive Elimination
        se_params = {}
        avg_regret, std_error, _ = simulate_policy(
            env, SuccessiveElimination, se_params, HORIZON, N_SIMULATIONS, SEED)
        results['SE']['regrets'].append(avg_regret)
        results['SE']['std_errors'].append(std_error)
        
        # Calculate theoretical bound for SE
        k = 2  # Number of arms
        theoretical_se = np.sqrt(k * HORIZON * np.log(HORIZON))
        results['SE']['theoretical'].append(theoretical_se)
        
        # Epsilon Greedy
        eg_params = {'c': 50}
        avg_regret, std_error, _ = simulate_policy(
            env, EpsilonGreedy, eg_params, HORIZON, N_SIMULATIONS, SEED)
        results['EG']['regrets'].append(avg_regret)
        results['EG']['std_errors'].append(std_error)
        
        # Calculate theoretical bound for EG
        c = 50
        theoretical_eg = c * delta + delta * HORIZON / c
        results['EG']['theoretical'].append(theoretical_eg)
    
    # Save results to file
    np.savez('results/simulation_results.npz', 
             delta_values=DELTA_VALUES, 
             etc_regrets=results['ETC']['regrets'],
             etc_std_errors=results['ETC']['std_errors'],
             etc_theoretical=results['ETC']['theoretical'],
             se_regrets=results['SE']['regrets'],
             se_std_errors=results['SE']['std_errors'],
             se_theoretical=results['SE']['theoretical'],
             eg_regrets=results['EG']['regrets'],
             eg_std_errors=results['EG']['std_errors'],
             eg_theoretical=results['EG']['theoretical'])
    
    return results


def plot_results(results):
    """
    Plot the simulation results.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot empirical results with error bars
    plt.errorbar(DELTA_VALUES, results['ETC']['regrets'], yerr=results['ETC']['std_errors'], 
                 fmt='o-', capsize=3, label='Explore-Then-Commit (Empirical)', color='blue')
    
    plt.errorbar(DELTA_VALUES, results['SE']['regrets'], yerr=results['SE']['std_errors'], 
                 fmt='s-', capsize=3, label='Successive Elimination (Empirical)', color='green')
    
    plt.errorbar(DELTA_VALUES, results['EG']['regrets'], yerr=results['EG']['std_errors'], 
                 fmt='^-', capsize=3, label='ε-Greedy (Empirical)', color='red')
    
    # Plot theoretical bounds
    plt.plot(DELTA_VALUES, results['ETC']['theoretical'], '--', label='ETC (Theoretical)', color='blue', alpha=0.7)
    plt.plot(DELTA_VALUES, results['SE']['theoretical'], '--', label='SE (Theoretical)', color='green', alpha=0.7)
    plt.plot(DELTA_VALUES, results['EG']['theoretical'], '--', label='EG (Theoretical)', color='red', alpha=0.7)
    
    plt.xlabel('Gap (Δ)', fontsize=14)
    plt.ylabel('Expected Regret', fontsize=14)
    plt.title(f'Regret vs. Gap (Horizon = {HORIZON}, Simulations = {N_SIMULATIONS})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Format y-axis with scientific notation if values are large
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    
    plt.tight_layout()
    plt.savefig('results/regret_vs_gap.png', dpi=300)
    
    # Create log-log plot for better visualization
    plt.figure(figsize=(12, 8))
    
    plt.loglog(DELTA_VALUES, results['ETC']['regrets'], 'o-', label='Explore-Then-Commit (Empirical)', color='blue')
    plt.loglog(DELTA_VALUES, results['SE']['regrets'], 's-', label='Successive Elimination (Empirical)', color='green')
    plt.loglog(DELTA_VALUES, results['EG']['regrets'], '^-', label='ε-Greedy (Empirical)', color='red')
    
    plt.loglog(DELTA_VALUES, results['ETC']['theoretical'], '--', label='ETC (Theoretical)', color='blue', alpha=0.7)
    plt.loglog(DELTA_VALUES, results['SE']['theoretical'], '--', label='SE (Theoretical)', color='green', alpha=0.7)
    plt.loglog(DELTA_VALUES, results['EG']['theoretical'], '--', label='EG (Theoretical)', color='red', alpha=0.7)
    
    plt.xlabel('Gap (Δ)', fontsize=14)
    plt.ylabel('Expected Regret (log scale)', fontsize=14)
    plt.title(f'Regret vs. Gap (Log-Log Scale, Horizon = {HORIZON})', fontsize=16)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/regret_vs_gap_loglog.png', dpi=300)


def simulate_regret_trajectories():
    """
    Simulate and plot regret trajectories for a specific delta value.
    """
    delta = 0.2  # Choose a representative delta value
    env = GaussianBandit.create_two_armed_instance(delta)
    
    # Dictionary to store trajectories
    trajectories = {}
    
    # Explore Then Commit
    _, _, all_regrets_etc = simulate_policy(
        env, ExploreThenCommit, {}, HORIZON, 10, SEED)
    trajectories['ETC'] = np.mean(all_regrets_etc, axis=0)
    
    # Successive Elimination
    _, _, all_regrets_se = simulate_policy(
        env, SuccessiveElimination, {}, HORIZON, 10, SEED)
    trajectories['SE'] = np.mean(all_regrets_se, axis=0)
    
    # Epsilon Greedy
    _, _, all_regrets_eg = simulate_policy(
        env, EpsilonGreedy, {'c': 50}, HORIZON, 10, SEED)
    trajectories['EG'] = np.mean(all_regrets_eg, axis=0)
    
    # Plot trajectories
    plt.figure(figsize=(12, 8))
    
    plt.plot(range(1, HORIZON + 1), trajectories['ETC'], label='Explore-Then-Commit', color='blue')
    plt.plot(range(1, HORIZON + 1), trajectories['SE'], label='Successive Elimination', color='green')
    plt.plot(range(1, HORIZON + 1), trajectories['EG'], label='ε-Greedy', color='red')
    
    plt.xlabel('Time Step (t)', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.title(f'Cumulative Regret Trajectories (Δ = {delta})', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('results/regret_trajectories.png', dpi=300)


if __name__ == "__main__":
    print("Running Multi-Armed Bandit simulations...")
    results = run_simulations()
    plot_results(results)
    simulate_regret_trajectories()
    print("Simulations complete. Results saved in the 'results' directory.")