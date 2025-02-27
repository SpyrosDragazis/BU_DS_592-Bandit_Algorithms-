import numpy as np
from policies.epsilon_greedy import EpsilonGreedy

class BanditGame:
    """
    A class to simulate a multi-armed bandit game.
    
    This class orchestrates the interaction between a bandit environment and a policy,
    and collects statistics about the game.
    
    Attributes:
        env (environment.GaussianBandit): The bandit environment.
        policy (BasePolicy): The policy used to select arms.
        horizon (int): The time horizon for the game.
        rewards (list): The rewards collected in each round.
        regrets (list): The regrets incurred in each round.
        arms_pulled (list): The arms pulled in each round.
    """
    
    def __init__(self, env, policy, horizon=None):
        """
        Initialize a BanditGame.
        
        Args:
            env (environment.GaussianBandit): The bandit environment.
            policy (BasePolicy): The policy used to select arms.
            horizon (int, optional): The time horizon for the game.
                If None, uses the policy's horizon.
        """
        self.env = env
        self.policy = policy
        
        # Set horizon
        if horizon is None:
            if policy.horizon is None:
                raise ValueError("Horizon must be specified either in the policy or the game.")
            self.horizon = policy.horizon
        else:
            self.horizon = horizon
            
        # Initialize statistics
        self.rewards = []
        self.regrets = []
        self.cumulative_regret = []
        self.arms_pulled = []
        
    def play_round(self):
        """
        Play a single round of the bandit game.
        
        Returns:
            tuple: (arm_pulled, reward, regret)
        """
        # Select arm according to policy
        arm = self.policy.select_arm()
        
        # Pull the arm and observe reward
        reward = self.env.pull(arm)
        
        # Calculate regret
        regret = self.env.regret(arm)
        
        # Update policy with the observed reward
        self.policy.update(arm, reward)
        
        return arm, reward, regret
    
    def run(self):
        """
        Run the bandit game for the specified horizon.
        
        Returns:
            tuple: (total_reward, total_regret, average_reward, average_regret)
        """
        # Reset statistics
        self.rewards = []
        self.regrets = []
        self.cumulative_regret = []
        self.arms_pulled = []
        
        total_regret = 0
        
        # Play for specified number of rounds
        for _ in range(self.horizon):
            arm, reward, regret = self.play_round()
            
            self.rewards.append(reward)
            self.regrets.append(regret)
            total_regret += regret
            self.cumulative_regret.append(total_regret)
            self.arms_pulled.append(arm)
            
        total_reward = sum(self.rewards)
        avg_reward = total_reward / self.horizon
        avg_regret = total_regret / self.horizon
        
        return total_reward, total_regret, avg_reward, avg_regret
    
    def reset(self):
        """
        Reset the game and policy to their initial states.
        """
        self.policy.reset()
        self.rewards = []
        self.regrets = []
        self.cumulative_regret = []
        self.arms_pulled = []


def simulate_policy(env, policy_class, policy_params, horizon, n_simulations=100, seed=None):
    """
    Run multiple simulations of a policy on a bandit environment.
    
    Args:
        env (environment.GaussianBandit): The bandit environment.
        policy_class: The policy class to use.
        policy_params (dict): Parameters for the policy.
        horizon (int): The time horizon for each simulation.
        n_simulations (int, optional): The number of simulations to run.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        tuple: (avg_regret, std_error, all_regrets)
            avg_regret (float): Average total regret across simulations.
            std_error (float): Standard error of the regret.
            all_regrets (list): List of cumulative regret trajectories.
    """
    if seed is not None:
        np.random.seed(seed)
    
    total_regrets = []
    all_regrets = []
    
    for sim in range(n_simulations):
        # Create a new policy instance with a different seed
        sim_seed = None if seed is None else seed + sim
        
        # Check if the policy class is EpsilonGreedy, which doesn't need horizon
        if policy_class == EpsilonGreedy:
            policy = policy_class(n_arms=env.n_arms, seed=sim_seed, **policy_params)
        else:
            policy = policy_class(n_arms=env.n_arms, horizon=horizon, seed=sim_seed, **policy_params)
        
        # Run the game
        game = BanditGame(env, policy, horizon)
        _, total_regret, _, _ = game.run()
        
        total_regrets.append(total_regret)
        all_regrets.append(game.cumulative_regret)
    
    # Calculate statistics
    avg_regret = np.mean(total_regrets)
    std_error = np.std(total_regrets) / np.sqrt(n_simulations)
    
    return avg_regret, std_error, all_regrets