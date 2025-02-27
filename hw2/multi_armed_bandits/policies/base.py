from abc import ABC, abstractmethod
import numpy as np

class BasePolicy(ABC):
    """
    Base class for all multi-armed bandit policies.
    
    This abstract class defines the interface that all policies should implement.
    """
    
    def __init__(self, n_arms, horizon=None, seed=None):
        """
        Initialize the policy.
        
        Args:
            n_arms (int): Number of arms in the bandit.
            horizon (int, optional): Time horizon for the policy.
            seed (int, optional): Random seed for reproducibility.
        """
        self.n_arms = n_arms
        self.horizon = horizon
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            
        # Initialize counters and estimates
        self.t = 0  # Current time step
        self.pulls = np.zeros(n_arms, dtype=int)  # Number of pulls for each arm
        self.rewards = np.zeros(n_arms)  # Cumulative reward for each arm
        self.estimates = np.zeros(n_arms)  # Estimated mean reward for each arm
        
    def update(self, arm, reward):
        """
        Update the policy based on the observed reward.
        
        Args:
            arm (int): The arm that was pulled.
            reward (float): The reward that was observed.
        """
        self.t += 1
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
        # Update estimate of the arm's expected reward
        if self.pulls[arm] > 0:
            self.estimates[arm] = self.rewards[arm] / self.pulls[arm]
    
    @abstractmethod
    def select_arm(self):
        """
        Select which arm to pull next.
        
        Returns:
            int: The index of the selected arm.
        """
        pass
    
    def reset(self):
        """
        Reset the policy to its initial state.
        """
        self.t = 0
        self.pulls = np.zeros(self.n_arms, dtype=int)
        self.rewards = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms)