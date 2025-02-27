import numpy as np
from .base import BasePolicy

class EpsilonGreedy(BasePolicy):
    """
    Epsilon-Greedy policy implementation.
    
    This policy selects the arm with the highest estimated reward with probability 1-epsilon,
    and explores a random arm with probability epsilon. Epsilon can decay over time.
    """
    
    def __init__(self, n_arms, c=150, min_epsilon=0, seed=None):
        """
        Initialize the Epsilon-Greedy policy.
        
        Args:
            n_arms (int): Number of arms in the bandit.
            c (float, optional): Parameter for epsilon scheduling.
                Higher values lead to more exploration.
            min_epsilon (float, optional): Minimum value for epsilon.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(n_arms, seed=seed)
        self.c = c
        self.min_epsilon = min_epsilon
        
    def get_epsilon(self):
        """
        Calculate the current epsilon value.
        
        Returns:
            float: The current epsilon value in [0, 1].
        """
        # Epsilon decays as 1/t
        if self.t == 0:
            return 1.0  # Always explore in the first round
        
        epsilon = min(1.0, self.c / (self.t + 1))
        return max(epsilon, self.min_epsilon)
        
    def select_arm(self):
        """
        Select an arm according to the epsilon-greedy policy.
        
        With probability 1-epsilon, selects the arm with highest estimated reward.
        With probability epsilon, selects a random arm.
        
        Returns:
            int: The index of the selected arm.
        """
        # Get current epsilon
        epsilon = self.get_epsilon()
        
        # Explore: select a random arm with probability epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.n_arms)
        
        # Exploit: select arm with highest estimated reward
        # In case of ties, break randomly
        max_value = np.max(self.estimates)
        max_indices = np.where(self.estimates == max_value)[0]
        return np.random.choice(max_indices)