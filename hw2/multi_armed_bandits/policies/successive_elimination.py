import numpy as np
from .base import BasePolicy

class SuccessiveElimination(BasePolicy):
    """
    Successive Elimination policy implementation.
    
    This policy maintains a set of active arms and successively eliminates 
    suboptimal arms based on confidence bounds.
    """
    
    def __init__(self, n_arms, horizon, seed=None):
        """
        Initialize the Successive Elimination policy.
        
        Args:
            n_arms (int): Number of arms in the bandit.
            horizon (int): Time horizon for the policy.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(n_arms, horizon, seed)
        
        # Keep track of active arms
        self.active_arms = np.ones(n_arms, dtype=bool)
        self.n_active = n_arms
        
        # Current arm index for round-robin selection
        self.current_arm_idx = 0
    
    def get_bonus(self, arm):
        """
        Calculate the bonus term for the confidence bound.
        
        Args:
            arm (int): The arm index.
            
        Returns:
            float: The bonus term for the confidence bound.
        """
        if self.pulls[arm] == 0:
            return float('inf')
        
        # Use the formula from the assignment
        return np.sqrt(2 * np.log(self.horizon) / self.pulls[arm])
    
    def get_ucb(self, arm):
        """
        Calculate the Upper Confidence Bound (UCB) for an arm.
        
        Args:
            arm (int): The arm index.
            
        Returns:
            float: The UCB value.
        """
        return self.estimates[arm] + self.get_bonus(arm)
    
    def get_lcb(self, arm):
        """
        Calculate the Lower Confidence Bound (LCB) for an arm.
        
        Args:
            arm (int): The arm index.
            
        Returns:
            float: The LCB value.
        """
        return self.estimates[arm] - self.get_bonus(arm)
    
    def eliminate_arms(self):
        """
        Eliminate suboptimal arms based on confidence bounds.
        """
        if self.n_active <= 1:
            return
        
        # For each active arm, compare its UCB with the LCB of all other active arms
        for i in np.where(self.active_arms)[0]:
            ucb_i = self.get_ucb(i)
            
            # Check if there's any arm j such that LCB_j > UCB_i
            for j in np.where(self.active_arms)[0]:
                if i != j and self.get_lcb(j) > ucb_i:
                    # Arm i is suboptimal, eliminate it
                    self.active_arms[i] = False
                    self.n_active -= 1
                    break
    
    def select_arm(self):
        """
        Select an arm according to the Successive Elimination policy.
        
        During each round, plays each active arm once, and then eliminates
        arms that are determined to be suboptimal.
        
        Returns:
            int: The index of the selected arm.
        """
        # If all arms have been played at least once, eliminate suboptimal arms
        min_pulls = min(self.pulls[self.active_arms])
        if min_pulls > 0 and all(self.pulls[self.active_arms] == min_pulls):
            self.eliminate_arms()
        
        # If only one arm remains active, always select it
        if self.n_active == 1:
            return np.where(self.active_arms)[0][0]
        
        # Otherwise, choose the next active arm in a round-robin fashion
        active_indices = np.where(self.active_arms)[0]
        current_idx = self.current_arm_idx % len(active_indices)
        self.current_arm_idx += 1
        return active_indices[current_idx]
    
    def reset(self):
        """
        Reset the policy to its initial state.
        """
        super().reset()
        self.active_arms = np.ones(self.n_arms, dtype=bool)
        self.n_active = self.n_arms
        self.current_arm_idx = 0