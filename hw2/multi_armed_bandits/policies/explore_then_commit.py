import numpy as np
from .base import BasePolicy

class ExploreThenCommit(BasePolicy):
    """
    Explore Then Commit (ETC) policy implementation.
    
    This policy explores each arm a fixed number of times during an exploration phase,
    then commits to the arm with the highest estimated reward.
    """
    
    def __init__(self, n_arms, horizon, exploration_phase=None, seed=None):
        """
        Initialize the Explore Then Commit policy.
        
        Args:
            n_arms (int): Number of arms in the bandit.
            horizon (int): Time horizon for the policy.
            exploration_phase (int, optional): Length of the exploration phase.
                If None, defaults to ceil(horizon^(2/3)).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(n_arms, horizon, seed)
        
        # Set exploration phase length
        if exploration_phase is None:
            self.exploration_phase = int(np.ceil(horizon**(2/3)))
        else:
            self.exploration_phase = exploration_phase
            
        # Ensure exploration phase is a multiple of n_arms
        self.exploration_phase = self.exploration_phase - (self.exploration_phase % n_arms)
        if self.exploration_phase == 0:
            self.exploration_phase = n_arms
            
        # Set up for round-robin during exploration
        self.exploration_round = 0
        self.committed_arm = None
        
    def select_arm(self):
        """
        Select an arm according to the ETC policy.
        
        During the exploration phase, pulls arms in a round-robin fashion.
        After exploration, commits to the arm with highest estimated reward.
        
        Returns:
            int: The index of the selected arm.
        """
        # Check if we're in the exploration phase
        if self.t < self.exploration_phase:
            # Round-robin selection during exploration
            arm = self.t % self.n_arms
            return arm
        
        # If we just finished exploration, commit to the best arm
        if self.committed_arm is None:
            self.committed_arm = np.argmax(self.estimates)
            
        # Return the committed arm
        return self.committed_arm
    
    def reset(self):
        """
        Reset the policy to its initial state.
        """
        super().reset()
        self.exploration_round = 0
        self.committed_arm = None