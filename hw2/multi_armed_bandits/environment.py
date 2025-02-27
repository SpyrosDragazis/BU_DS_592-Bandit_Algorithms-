import numpy as np

class GaussianBandit:
    """
    A Multi-Armed Bandit environment with Gaussian reward distributions.
    
    Attributes:
        n_arms (int): Number of arms in the bandit.
        means (numpy.ndarray): Mean rewards for each arm.
        stds (numpy.ndarray): Standard deviations for each arm.
        best_arm (int): Index of the arm with the highest mean reward.
        optimal_reward (float): Mean reward of the best arm.
    """
    
    def __init__(self, means, stds=None):
        """
        Initialize a Gaussian Bandit environment.
        
        Args:
            means (list or numpy.ndarray): Mean rewards for each arm.
            stds (list or numpy.ndarray, optional): Standard deviations for each arm.
                If None, all arms will have std=1.0.
        """
        self.means = np.array(means)
        self.n_arms = len(means)
        
        if stds is None:
            self.stds = np.ones(self.n_arms)
        else:
            self.stds = np.array(stds)
            
        self.best_arm = np.argmax(self.means)
        self.optimal_reward = self.means[self.best_arm]
        
    def pull(self, arm):
        """
        Pull an arm and receive a reward.
        
        Args:
            arm (int): The arm to pull.
            
        Returns:
            float: The reward obtained.
        """
        if arm < 0 or arm >= self.n_arms:
            raise ValueError(f"Invalid arm index: {arm}. Must be between 0 and {self.n_arms-1}.")
        
        # Generate reward from Gaussian distribution
        reward = np.random.normal(self.means[arm], self.stds[arm])
        
        return reward
    
    def regret(self, arm):
        """
        Calculate the regret of pulling a specific arm.
        
        Args:
            arm (int): The arm that was pulled.
            
        Returns:
            float: The regret of the action.
        """
        return self.optimal_reward - self.means[arm]
    
    @staticmethod
    def create_two_armed_instance(delta, sigma=1.0):
        """
        Create a specific two-armed bandit instance as described in the homework 2.
        
        Args:
            delta (float): The gap between the optimal and suboptimal arm.
            sigma (float, optional): Standard deviation for both arms.
            
        Returns:
            GaussianBandit: A configured two-armed bandit instance.
        """
        means = [0, -delta]  # Arm 1 is optimal with mean 0, Arm 2 is suboptimal with mean -delta
        stds = [sigma, sigma]
        
        return GaussianBandit(means, stds)