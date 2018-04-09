from lagom.core.preprocessors import BasePreprocessor


class CalcReturn(BasePreprocessor):
    def __init__(self, gamma):
        """
        Args:
            gamma (float): discounted factor in (0, 1]
        """
        self.gamma = gamma
        
    def process(self, x):
        """
        Compute returns for an episode
        
        Args:
            x (list): a list of rewards for each time step
            
        Returns:
            returns (list): a list of discounted returns for each time step
        """
        returns = []
        R = 0
        for r in x[::-1]:
            R = r + self.gamma*R
            returns.insert(0, R)

        return returns
