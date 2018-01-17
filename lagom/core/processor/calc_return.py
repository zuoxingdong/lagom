from lagom.core.processor import BaseProcessor


class CalcReturn(BaseProcessor):
    def __init__(self, gamma):
        self.gamma = gamma
        
    def process(self, x):
        """
        Compute returns for an episode
        """
        returns = []
        R = 0
        for r in x[::-1]:
            R = r + self.gamma*R
            returns.insert(0, R)

        return returns
