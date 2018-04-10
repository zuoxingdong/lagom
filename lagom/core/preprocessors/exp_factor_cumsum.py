from lagom.core.preprocessors import BasePreprocessor


class ExponentialFactorCumSum(BasePreprocessor):
    """
    Calculate future accumulated sums with exponential factor. 
    
    e.g. Given input [x_1, ..., x_n] and factor \alpha, the computation returns an array y with same length
    and y_i = x_i + \alpha*x_{i+1} + (\alpha)^2*x_{i+2} + ... + (\alpha)^{n-i-1}*x_{n-1} + (\alpha)^{n-i}*x_{n}
    
    Commonly useful for calculating returns in RL. 
    """
    def __init__(self, alpha):
        """
        Args:
            alpha (float): exponential factor
        """
        self.alpha = alpha
        
    def process(self, x):
        """
        Calculate future accumulated sums with exponential factor. 
        
        Args:
            x (list): a list of input values
            
        Returns:
            y (list): a list of output values
        """
        y = []
        
        cumsum = 0  # buffer of accumulated sum
        
        for val in x[::-1]:  # iterate items in reverse ordering
            cumsum = val + self.alpha*cumsum  # recursive update
            y.insert(0, cumsum)  # insert to the front

        return y
