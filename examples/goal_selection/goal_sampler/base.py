class BaseGoalSampler(object):
    """
    Base class for goal sampler
    """
    def __init__(self, runner, config):
        self.runner = runner
        self.config = config
        
    def sample(self):
        """
        Returns a sample in goal space
        
        Returns:
            goal (object): sampled goal
        """
        raise NotImplementedError
        