class BaseExperiment(object):
    def __init__(self, num_configs):
        self.num_configs = num_configs
        
        self.list_args = self.configure(self.num_configs)
    
    def add_algo(self, algo):
        self.algo = algo
    
    def configure(self, num_configs):
        """Generate all settings and needed set of hyperparameters as list of args"""
        raise NotImplementedError
        
    def benchmark(self):
        raise NotImplementedError