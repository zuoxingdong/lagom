from lagom.experiment import GridConfig
from lagom.experiment import BaseExperiment


class Experiment(BaseExperiment):
    def _configure(self):
        config = GridConfig()
        
        config.add('batch_size', [128])
        config.add('num_epochs', [1])
        config.add('seed', [1])
        config.add('log_interval', [100])
        config.add('cuda', [True])
        
        return config.make_configs()
    
    def _make_env(self, config):
        pass