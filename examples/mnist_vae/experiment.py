from algo import Algorithm

from lagom.experiment import Config
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster


class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = Algorithm(name='VAE on MNIST')
        
        return algo
    
class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, result):
        assert result == None
        
    def make_configs(self):
        config = Config()
        
        config.add_item(name='use_ConvVAE', val=[True, False])
        
        config.add_item(name='num_epochs', val=100)
        config.add_item(name='cuda', val=True)
        config.add_item(name='seed', val=1)
        config.add_item(name='batch_size', val=128)
        config.add_item(name='log_interval', val=100)
        
        configs = config.make_configs()
        
        return configs