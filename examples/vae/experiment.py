from lagom.experiment import Configurator
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster

from algo import Algorithm


class ExperimentWorker(BaseExperimentWorker):
    def prepare(self):
        pass
        
    
    def make_algo(self):
        algo = Algorithm()
        
        return algo


class ExperimentMaster(BaseExperimentMaster):
    def make_configs(self):
        configurator = Configurator('grid')
        
        configurator.fixed('cuda', True)
        
        configurator.grid('network.type', ['VAE', 'ConvVAE'])
        configurator.fixed('network.z_dim', 8)
        
        configurator.fixed('train.num_epoch', 100)
        configurator.fixed('train.batch_size', 128)
        configurator.fixed('eval.batch_size', 128)
        
        configurator.fixed('log.interval', 100)
        configurator.fixed('log.dir', 'logs')

        list_config = configurator.make_configs()
        
        return list_config
    
    def make_seeds(self):
        list_seed = [1]
        
        return list_seed
    
    def process_results(self, results):
        assert all([result is None for result in results])
