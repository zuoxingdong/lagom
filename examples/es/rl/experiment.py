from lagom.experiment import Configurator
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster

from algo import Algorithm


class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = Algorithm(name='ES for RL')
        
        return algo


class ExperimentMaster(BaseExperimentMaster):
    def make_configs(self):
        configurator = Configurator('grid')
        
        configurator.fixed('cuda', False)  # ES for small net, do not use GPU
        
        configurator.fixed('env.id', 'Pendulum-v0')
        
        configurator.fixed('network.recurrent', False)
        configurator.fixed('network.hidden_size', [32])
        
        configurator.fixed('es.popsize', 128)
        configurator.fixed('es.mu0', 0.0)
        configurator.fixed('es.std0', 0.5)
        
        """Hyperparameter search, later time
        configurator.grid('env.id', ['Pendulum-v0', 'Reacher-v2', 'InvertedPendulum-v2', 'HumanoidStandup-v2'])
        
        configurator.grid('network.hidden_size', [[32], [32, 32], [64, 64]])
        
        configurator.grid('es.popsize', [8, 16, 32, 64])
        configurator.grid('es.mu0', [0.0, 0.3, 0.5])
        configurator.grid('es.std0', [0.1, 0.5, 1.0])
        """
        configurator.fixed('train.num_iteration', 1000)
        configurator.fixed('train.N', 5)
        configurator.fixed('train.T', 300)
        
        configurator.fixed('log.interval', 100)
        configurator.fixed('log.dir', 'logs')

        list_config = configurator.make_configs()
        
        return list_config
    
    def make_seeds(self):
        list_seed = [209652396, 398764591, 924231285, 1478610112, 441365315]
        
        return list_seed
    
    def process_algo_result(self, config, seed, result):
        assert result is None
