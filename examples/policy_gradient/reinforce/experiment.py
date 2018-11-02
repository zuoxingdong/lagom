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
        
        configurator.fixed('cuda', True)  # whether to use GPU
        
        configurator.fixed('env.id', 'HalfCheetah-v2')
        configurator.fixed('env.standardize', True)  # whether to use VecStandardize
        
        configurator.fixed('network.recurrent', False)
        configurator.fixed('network.hidden_sizes', [64, 64])  # TODO: [64, 64]
        
        configurator.fixed('algo.lr', 1e-3)
        configurator.fixed('algo.use_lr_scheduler', True)
        configurator.fixed('algo.gamma', 0.99)
        
        configurator.fixed('agent.standardize_Q', True)  # whether to standardize discounted returns
        configurator.fixed('agent.max_grad_norm', 0.5)  # grad clipping, set None to turn off
        configurator.fixed('agent.entropy_coef', 0.01)
        # only for continuous control
        configurator.fixed('agent.min_std', 1e-6)  # min threshould for std, avoid numerical instability
        configurator.fixed('agent.std_style', 'exp')  # std parameterization, 'exp' or 'softplus'
        configurator.fixed('agent.constant_std', None)  # constant std, set None to learn it
        configurator.fixed('agent.std_state_dependent', False)  # whether to learn std with state dependency
        configurator.fixed('agent.init_std', 1.0)  # initial std for state-independent std
        
        configurator.fixed('train.timestep', 1e6)  # either 'train.iter' or 'train.timestep'
        configurator.fixed('train.N', 1)  # number of trajectories per training iteration
        configurator.fixed('train.T', 200)  # max allowed horizon
        configurator.fixed('eval.N', 10)  # number of episodes to evaluate, do not specify T for complete episode
        
        configurator.fixed('log.record_interval', 100)  # interval to record the logging
        configurator.fixed('log.print_interval', 100)  # interval to print the logging to screen
        configurator.fixed('log.dir', 'logs')  # logging directory
        
        list_config = configurator.make_configs()
        
        return list_config

    def make_seeds(self):
        list_seed = [209652396, 398764591, 924231285, 1478610112, 441365315]
        
        return list_seed
    
    def process_results(self, results):
        assert all([result is None for result in results])
