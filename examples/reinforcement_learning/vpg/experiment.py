from lagom.experiment import Configurator
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster
from lagom.experiment import run_experiment

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
        configurator.fixed('env.time_aware_obs', False)  # whether to append time step to observation
        
        configurator.fixed('network.recurrent', False)
        configurator.fixed('network.hidden_sizes', [64, 64])  # TODO: [64, 64]
        configurator.fixed('network.independent_V', False)  # share or not for params of policy and value network
        
        configurator.fixed('algo.lr', 7e-4)
        configurator.fixed('algo.lr_V', 1e-3)
        configurator.fixed('algo.use_lr_scheduler', True)
        configurator.fixed('algo.gamma', 0.99)
        configurator.fixed('algo.gae_lambda', 0.97)
        
        configurator.fixed('agent.standardize_Q', False)  # whether to standardize discounted returns
        configurator.fixed('agent.standardize_adv', True)  # whether to standardize advantage estimates
        configurator.fixed('agent.max_grad_norm', 0.5)  # grad clipping, set None to turn off
        configurator.fixed('agent.entropy_coef', 0.01)
        configurator.fixed('agent.value_coef', 0.5)
        configurator.fixed('agent.fit_terminal_value', False)
        configurator.fixed('agent.terminal_value_coef', 0.1)
        # only for continuous control
        configurator.fixed('env.clip_action', True)  # clip sampled action within valid bound before step()
        configurator.fixed('agent.min_std', 1e-6)  # min threshould for std, avoid numerical instability
        configurator.fixed('agent.std_style', 'exp')  # std parameterization, 'exp' or 'softplus'
        configurator.fixed('agent.constant_std', None)  # constant std, set None to learn it
        configurator.fixed('agent.std_state_dependent', False)  # whether to learn std with state dependency
        configurator.fixed('agent.init_std', 0.5)  # initial std for state-independent std
        
        configurator.fixed('train.timestep', 1e6)  # either 'train.iter' or 'train.timestep'
        configurator.fixed('train.N', 1)  # number of trajectories per training iteration
        configurator.fixed('train.ratio_T', 1.0)  # percentage of max allowed horizon
        configurator.fixed('eval.independent', False)
        configurator.fixed('eval.N', 10)  # number of episodes to evaluate, do not specify T for complete episode
        
        configurator.fixed('log.interval', 10)  # logging interval
        configurator.fixed('log.dir', 'logs')  # logging directory
        
        list_config = configurator.make_configs()
        
        return list_config

    def make_seeds(self):
        list_seed = [1770966829, 1500925526, 2054191100]
        
        return list_seed
    
    def process_results(self, results):
        assert all([result is None for result in results])

        
if __name__ == '__main__':
    run_experiment(worker_class=ExperimentWorker, 
                   master_class=ExperimentMaster, 
                   num_worker=100)
