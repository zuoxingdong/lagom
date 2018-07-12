from algo import Algorithm

from lagom.experiment import Config
from lagom.experiment import BaseExperimentWorker
from lagom.experiment import BaseExperimentMaster


class ExperimentWorker(BaseExperimentWorker):
    def make_algo(self):
        algo = Algorithm(name='Policy gradient in CartPole')
        
        return algo


class ExperimentMaster(BaseExperimentMaster):
    def process_algo_result(self, config, result):
        assert result == None
        
    def make_configs(self):
        config = Config()
        
        config.add_item(name='cuda', val=False)
        config.add_item(name='seed', val=1)
        
        config.add_item(name='lr', val=1e-2)
        config.add_item(name='gamma', val=0.995)
        config.add_item(name='standardize_r', val=True)
        config.add_item(name='train_iter', val=1000)
        config.add_item(name='N', val=1)
        config.add_item(name='T', val=300)
        
        config.add_item(name='log_interval', val=100)
        
        configs = config.make_configs()
        
        return configs

"""
class Experiment(BaseExperiment):
    def _configure(self):
        config = GridConfig()
        
        config.add('seed', list(range(1)))  # random seeds
        
        config.add('hidden_sizes', [[32]])
        config.add('hidden_nonlinearity', [F.tanh])
        config.add('lr', [1e-2])  # learning rate of policy network
        config.add('gamma', [0.995])  # discount factor
        config.add('GAE_lambda', [0.86])  # GAE lambda
        config.add('value_coef', [0.5])  # value function loss coefficient
        config.add('entropy_coef', [0.0])  # policy entropy loss coefficient
        #config.add('max_grad_norm', [0.5])  # clipping for max gradient norm
        config.add('T', [10000])  # Max time step per episode
        config.add('use_optimal_T', [False])  # True: args.T will be modified to optimal steps before rollout for each new goal
        config.add('predict_value', [True])  # Value function head
        
        
        config.add('train_iter', [50])  # number of training iterations
        config.add('eval_iter', [1])  # number of evaluation iterations
        config.add('train_num_epi', [1])  # Number of episodes per training iteration
        config.add('eval_num_epi', [10])  # Number of episodes per evaluation iteration
        
        config.add('init_state', [[6, 1]])  # initial position for each created environment
        
        config.add('log_interval', [1])
        
        return config.make_configs()
            
    def _make_env(self, config):
        env = gym.make('CartPole-v0')
        env = GymEnv(env)
        
        return env
    
"""