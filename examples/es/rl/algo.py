from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lagom import Logger

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import ParallelVecEnv
from lagom.runner import TrajectoryRunner

from lagom.core.policies import CategoricalPolicy
from lagom.core.policies import GaussianPolicy

from lagom.core.es import CMAES
from lagom.core.es import OpenAIES

from lagom.core.es import BaseESWorker
from lagom.core.es import BaseESMaster

from lagom import set_global_seeds
from lagom import BaseAlgorithm

from lagom import pickle_dump

from policy import Network
from policy import LSTM
from policy import Agent


class ESWorker(BaseESWorker):
    def prepare(self):
        self.agent = None
        
    def init(self, seed, config):
        # Make environment
        # Remember to seed it in each working function !
        self.env = make_vec_env(vec_env_class=SerialVecEnv, 
                                make_env=make_gym_env, 
                                env_id=config['env.id'], 
                                num_env=config['train.N'], 
                                init_seed=seed, 
                                rolling=False)
        self.env_spec = EnvSpec(self.env)
        
        # Make agent
        if config['network.recurrent']:
            self.network = LSTM(config=config, env_spec=self.env_spec)
        else:
            self.network = Network(config=config, env_spec=self.env_spec)
        if self.env_spec.control_type == 'Discrete':
            self.policy = CategoricalPolicy(config=config, network=self.network, env_spec=self.env_spec, device=None)
        elif self.env_spec.control_type == 'Continuous':
            self.policy = GaussianPolicy(config=config, network=self.network, env_spec=self.env_spec, device=None)
        self.agent = Agent(config=config, policy=self.policy)
        
    def f(self, solution, seed, config):
        if self.agent is None:
            self.init(seed, config)

        # load solution parameters to the agent internal network
        msg = f'expected {self.network.num_params}, got {np.asarray(solution).size}'
        assert np.asarray(solution).size == self.network.num_params, msg
        self.agent.policy.network.from_vec(torch.from_numpy(solution).float())
        
        # create runner
        runner = TrajectoryRunner(agent=self.agent, env=self.env, gamma=1.0)
        
        # take rollouts and calculate mean return (no discount)
        with torch.no_grad():
            D = runner(T=config['train.T'])
        
        mean_return = np.mean([sum(trajectory.all_r) for trajectory in D])
        
        # Negate return to be objective value, because ES does minimization
        function_value = -mean_return
        
        return function_value


class ESMaster(BaseESMaster):
    def _network_size(self):
        worker = ESWorker()
        tmp_agent = worker.init(seed=0, config=self.config)
        num_params = worker.network.num_params
        
        del worker, tmp_agent
        
        return num_params
    
    def make_es(self, config):
        if self.config['es.algo'] == 'CMAES':
            es = CMAES(mu0=[self.config['es.mu0']]*self._network_size(),
                       std0=self.config['es.std0'], 
                       popsize=self.config['es.popsize'])
        elif self.config['es.algo'] == 'OpenAIES':
            es = OpenAIES(mu0=[self.config['es.mu0']]*self._network_size(), 
                          std0=self.config['es.std0'], 
                          popsize=self.config['es.popsize'], 
                          std_decay=0.999,
                          min_std=0.01, 
                          lr=5e-2, 
                          lr_decay=0.99, 
                          min_lr=1e-3, 
                          antithetic=True, 
                          rank_transform=True)
        
        self.logger = Logger()
        
        return es
        
    def _process_es_result(self, result):
        best_f_val = result['best_f_val']
        best_return = -best_f_val  # negate to get back reward
        
        # logging
        self.logger.log('generation', self.generation)
        self.logger.log('best_return', best_return)
        
        if self.generation == 0 or (self.generation+1) % self.config['log.interval'] == 0:
            print('-'*50)
            self.logger.dump(keys=None, index=-1, indent=0)
            print('-'*50)
            
        # Save the loggings and final parameters
        if (self.generation+1) == self.num_iteration:
            pickle_dump(obj=self.logger.logs, f=self.logdir/'result', ext='.pkl')
            np.save(self.logdir/'trained_param', result['best_param'])


class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device_str):
        # Set random seeds
        set_global_seeds(seed)
        # Use log dir for current job (run_experiment)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)
        
        # train
        es = ESMaster(num_iteration=config['train.num_iteration'], 
                      worker_class=ESWorker, 
                      init_seed=seed, 
                      daemonic_worker=None, 
                      config=config, 
                      logdir=logdir)
        es()
        
        return None
