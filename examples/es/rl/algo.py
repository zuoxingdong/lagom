from pathlib import Path

import numpy as np
import torch

from lagom import Logger
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds

from lagom.es import BaseESMaster
from lagom.es import BaseESWorker

from lagom import BaseAlgorithm

from lagom.envs import make_gym_env
from lagom.envs import make_vec_env
from lagom.envs import EnvSpec
from lagom.envs.vec_env import SerialVecEnv
from lagom.envs.vec_env import VecStandardize
from lagom.envs.vec_env import VecClipAction

from lagom.runner import EpisodeRunner

from model import Agent
from examples.es import CMAES
from examples.es import OpenAIES


class ESWorker(BaseESWorker):
    def prepare(self):
        self.agent = None
        
    def _prepare(self, config):
        self.env = make_vec_env(SerialVecEnv, make_gym_env, config['env.id'], config['train.N'], 0)
        self.env = VecClipAction(self.env)
        if config['env.standardize']:
            self.env = VecStandardize(self.env, 
                                      use_obs=True, 
                                      use_reward=False, 
                                      clip_obs=10.0, 
                                      clip_reward=10.0, 
                                      gamma=0.99, 
                                      eps=1e-08)
        self.env_spec = EnvSpec(self.env)
        
        self.device = torch.device('cpu')
            
        self.agent = Agent(config, self.env_spec, self.device)
    
    def f(self, config, solution):
        if self.agent is None:
            self._prepare(config)
            
        solution = torch.from_numpy(np.asarray(solution)).float().to(self.device)
        assert solution.numel() == self.agent.num_params
        
        # Load solution params to agent
        self.agent.from_vec(solution)
        
        runner = EpisodeRunner(config, self.agent, self.env)
        with torch.no_grad():
            D = runner(self.env_spec.T)
        mean_return = D.numpy_rewards.sum(-1).mean()
        
        # ES does minimization, so use negative returns
        function_value = -mean_return
        
        return function_value


class ESMaster(BaseESMaster):
    @property
    def _num_params(self):
        worker = ESWorker()
        worker._prepare(self.config)
        num_params = worker.agent.num_params
        del worker
        
        return num_params
    
    def make_es(self, config):
        if self.config['es.algo'] == 'CMAES':
            es = CMAES(mu0=[self.config['es.mu0']]*self._num_params,
                       std0=self.config['es.std0'], 
                       popsize=self.config['es.popsize'])
        elif self.config['es.algo'] == 'OpenAIES':
            es = OpenAIES(mu0=[self.config['es.mu0']]*self._num_params, 
                          std0=self.config['es.std0'], 
                          popsize=self.config['es.popsize'], 
                          std_decay=0.999,
                          min_std=0.01, 
                          lr=1e-1, 
                          lr_decay=0.99, 
                          min_lr=1e-3, 
                          antithetic=True, 
                          rank_transform=True)
        
        self.logger = Logger()
        
        return es
        
    def process_es_result(self, result):
        best_f_val = result['best_f_val']
        best_return = -best_f_val
        
        self.logger('generation', self.generation + 1)
        self.logger('best_return', best_return)
        
        if self.generation == 0 or (self.generation+1) % self.config['log.interval'] == 0:
            print('-'*50)
            self.logger.dump(keys=None, index=-1, indent=0)
            print('-'*50)
            
        # Save the loggings and final parameters
        if (self.generation+1) == self.config['train.num_iteration']:
            pickle_dump(obj=self.logger.logs, f=self.logdir/'result', ext='.pkl')
            np.save(self.logdir/'trained_param', result['best_param'])
            

class Algorithm(BaseAlgorithm):
    def __call__(self, config, seed, device):
        set_global_seeds(seed)
        logdir = Path(config['log.dir']) / str(config['ID']) / str(seed)

        es = ESMaster(config, ESWorker, logdir=logdir)
        es()
        
        return None
