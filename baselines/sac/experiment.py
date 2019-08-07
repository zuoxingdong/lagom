import os
import gym

from lagom import EpisodeRunner
from lagom import RandomAgent
from lagom.utils import pickle_dump
from lagom.utils import set_global_seeds
from lagom.experiment import Config
from lagom.experiment import Grid
from lagom.experiment import run_experiment
from lagom.envs import RecordEpisodeStatistics
from lagom.envs import TimeStepEnv
from lagom.envs.wrappers import NormalizeAction

from baselines.sac.agent import Agent
from baselines.sac.engine import Engine
from baselines.sac.replay_buffer import ReplayBuffer


config = Config(
    {'log.freq': 10,
     'checkpoint.num': 3,
     
     'env.id': Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3', 'Swimmer-v3']),
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 3e-4, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 3e-4,
     'agent.critic.use_lr_scheduler': False,
     'agent.initial_temperature': 1.0,
     'agent.max_grad_norm': 999999,  # grad clipping by norm
     
     'replay.capacity': 1000000, 
     'replay.init_trial': 10,  # number of random rollouts initially
     'replay.batch_size': 256,
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'eval.num': 200
    })


def make_env(config, seed, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(seed)
    env.observation_space.seed(seed)
    env.action_space.seed(seed)
    env = NormalizeAction(env)  # TODO: use gym new wrapper RescaleAction when it's merged
    if mode == 'eval':
        env = RecordEpisodeStatistics(env, deque_size=100)
    env = TimeStepEnv(env)
    return env


def run(config, seed, device, logdir):
    set_global_seeds(seed)
    
    env = make_env(config, seed, 'train')
    eval_env = make_env(config, seed, 'eval')
    random_agent = RandomAgent(config, env, device)
    agent = Agent(config, env, device)
    runner = EpisodeRunner()
    replay = ReplayBuffer(env, config['replay.capacity'], device)
    engine = Engine(config, agent=agent, random_agent=random_agent, env=env, eval_env=eval_env, runner=runner, replay=replay, logdir=logdir)
    
    train_logs, eval_logs = engine.train()
    pickle_dump(obj=train_logs, f=logdir/'train_logs', ext='.pkl')
    pickle_dump(obj=eval_logs, f=logdir/'eval_logs', ext='.pkl')
    return None  
    

if __name__ == '__main__':
    run_experiment(run=run, 
                   config=config, 
                   seeds=[4153361530, 3503522377, 2876994566, 172236777, 3949341511], 
                   log_dir='logs/default',
                   max_workers=os.cpu_count(), 
                   chunksize=1, 
                   use_gpu=True,  # GPU much faster, note that performance differs between CPU/GPU
                   gpu_ids=None)
