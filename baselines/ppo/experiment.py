import gym
import gym.wrappers

import lagom
import lagom.utils as utils

from baselines.ppo.agent import Agent
from baselines.ppo.engine import Engine


configurator = lagom.Configurator(
    {'log.freq': 10, 
     'checkpoint.agent.num': 3,
     'checkpoint.resume.num': 3,
     
     'env.id': lagom.Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']),
     'env.normalize_obs': True,
     'env.normalize_reward': True,
     
     'agent.policy.sizes': [64, 64],
     'agent.policy.lr': 3e-4,
     'agent.value.sizes': [64, 64],
     'agent.value.lr': 1e-3,
     'agent.gamma': 0.99,
     'agent.gae_lambda': 0.95,
     'agent.standardize_adv': True,  # standardize advantage estimates
     'agent.max_grad_norm': 0.5,  # grad clipping by norm
     'agent.clip_range': 0.2,  # ratio clipping
     
     # only for continuous control
     'env.clip_action': True,  # clip action within valid bound before step()
     'agent.std0': 0.4,  # initial std
     
     'train.timestep': int(1e6),  # total number of training (environmental) timesteps
     'train.timestep_per_iter': 2048,  # number of timesteps per iteration
     'train.batch_size': 64,
     'train.num_epochs': 10,
    }, 
    num_sample=1)


def make_env(config, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)
    if config['env.clip_action'] and isinstance(env.action_space, gym.spaces.Box):
        env = gym.wrappers.ClipAction(env)
    if mode == 'train':
        env = lagom.envs.RecordEpisodeStatistics(env, deque_size=100)
        if config['env.normalize_obs']:
            env = lagom.envs.NormalizeObservation(env, clip=5.)
        if config['env.normalize_reward']:
            env = lagom.envs.NormalizeReward(env, clip=10., gamma=config['agent.gamma'])
    env = lagom.envs.TimeStepEnv(env)
    return env
    

def run(config):
    lagom.set_global_seeds(config.seed)
    
    env = make_env(config, 'train')
    agent = Agent(config, env)
    runner = lagom.StepRunner(reset_on_call=False)
    engine = Engine(config, agent=agent, env=env, runner=runner)
    
    cond_agent = utils.Conditioner(stop=config['train.timestep'], step=config['train.timestep']//config['checkpoint.agent.num'])
    cond_resume = utils.Conditioner(stop=config['train.timestep'], step=config['train.timestep']//config['checkpoint.resume.num'])
    cond_log = utils.Conditioner(step=config['log.freq'])
    train_logs = []
    iteration = 0
    if config.resume_checkpointer.exists():
        env, runner, train_logs, iteration = lagom.checkpointer('load', config, state_obj=[agent, agent.policy_optimizer, agent.value_optimizer])
        agent.env = env
        engine.env = env
        engine.runner = runner

    while agent.total_timestep < config['train.timestep']:
        train_logger = engine.train(iteration, cond_log=cond_log)
        train_logs.append(train_logger.logs)
        if cond_agent(int(agent.total_timestep)):
            agent.checkpoint(config.logdir, iteration+1)
        if cond_resume(int(agent.total_timestep)):
            utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
            lagom.checkpointer('save', config, obj=[env, runner, train_logs, iteration+1], state_obj=[agent, agent.policy_optimizer, agent.value_optimizer])
        iteration += 1
    utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
    agent.checkpoint(config.logdir, iteration+1)
    return None
    

if __name__ == '__main__':
    lagom.run_experiment(run=run,
                         configurator=configurator,
                         seeds=lagom.SEEDS[:3],
                         log_dir='logs/default',
                         max_workers=None,
                         chunksize=1,
                         use_gpu=False,  # CPU a bit faster
                         gpu_ids=None)
