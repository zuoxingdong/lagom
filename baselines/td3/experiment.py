import gym
import gym.wrappers

import lagom
import lagom.utils as utils

from baselines.td3.agent import Agent
from baselines.td3.engine import Engine


configurator = lagom.Configurator(
    {'log.freq': 10,
     'checkpoint.inference.num': 3,
     
     'env.id': lagom.Grid(['HalfCheetah-v3', 'Hopper-v3', 'Walker2d-v3']),
     
     'agent.gamma': 0.99,
     'agent.polyak': 0.995,  # polyak averaging coefficient for targets update
     'agent.actor.lr': 1e-3, 
     'agent.actor.use_lr_scheduler': False,
     'agent.critic.lr': 1e-3,
     'agent.critic.use_lr_scheduler': False,
     'agent.action_noise': 0.1,
     'agent.max_grad_norm': 1e5,  # grad clipping by norm
     
     # TD3 hyperparams
     'agent.target_noise': 0.2,
     'agent.target_noise_clip': 0.5,
     'agent.policy_delay': 2,
     
     'replay.capacity': int(1e6), 
     'replay.init_trial': 10,  # number of random rollouts initially
     'replay.batch_size': 100,
     
     'train.timestep': int(1e6),  # total number of training (environment) timesteps
     'eval.num': 200
    }, 
    num_sample=1)


def make_env(config, mode):
    assert mode in ['train', 'eval']
    env = gym.make(config['env.id'])
    env.seed(config.seed)
    env.observation_space.seed(config.seed)
    env.action_space.seed(config.seed)
    env = gym.wrappers.RescaleAction(env, -1.0, 1.0)
    if mode == 'eval':
        env = lagom.envs.RecordEpisodeStatistics(env, deque_size=100)
    env = lagom.envs.TimeStepEnv(env)
    return env


def run(config):
    lagom.set_global_seeds(config.seed)
    
    env = make_env(config, 'train')
    eval_env = make_env(config, 'eval')
    agent = Agent(config, env)
    runner = lagom.EpisodeRunner()
    replay = lagom.UniformTransitionBuffer(env, config['replay.capacity'])
    engine = Engine(config, agent=agent)
    
    cond_save = utils.NConditioner(max_n=config['train.timestep'], num_conditions=config['checkpoint.inference.num'], mode='accumulative')
    cond_eval = utils.NConditioner(max_n=config['train.timestep'], num_conditions=config['eval.num'], mode='accumulative')
    train_logs, eval_logs = [], []
    iteration = 0
    if config.resume_checkpointer.exists():
        out = lagom.checkpointer('load', config, state_obj=[agent, agent.actor_optimizer, agent.critic_optimizer])
        env, eval_env, runner, train_logs, eval_logs, iteration = out
        agent.env = env

    while agent.total_timestep < config['train.timestep']:
        train_logger = engine.train(iteration, env=env, runner=runner, replay=replay)
        train_logs.append(train_logger.logs)
        if cond_save(int(agent.total_timestep)):
            agent.checkpoint(config.logdir, iteration+1)
        if cond_eval(int(agent.total_timestep)):
            eval_logger = engine.eval(iteration, eval_env=eval_env, runner=runner)
            eval_logs.append(eval_logger.logs)
        utils.pickle_dump(obj=train_logs, f=config.logdir/'train_logs', ext='.pkl')
        utils.pickle_dump(obj=eval_logs, f=config.logdir/'eval_logs', ext='.pkl')
        lagom.checkpointer('save', config, obj=[env, eval_env, runner, train_logs, eval_logs, iteration+1], state_obj=[agent, agent.actor_optimizer, agent.critic_optimizer])
        iteration += 1
    agent.checkpoint(config.logdir, iteration+1)    
    return None


if __name__ == '__main__':
    lagom.run_experiment(run=run, 
                         configurator=configurator, 
                         seeds=lagom.SEEDS[:5],
                         log_dir='logs/default',
                         max_workers=None,
                         chunksize=1, 
                         use_gpu=False,  # GPU much faster, note that performance differs between CPU/GPU
                         gpu_ids=None)
