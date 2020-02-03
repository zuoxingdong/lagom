This example includes the implementations of the following reinforcement learning algorithms:

- ES
    - [Cross Entropy Method (CEM)](cem)
    - [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](cmaes)

- RL
    - [Vanilla Policy Gradient (VPG)](vpg)
    - [Proximal Policy Optimization (PPO)](ppo)
    - [Deep Deterministic Policy Gradients (DDPG)](ddpg)
    - [Twin Delayed DDPG (TD3)](td3)
    - [Soft Actor-Critic (SAC)](sac)
    
# Wrapping loggings for reproducibility
- Run `python zip_logs.py` to wrap the `logs` folders of all algorithms as `all_logs.tar.gz`
- Run `python unzip_logs.py` to automatically extract all `logs` folders to each algorithm folder. 

Note: The files of loggings and checkpoints are very large, they can slow down the cloning, thus we exclude them from the repo. 

# Benchmarks

## ES
<img src='https://i.imgur.com/2aLq7Pr.png' width='100%'>

## Model-free RL
<img src='https://i.imgur.com/nvl36RF.png' width='100%'>

## FAQ:
- How to train with [dm_control](https://github.com/deepmind/dm_control) environments?
    - Install [dm2gym](https://github.com/zuoxingdong/dm2gym) wrapper.
    - Modify `experiment.py`: change `env.id`, e.g.
    ```python
    import gym
    import gym.wrappers

    configurator = lagom.Configurator(
        ...
        'env.id': lagom.Grid(['dm2gym:CheetahRun-v0', 'dm2gym:HopperHop-v0']),
        ...
        )

    def make_env(config, mode):
        env = gym.make(config['env.id'])
        env = gym.wrappers.FlattenObservation(env)
        ...
    ```
