This example includes the implementations of the following reinforcement learning algorithms:

- ES
    - [Cross Entropy Method (CEM)](cem)
    - [Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](cmaes)
    - [OpenAI-ES](openaies)
- RL
    - [Vanilla Policy Gradient (VPG)](vpg)
    - [Proximal Policy Optimization (PPO)](ppo)
    - [Deep Deterministic Policy Gradients (DDPG)](ddpg)
    - [Twin Delayed DDPG (TD3)](td3)
    - [Soft Actor-Critic (SAC)](sac)

# Benchmarks

## ES
<img src='benchmark_es.png' width='100%'>

## RL
<img src='benchmark_rl.png' width='100%'>

## FAQ:
- How to train with [dm_control](https://github.com/deepmind/dm_control) environments?
    - Modify `experiment.py`: use [dm2gym](https://github.com/zuoxingdong/dm2gym) wrapper, e.g.
    ```python
    from gym.wrappers import FlattenDictWrapper
    from dm_control import suite
    from dm2gym import DMControlEnv

    config = Config(
        ...
        'env.id': Grid([('cheetah', 'run'), ('hopper', 'hop'), ('walker', 'run'), ('fish', 'upright')]),
        ...
        )

    def make_env(config, seed):
        domain_name, task_name = config['env.id']
        env = suite.load(domain_name, task_name, environment_kwargs=dict(flat_observation=True))
        env = DMControlEnv(env)
        env = FlattenDictWrapper(env, ['observations'])
        ...
    ```
