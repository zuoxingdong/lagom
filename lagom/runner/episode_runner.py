from lagom.envs import VecEnv

from .base_runner import BaseRunner
from .trajectory import Trajectory


class EpisodeRunner(BaseRunner):
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, VecEnv)
        assert len(env) == 1, 'for cleaner API, one should use single VecEnv'
        
        D = [Trajectory()]
        observation = env.reset()
        D[-1].add_observation(observation)
        for t in range(T):
            out_agent = agent.choose_action(observation, **kwargs)
            action = out_agent.pop('raw_action')
            next_observation, reward, done, info = env.step(action)
            # unbatched for [reward, done, info]
            reward, done, info = map(lambda x: x[0], [reward, done, info])
            info = {**info, **out_agent}
            if done:  
                D[-1].add_observation([info['last_observation']])  # add a batch one, consistent
            else:
                D[-1].add_observation(next_observation)
            D[-1].add_action(action)
            D[-1].add_reward(reward)
            D[-1].add_info(info)
            D[-1].add_done(done)
            if done:
                assert D[-1].completed
                D.append(Trajectory())
                D[-1].add_observation(next_observation)  # initial observation
            observation = next_observation
        if len(D[-1]) == 0:
            D = D[:-1]
        return D
