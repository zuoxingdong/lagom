import numpy as np


class Trajectory(object):
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.step_infos = []
        
    def __len__(self):
        return len(self.step_infos)
        
    @property
    def completed(self):
        return len(self.step_infos) > 0 and self.step_infos[-1].last
    
    @property
    def reach_time_limit(self):
        return self.step_infos[-1].time_limit
    
    @property
    def reach_terminal(self):
        return self.step_infos[-1].terminal
        
    def add_observation(self, observation):
        assert not self.completed
        self.observations.append(observation)
    
    def add_action(self, action):
        assert not self.completed
        self.actions.append(action)
    
    def add_reward(self, reward):
        assert not self.completed
        self.rewards.append(reward)
        
    def add_step_info(self, step_info):
        assert not self.completed
        self.step_infos.append(step_info)
        if step_info.last:
            assert self.completed
    
    @property
    def last_observation(self):
        return self.observations[-1]
    
    @property
    def numpy_observations(self):
        return np.concatenate(self.observations, axis=0)
    
    @property
    def numpy_actions(self):
        return np.concatenate(self.actions, axis=0)
        
    @property
    def numpy_rewards(self):
        return np.asarray(self.rewards)
    
    @property
    def numpy_dones(self):
        return np.asarray([step_info.done for step_info in self.step_infos])
    
    @property
    def numpy_masks(self):
        return 1. - self.numpy_dones
    
    @property
    def infos(self):
        return [step_info.info for step_info in self.step_infos]
    
    def get_all_info(self, key):
        return [step_info[key] for step_info in self.step_infos]
    
    def __repr__(self):
        return f'Trajectory(T: {len(self)}, Completed: {self.completed}, Reach time limit: {self.reach_time_limit}, Reach terminal: {self.reach_terminal})'



@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_trajectory(init_seed, T):
    make_env = lambda: TimeLimit(SanityEnv())
    env = make_vec_env(make_env, 1, init_seed)  # single environment
    env = VecStepInfo(env)
    D = Trajectory()
    assert len(D) == 0
    assert not D.completed
    
    observation, _ = env.reset()
    D.add_observation(observation)
    for t in range(T):
        action = [env.action_space.sample()]
        next_observation, reward, [step_info] = env.step(action)
        if step_info.last:
            D.add_observation([step_info['last_observation']])
        else:
            D.add_observation(next_observation)
        D.add_action(action)
        D.add_reward(reward)
        D.add_step_info(step_info)
        observation = next_observation
        if step_info.last:
            with pytest.raises(AssertionError):
                D.add_observation(observation)
            break
    assert len(D) > 0
    assert len(D) <= T
    assert len(D) + 1 == len(D.observations)
    assert len(D) + 1 == len(D.numpy_observations)
    assert len(D) == len(D.actions)
    assert len(D) == len(D.numpy_actions)
    assert len(D) == len(D.rewards)
    assert len(D) == len(D.numpy_rewards)
    assert len(D) == len(D.numpy_dones)
    assert len(D) == len(D.numpy_masks)
    assert np.allclose(np.logical_not(D.numpy_dones), D.numpy_masks)
    assert len(D) == len(D.step_infos)
    if len(D) < T:
        assert step_info.last
        assert D.completed
        assert D.reach_terminal
        assert not D.reach_time_limit
        assert np.allclose(D.observations[-1], [step_info['last_observation']])
    if not step_info.last:
        assert not D.completed
        assert not D.reach_terminal
        assert not D.reach_time_limit
