class EpisodeRunner(BaseRunner):
    def __init__(self, reset_on_call=True):
        self.reset_on_call = reset_on_call
        self.observation = None
    
    def __call__(self, agent, env, T, **kwargs):
        assert isinstance(env, VecEnv) and isinstance(env, VecStepInfo) and len(env) == 1
        
        D = [Trajectory()]
        if self.reset_on_call:
            observation, _ = env.reset()
        else:
            if self.observation is None:
                self.observation, _ = env.reset()
            observation = self.observation
        D[-1].add_observation(observation)
        for t in range(T):
            out_agent = agent.choose_action(observation, **kwargs)
            action = out_agent.pop('raw_action')
            next_observation, [reward], [step_info] = env.step(action)
            step_info.info = {**step_info.info, **out_agent}
            if step_info.last:
                D[-1].add_observation([step_info['last_observation']])  # add a batch dim    
            else:
                D[-1].add_observation(next_observation)
            D[-1].add_action(action)
            D[-1].add_reward(reward)
            D[-1].add_step_info(step_info)
            if step_info.last:
                assert D[-1].completed
                D.append(Trajectory())
                D[-1].add_observation(next_observation)  # initial observation
            observation = next_observation
        if len(D[-1]) == 0:
            D = D[:-1]
        self.observation = observation
        return D


@pytest.mark.parametrize('env_id', ['Sanity', 'CartPole-v1', 'Pendulum-v0', 'Pong-v0'])
@pytest.mark.parametrize('num_env', [1, 3])
@pytest.mark.parametrize('init_seed', [0, 10])
@pytest.mark.parametrize('T', [1, 5, 100])
def test_episode_runner(env_id, num_env, init_seed, T):    
    if env_id == 'Sanity':
        make_env = lambda: TimeLimit(SanityEnv())
    else:
        make_env = lambda: gym.make(env_id)
    env = make_vec_env(make_env, num_env, init_seed)
    env = VecStepInfo(env)
    agent = RandomAgent(None, env, None)
    runner = EpisodeRunner()
    
    if num_env > 1:
        with pytest.raises(AssertionError):
            D = runner(agent, env, T)
    else:
        with pytest.raises(AssertionError):
            runner(agent, env.env, T)  # must be VecStepInfo
        D = runner(agent, env, T)
        for traj in D:
            assert isinstance(traj, Trajectory)
            assert len(traj) <= env.spec.max_episode_steps
            assert traj.numpy_observations.shape == (len(traj) + 1, *env.observation_space.shape)
            if isinstance(env.action_space, gym.spaces.Discrete):
                assert traj.numpy_actions.shape == (len(traj),)
            else:
                assert traj.numpy_actions.shape == (len(traj), *env.action_space.shape)
            assert traj.numpy_rewards.shape == (len(traj),)
            assert traj.numpy_dones.shape == (len(traj), )
            assert traj.numpy_masks.shape == (len(traj), )
            assert len(traj.step_infos) == len(traj)
            if traj.completed:
                assert np.allclose(traj.observations[-1], traj.step_infos[-1]['last_observation'])
