import torch

from lagom.history import Transition
from lagom.history import Segment

from .base_runner import BaseRunner


class SegmentRunner(BaseRunner):
    r"""Define a data collection interface by running the agent in an environment and collecting a batch
    of segments for a certain time steps. 
    
    .. note::
        
        By default, the agent handles batched data returned from :class:`VecEnv` type of environment.
        And the collected data is a list of :class:`Segment`. 
    
    Each :class:`Segment` should store successive transitions i.e. :math:`(s_t, a_t, r_t, s_{t+1}, \text{done}_t)`
    and all corresponding useful information such as log-probabilities of actions, state values, Q values etc.
    
    The collected transitions in each :class:`Segment` come from one or multiple episodes. The first state is not
    restricted to be initial observation from an episode, this allows a rolling segments of episodic transitions. 
    And all segments have the same number of transitions (time steps). For detailed description, see the docstring
    in :class:`Segment`. 
    
    Be aware that if the transitions come from more than one episode, the succeeding transitions for the next episode
    start from initial observation. 
    
    .. note::
        
        For collecting batch of trajectories, one should use :class:`TrajectoryRunner` instead. 
    
    Example::
    
        >>> from lagom.agents import RandomAgent
        >>> from lagom.envs import make_envs, make_gym_env, EnvSpec
        >>> from lagom.envs.vec_env import SerialVecEnv
        >>> list_make_env = make_envs(make_env=make_gym_env, env_id='CartPole-v1', num_env=2, init_seed=0)
        >>> env = SerialVecEnv(list_make_env=list_make_env, rolling=True)
        >>> env_spec = EnvSpec(env)
    
        >>> agent = RandomAgent(config=None, env_spec=env_spec)
        >>> runner = SegmentRunner(agent=agent, env=env, gamma=1.0)

        >>> runner(T=3, reset=False)
        [Segment: 
            Transition: (s=[-0.04002427  0.00464987 -0.01704236 -0.03673052], a=1, r=1.0, s_next=[-0.03993127  0.20001201 -0.01777697 -0.33474139], done=False)
            Transition: (s=[-0.03993127  0.20001201 -0.01777697 -0.33474139], a=0, r=1.0, s_next=[-0.03593103  0.00514751 -0.0244718  -0.04771698], done=False)
            Transition: (s=[-0.03593103  0.00514751 -0.0244718  -0.04771698], a=0, r=1.0, s_next=[-0.03582808 -0.18961514 -0.02542614  0.23714553], done=False),
         Segment: 
            Transition: (s=[ 0.00854682  0.00830137 -0.03052506  0.03439879], a=0, r=1.0, s_next=[ 0.00871284 -0.18636984 -0.02983709  0.31729661], done=False)
            Transition: (s=[ 0.00871284 -0.18636984 -0.02983709  0.31729661], a=0, r=1.0, s_next=[ 0.00498545 -0.38105439 -0.02349115  0.60042265], done=False)
            Transition: (s=[ 0.00498545 -0.38105439 -0.02349115  0.60042265], a=0, r=1.0, s_next=[-0.00263564 -0.57583997 -0.0114827   0.88561464], done=False)]
    
        >>> runner(T=1, reset=False)
        [Segment: 
            Transition: (s=[-0.03582808 -0.18961514 -0.02542614  0.23714553], a=0, r=1.0, s_next=[-0.03962039 -0.38436478 -0.02068323  0.52170109], done=False),
         Segment: 
            Transition: (s=[-0.00263564 -0.57583997 -0.0114827   0.88561464], a=0, r=1.0, s_next=[-0.01415244 -0.77080416  0.00622959  1.17466581], done=False)]
        
    """
    def __init__(self, config, agent, env):
        super().__init__(config, agent, env)
        assert self.env.rolling, 'SegmentRunner must use rolling VecEnv'
        
        self.obs_buffer = None  # for next call
        self.done_buffer = None  # masking
        
    def __call__(self, T, reset=False):
        r"""Run the agent in the vectorized environment (one or multiple environments) and collect 
        a number of segments each with exactly T time steps. 
        
        .. note::
            
            One can continuously call this method to collect a rolling of segments for episodic transitions
            until :attr:`reset` set to be ``True``. 
        
        Args:
            T (int): number of time steps to collect
            reset (bool, optional): If ``True``, then reset all internal environments in VecEnv. 
                Default: ``False``
            
        Returns
        -------
        D : list
            a list of collected :class:`Segment`
        """
        D = [Segment() for _ in range(self.env.num_env)]
        
        # Reset environment if reset=True or first time call
        if self.obs_buffer is None or reset:
            self.obs_buffer = self.env.reset()
            self.done_buffer = [False]*self.env.num_env
            
            # reset agent: e.g. RNN states because initial observation
            self.agent.reset(self.config)
            
        for t in range(T):
            if any(self.done_buffer):
                info = {'mask': self.done_buffer}
            else:
                info = {}
            
            out_agent = self.agent.choose_action(self.obs_buffer, info=info)
            
            action = out_agent.pop('action')
            if torch.is_tensor(action):
                raw_action = list(action.detach().cpu().numpy())
            else:
                raw_action = action
            
            obs_next, reward, done, info = self.env.step(raw_action)
            self.done_buffer = done
            
            for i, segment in enumerate(D):
                transition = Transition(s=self.obs_buffer[i], 
                                        a=action[i], 
                                        r=reward[i], 
                                        s_next=obs_next[i], 
                                        done=done[i])
                
                # Record additional information
                [transition.add_info(key, val[i]) for key, val in out_agent.items()]
            
                segment.add_transition(transition)
            
            # Update self.obs_buffer as obs_next for next iteration to feed into agent
            # When done=True, use info['init_observation'] as initial observation
            # Because VecEnv automaticaly reset and continue with new episode
            for k in range(len(D)):  # iterate over each result
                if done[k]:  # terminated, use info['init_observation']
                    self.obs_buffer[k] = info[k]['init_observation']
                else:  # non-terminal, continue with obs_next
                    self.obs_buffer[k] = obs_next[k]

        return D
