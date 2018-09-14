import torch

from lagom.runner import Transition
from lagom.runner import Segment

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
        >>> env = SerialVecEnv(list_make_env=list_make_env)
        >>> env_spec = EnvSpec(env)

        >>> agent = RandomAgent(env_spec=env_spec)
        >>> runner = SegmentRunner(agent=agent, env=env, gamma=1.0)

        >>> runner(T=3, reset=False)
        [Segment: 
            Transition: (s=[-0.04002427  0.00464987 -0.01704236 -0.03673052], a=0, r=1.0, s_next=[-0.03993127 -0.1902236  -0.01777697  0.25052702], done=False)
            Transition: (s=[-0.03993127 -0.1902236  -0.01777697  0.25052702], a=1, r=1.0, s_next=[-0.04373575  0.00514764 -0.01276643 -0.04770968], done=False)
            Transition: (s=[-0.04373575  0.00514764 -0.01276643 -0.04770968], a=0, r=1.0, s_next=[-0.04363279 -0.18978895 -0.01372062  0.24091814], done=False),
         Segment: 
            Transition: (s=[ 0.00854682  0.00830137 -0.03052506  0.03439879], a=0, r=1.0, s_next=[ 0.00871284 -0.18636984 -0.02983709  0.31729661], done=False)
            Transition: (s=[ 0.00871284 -0.18636984 -0.02983709  0.31729661], a=0, r=1.0, s_next=[ 0.00498545 -0.38105439 -0.02349115  0.60042265], done=False)
            Transition: (s=[ 0.00498545 -0.38105439 -0.02349115  0.60042265], a=1, r=1.0, s_next=[-0.00263564 -0.18561182 -0.0114827   0.30043392], done=False)]
            
        >>> runner(T=1, reset=False)
        [Segment: 
            Transition: (s=[-0.04363279 -0.18978895 -0.01372062  0.24091814], a=1, r=1.0, s_next=[-0.04742857  0.00552629 -0.00890226 -0.05606087], done=False),
         Segment: 
            Transition: (s=[-0.00263564 -0.18561182 -0.0114827   0.30043392], a=1, r=1.0, s_next=[-0.00634788  0.0096719  -0.00547402  0.00415181], done=False)]
        
    """
    def __init__(self, agent, env, gamma):
        super().__init__(agent=agent, env=env, gamma=gamma)
        
        # Buffer for current observation (continuous with next call)
        self.obs_buffer = None
        
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
        # Initialize all Segment for each environment
        D = [Segment(gamma=self.gamma) for _ in range(self.env.num_env)]
        
        # Reset the environment and returns initial state if reset=True or first time call
        if self.obs_buffer is None or reset:
            self.obs_buffer = self.env.reset()
            
        # Iterate over the number of time steps
        for t in range(T):
            # Action selection by the agent (handles batched data)
            out_agent = self.agent.choose_action(self.obs_buffer)
            
            # Unpack action
            action = out_agent.pop('action')  # pop-out
            # Get raw action if Tensor dtype for feeding the environment
            if torch.is_tensor(action):
                raw_action = list(action.detach().cpu().numpy())
            else:  # Non Tensor action, e.g. from RandomAgent
                raw_action = action
                
            # Unpack state value if available
            state_value = out_agent.pop('state_value', None)
            
            # Execute the action in the environment
            obs_next, reward, done, info = self.env.step(raw_action)
            
            # Iterate over all Segments to add data to transitions
            for i, segment in enumerate(D):
                # Create and record a Transition
                # Retrieve the corresponding values for each Segment
                transition = Transition(s=self.obs_buffer[i], 
                                        a=action[i], 
                                        r=reward[i], 
                                        s_next=obs_next[i], 
                                        done=done[i])
                
                # Record state value if available
                if state_value is not None:
                    transition.add_info('V_s', state_value[i])
                
                # Record additional information from out_agent to transitions
                # Note that 'action' and 'state_value' already poped out
                [transition.add_info(key, val[i]) for key, val in out_agent.items()]
            
                # Add transition to Segment
                segment.add_transition(transition)
            
            # Update self.obs_buffer as obs_next for next iteration to feed into agent
            # When done=True, use info['init_observation'] as initial observation
            # Because VecEnv automaticaly reset and continue with new episode
            for k in range(len(D)):  # iterate over each result
                if done[k]:  # terminated, use info['init_observation']
                    self.obs_buffer[k] = info[k]['init_observation']
                else:  # non-terminal, continue with obs_next
                    self.obs_buffer[k] = obs_next[k]
            
        # If state value available, calculate state value for final observation
        # Note that do NOT use self.obs_buffer, because it contains initial observation for new episode
        # but we want to have state value for terminal states instead
        if state_value is not None:
            V_s_next = self.agent.choose_action(obs_next)['state_value']
            # Add to final transition in each Segment
            # Do not set zero for terminal state, useful for backprop to learn value function
            # It will be handled in Trajectory or Segment method when calculating things like returns/TD errors
            for i, segment in enumerate(D):
                segment.transitions[-1].add_info('V_s_next', V_s_next[i])

        return D
