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
    
    Each :class:`Segment` should store successive transitions i.e. :math:`(s_t, a_t, r_t, s_{t+1}, \text{done})`
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
    
    
    """
    def __init__(self, agent, env, gamma):
        super().__init__(agent=agent, env=env, gamma=gamma)
        
        # Buffer for observation (continuous with next call)
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
            # Action selection by the agent
            output_agent = self.agent.choose_action(self.obs_buffer)
            
            # Unpack action from output. 
            # We record Tensor dtype for backprop (propagate via Transitions)
            action = output_agent.pop('action')  # pop-out
            state_value = output_agent.pop('state_value', None)
            
            # Obtain raw action from Tensor for environment to execute
            if torch.is_tensor(action):
                raw_action = action.detach().cpu().numpy()
                raw_action = list(raw_action)
            else:  # Non Tensor action, e.g. from RandomAgent
                raw_action = action
            # Execute the action
            obs_next, reward, done, info = self.env.step(raw_action)
            
            # Iterate over all Segments to add transitions
            for i, segment in enumerate(D):
                # Create and record a Transition
                transition = Transition(s=self.obs_buffer[i], 
                                        a=action[i], 
                                        r=reward[i], 
                                        s_next=obs_next[i], 
                                        done=done[i])
                # Record state value if required
                if state_value is not None:
                    transition.add_info('V_s', state_value[i])
                # Record additional information from output_agent
                # Note that 'action' and 'state_value' already poped out
                for key, val in output_agent.items():
                    transition.add_info(key, val[i])
                    
                # Add transition to Segment
                segment.add_transition(transition)
                
            # Back up obs_next in self.obs_buffer for next iteration to feed into agent
            # Update the ones with done=True, use their info['init_observation']
            # Because VecEnv automatically reset and continue with new episode when done=True
            for k in range(len(D)):  # iterate over each result
                if done[k]:  # terminated, use info['init_observation']
                    self.obs_buffer[k] = info[k]['init_observation']
                else:  # non-terminal, continue with obs_next
                    self.obs_buffer[k] = obs_next[k]
            
        # Call agent again to compute state value for final observation in collected segment
        if state_value is not None:
            V_s_next = self.agent.choose_action(self.obs_buffer)['state_value']
            # Add V_s_next to final transitions in each segment
            for i, segment in enumerate(D):
                segment.transitions[-1].add_info('V_s_next', V_s_next[i])

        return D
