import torch

from .transition import Transition
from .trajectory import Trajectory

from .base_runner import BaseRunner


class TrajectoryRunner(BaseRunner):
    r"""Define a data collection interface by running the agent in an environment and collecting a batch of
    trajectories for a maximally allowed time steps. 
    
    .. note::
        
        By default, the agent handles batched data returned from :class:`VecEnv` type of environment.
        And the collected data is a list of :class:`Trajectory`. 
    
    Each :class:`Trajectory` should store successive transitions i.e. :math:`(s_t, a_t, r_t, s_{t+1}, \text{done}_t)`
    and all corresponding useful information such as log-probabilities of actions, state values, Q values etc.
    
    The collected transitions in each :class:`Trajectory` come from a single episode starting from initial observation
    until reaching maximally allowed time steps or reaching terminal states. For detailed description, see the docstring
    in :class:`Trajectory`. 
    
    .. note::
        
        For collecting batch of segments, one should use :class:`SegmentRunner` instead. 
    
    Example::
    
        >>> from lagom.agents import RandomAgent
        >>> from lagom.envs import make_envs, make_gym_env, EnvSpec
        >>> from lagom.envs.vec_env import SerialVecEnv
        >>> list_make_env = make_envs(make_env=make_gym_env, env_id='CartPole-v1', num_env=1, init_seed=0)
        >>> env = SerialVecEnv(list_make_env=list_make_env)
        >>> env_spec = EnvSpec(env)

        >>> agent = RandomAgent(env_spec=env_spec)
        >>> runner = TrajectoryRunner(agent=agent, env=env, gamma=1.0)

        >>> runner(N=2, T=3)
        [Trajectory: 
            Transition: (s=[-0.04002427  0.00464987 -0.01704236 -0.03673052], a=1, r=1.0, s_next=[-0.03993127  0.20001201 -0.01777697 -0.33474139], done=False)
            Transition: (s=[-0.03993127  0.20001201 -0.01777697 -0.33474139], a=1, r=1.0, s_next=[-0.03593103  0.39538239 -0.0244718  -0.63297681], done=False)
            Transition: (s=[-0.03593103  0.39538239 -0.0244718  -0.63297681], a=1, r=1.0, s_next=[-0.02802339  0.59083704 -0.03713133 -0.93326499], done=False),
         Trajectory: 
            Transition: (s=[-0.04892357  0.02011271  0.02775732 -0.04547827], a=1, r=1.0, s_next=[-0.04852131  0.21482587  0.02684775 -0.3292759 ], done=False)
            Transition: (s=[-0.04852131  0.21482587  0.02684775 -0.3292759 ], a=0, r=1.0, s_next=[-0.0442248   0.01933221  0.02026223 -0.0282488 ], done=False)
            Transition: (s=[-0.0442248   0.01933221  0.02026223 -0.0282488 ], a=1, r=1.0, s_next=[-0.04383815  0.21415782  0.01969726 -0.31447053], done=False)]        

    """
    def __init__(self, agent, env, gamma):
        super().__init__(agent=agent, env=env, gamma=gamma)
        assert self.env.num_env == 1, f'expected a single environment, got {self.env.num_env}'
        
    def __call__(self, N, T):
        r"""Run the agent in the environment and collect N trajectories each with maximally T time steps. 
        
        Args:
            N (int): number of trajectories to collect. 
            T (int): maximally allowed time steps. 
            
        Returns
        -------
        D : list
            a list of collected :class:`Trajectory`
        """ 
        D = []
        
        for n in range(N):  # Iterate over the number of trajectories
            # Create an trajectory object
            trajectory = Trajectory(gamma=self.gamma)
            
            # Reset the environment and returns initial state
            obs = self.env.reset()
            
            for t in range(T):  # Iterate over the number of time steps
                # Action selection by the agent (handles batched data)
                out_agent = self.agent.choose_action(obs)
                
                # Unpack action
                action = out_agent.pop('action')  # pop-out
                # Get raw action if Tensor dtype for feeding the environment
                if torch.is_tensor(action):
                    raw_action = list(action.detach().cpu().numpy())
                else:  # Non Tensor action, e.g. from RandomAgent
                    raw_action = action
                
                # Execute the agent in the environment
                obs_next, reward, done, info = self.env.step(raw_action)
                
                # Create and record a Transition
                # Take out first elements because of VecEnv with single environment (no batch dim in Transition)
                # Note that action can be Tensor type (can be used for backprop)
                transition = Transition(s=obs[0], 
                                        a=action[0], 
                                        r=reward[0], 
                                        s_next=obs_next[0], 
                                        done=done[0])
                
                # Handle state value if available
                state_value = out_agent.pop('state_value', None)
                if state_value is not None:
                    transition.add_info('V_s', state_value[0])
                    
                # Record additional information from out_agent to transitions
                # Note that 'action' and 'state_value' already poped out
                [transition.add_info(key, val[0]) for key, val in out_agent.items()]
                
                # Add transition to Trajectory
                trajectory.add_transition(transition)
                
                # Back up obs for next iteration to feed into agent
                obs = obs_next
                
                # Terminate if episode finishes
                if done[0]:
                    break
                    
            # If state value available, calculate state value for final observation
            if state_value is not None:
                V_s_next = self.agent.choose_action(obs)['state_value']
                # Add to final transition in the trajectory
                # Do not set zero for terminal state, useful for backprop to learn value function
                # It will be handled in Trajectory or Segment method when calculating things like returns/TD errors
                trajectory.transitions[-1].add_info('V_s_next', V_s_next[0])
                
            # Append trajectory to data
            D.append(trajectory)

        return D
