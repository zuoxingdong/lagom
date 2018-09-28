import numpy as np

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
        
        For collecting batch of rolling segments, one should use :class:`SegmentRunner` instead. 
    
    Example::
    
        >>> from lagom.agents import RandomAgent
        >>> from lagom.envs import make_gym_env, make_vec_env, EnvSpec
        >>> from lagom.envs.vec_env import SerialVecEnv

        >>> env = make_vec_env(vec_env_class=SerialVecEnv, make_env=make_gym_env, env_id='CartPole-v1', num_env=3, init_seed=0, rolling=False)
        >>> env_spec = EnvSpec(env)
        >>> agent = RandomAgent(config=None, env_spec=env_spec)
        >>> runner = TrajectoryRunner(agent=agent, env=env, gamma=0.99)

        >>> runner(T=2)
        [Trajectory: 
            Transition: (s=[ 0.01712847  0.01050531  0.00494782 -0.02698731], a=1, r=1.0, s_next=[ 0.01733857  0.20555595  0.00440808 -0.31810505], done=False)
            Transition: (s=[ 0.01733857  0.20555595  0.00440808 -0.31810505], a=0, r=1.0, s_next=[ 0.02144969  0.0103715  -0.00195402 -0.02403524], done=False),
         Trajectory: 
            Transition: (s=[ 0.01086211 -0.00049396 -0.04967628 -0.04839385], a=0, r=1.0, s_next=[ 0.01085223 -0.1948697  -0.05064416  0.22821125], done=False)
            Transition: (s=[ 0.01085223 -0.1948697  -0.05064416  0.22821125], a=1, r=1.0, s_next=[ 0.00695484  0.00093803 -0.04607993 -0.08000678], done=False),
         Trajectory: 
            Transition: (s=[-0.00990306 -0.00393545 -0.02013421  0.01405226], a=1, r=1.0, s_next=[-0.00998177  0.19146938 -0.01985317 -0.28491463], done=False)
            Transition: (s=[-0.00998177  0.19146938 -0.01985317 -0.28491463], a=1, r=1.0, s_next=[-0.00615238  0.38686877 -0.02555146 -0.58379241], done=False)]

    """
    def __init__(self, agent, env, gamma):
        super().__init__(agent=agent, env=env, gamma=gamma)
        assert not self.env.rolling, 'TrajectoryRunner should not use rolling VecEnv'
        
    def __call__(self, T):
        r"""Run the agent in the vectorized environment (one or multiple environments) and collect
        a number of trajectories each with maximally T time steps. 
        
        Args:
            T (int): maximally allowed time steps. 
            
        Returns
        -------
        D : list
            a list of collected :class:`Trajectory`
        """ 
        # Initialize all Trajectory for each environment
        D = [Trajectory(gamma=self.gamma) for _ in range(self.env.num_env)]
        
        # Reset all environments and returns all initial observations
        obs = self.env.reset()
        # Inform agent to reset RNN states (if valid): TrajectoryRunner always starts from initial observation
        self.agent.update_info('reset_rnn_states', True)  # turn it off after reset
        
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
            
            # Unpack state value if available
            state_value = out_agent.pop('state_value', None)
            
            # Execute the action in the environment
            obs_next, reward, done, info = self.env.step(raw_action)
            
            # One more forward pass to get state value for V_s_next if required
            if state_value is not None and any(done):  # require state value and at least one done=True
                list_V_s_next = self.agent.choose_action(obs_next, 
                                                         info={'rnn_state_no_update': True})['state_value']
            
            # Iterate over all Trajectory to add data of transitions
            for i, trajectory in enumerate(D):
                if done[i] is None:  # already stopped because an episode terminated before
                    obs_next[i] = np.zeros([self.env.observation_space.flat_dim], dtype=np.float32)
                else:  # non-terminating, store transition
                    # Create and record a Transition
                    # Retrieve the corresponding values for each Trajectory
                    transition = Transition(s=obs[i], 
                                            a=action[i], 
                                            r=reward[i], 
                                            s_next=obs_next[i], 
                                            done=done[i])

                    # Record state value if available
                    if state_value is not None:
                        transition.add_info('V_s', state_value[i])
                        if done[i]:  # add terminal state value
                            transition.add_info('V_s_next', list_V_s_next[i])
                    
                    # Record additional information from out_agent to transitions
                    # Note that 'action' and 'state_value' already poped out
                    [transition.add_info(key, val[i]) for key, val in out_agent.items()]

                    # Add transition to Trajectory
                    trajectory.add_transition(transition)
                
            # Back up obs for next iteration to feed into agent
            obs = obs_next

            # Terminate if all sub-environments finished before max allowed timesteps
            if all([x is None or x for x in done]):
                break
        
        # Calculate and record last state value for each trajectory
        if state_value is not None:
            list_V_s_next = self.agent.choose_action(obs_next, 
                                                     info={'rnn_state_no_update': True})['state_value']
            for i, trajectory in enumerate(D):  # check each trajectory
                if 'V_s_next' not in trajectory.transitions[-1].info:  # missing last state value
                    trajectory.transitions[-1].add_info('V_s_next', list_V_s_next[i])
        
        return D
