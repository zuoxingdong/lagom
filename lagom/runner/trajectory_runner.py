import torch

from .transition import Transition
from .trajectory import Trajectory

from lagom.envs.vec_env import VecEnv
from lagom.envs.spaces import Discrete


class TrajectoryRunner(object):
    """
    Batched data collection for an agent in one environment for a number of trajectories and a certain time steps. 
    It includes successive transitions (observation, action, reward, next observation, done) and 
    additional data useful for training the agent such as the action log-probabilities, policy entropies, 
    Q values etc.
    
    The collected data in each trajectory will be wrapped in an individual Trajectory object. Each call
    of the runner will return a list of Trajectory objects. 
    
    Note that the transitions in a Trajectory should come from a single episode and started from initial observation.
    The length of the trajectory can maximally be the allowed time steps or can be the time steps until it reaches
    terminal state. 
    
    For example, for a Trajectory with length 4, it can have either of following cases:
    
    Let s_t be state at time step t and s_T be terminal state.
    
    1. Part of single episode from initial observation: 
        s_0 -> s_1 -> s_2 -> s_3
    2. A complete episode:
        s_0 -> s_1 -> s_2 -> s_T
    
    For runner that collects transitions from multiple episodes, one can use SegmentRunner instead. 
    """
    def __init__(self, agent, env, gamma):
        """
        Args:
            agent (BaseAgent): agent
            env (Env): environment
            gamma (float): discount factor
        """
        self.agent = agent
        self.env = env
        assert not isinstance(self.env, VecEnv), 'The environment cannot be of type VecEnv. '
        self.gamma = gamma
        
    def __call__(self, N, T):
        """
        Run the agent in the environment and collect all necessary data for given number of trajectories
        and horizon (time steps) for each trajectory. 
        
        Args:
            N (int): Number of trajectories
            T (int): Number of time steps
            
        Returns:
            D (list of Trajectory): list of collected trajectories. 
        """ 
        D = []
        
        for n in range(N):  # Iterate over the number of trajectories
            # Create an trajectory object
            trajectory = Trajectory(gamma=self.gamma)
            
            # Reset the environment and returns initial state
            obs = self.env.reset()
            
            for t in range(T):  # Iterate over the number of time steps
                # Action selection by the agent
                output_agent = self.agent.choose_action(obs)
                
                # Unpack action from output. 
                # Note that do not convert to raw value from Tensor here
                # Because for Transition object, we often want to record Tensor action to backprop. 
                action = output_agent['action']
                state_value = output_agent.get('state_value', None)
                
                # Execute the action in the environment
                if torch.is_tensor(action):  # convert Tensor to raw numpy array
                    raw_action = action.detach().cpu().numpy()
                    # Handle with discrete action (must be int)
                    # Now raw action is ndarray
                    if isinstance(self.env.action_space, Discrete):
                        raw_action = raw_action.item()
                else:  # Non Tensor action, e.g. from RandomAgent
                    raw_action = action
                # Take action
                obs_next, reward, done, info = self.env.step(raw_action)
                
                # Create and record a Transition
                transition = Transition(s=obs, 
                                        a=action, 
                                        r=reward, 
                                        s_next=obs_next, 
                                        done=done)
                # Record state value if required
                if state_value is not None:
                    transition.add_info('V_s', state_value)
                # Record additional information from output_agent
                for key, val in output_agent.items():
                    if key != 'action' and key != 'state_value':  # action and state_value already recorded
                        transition.add_info(key, val)
                
                # Add transition to trajectory
                trajectory.add_transition(transition)
                
                # Back up obs for next iteration to feed into agent
                obs = obs_next
                
                # Terminate if episode finishes
                if done:
                    break
            
            # Call agent again to compute state value for final obsevation in collected trajectory
            if state_value is not None:
                V_s_next = self.agent.choose_action(obs)['state_value']
                # Set to zero if it is terminal state
                if done:
                    V_s_next = V_s_next.fill_(0.0)
                # Add to the final transition as 'V_s_next'
                trajectory.transitions[-1].add_info('V_s_next', V_s_next)
            
            # Append trajectory to data
            D.append(trajectory)

        return D
