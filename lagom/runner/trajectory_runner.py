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
        assert isinstance(self.env, VecEnv), 'The environment must be of type VecEnv. '
        msg = f'expected only one environment for TrajectoryRunner, got {self.env.num_env}'
        assert self.env.num_env == 1, msg    
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
                # Not using numpy because we don't know exact dtype, all Agent should handle batched data
                output_agent = self.agent.choose_action(obs)
                
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
                
                # Create and record a Transition
                # Take out first element because we only have one environment wrapped for TrajectoryRunner
                transition = Transition(s=obs[0], 
                                        a=action[0], 
                                        r=reward[0], 
                                        s_next=obs_next[0], 
                                        done=done[0])
                # Record state value if required
                if state_value is not None:
                    transition.add_info('V_s', state_value[0])
                # Record additional information from output_agent
                # Note that 'action' and 'state_value' already poped out
                for key, val in output_agent.items():
                    transition.add_info(key, val[0])
                    
                # Add transition to Trajectory
                trajectory.add_transition(transition)
                
                # Back up obs for next iteration to feed into agent
                obs = obs_next
                
                # Terminate if episode finishes
                if done[0]:
                    break
            
            # Call agent again to compute state value for final obsevation in collected trajectory
            if state_value is not None:
                V_s_next = self.agent.choose_action(obs)['state_value']
                # We do not set zero even if it is terminal state
                # Because it should be handled in Trajectory e.g. compute TD errors
                # Return original Tensor in general can help backprop to work properly, e.g. learning value function
                # Add to the final transition as 'V_s_next'
                trajectory.transitions[-1].add_info('V_s_next', V_s_next[0])
            
            # Append trajectory to data
            D.append(trajectory)

        return D
