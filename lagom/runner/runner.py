import torch

from .transition import Transition
from .trajectory import Trajectory

from lagom.envs.spaces import Discrete


class Runner(object):
    """
    Data collection for an agent in an environment.
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
        # Discount factor
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
                # Record additional information from output_agent
                for key, val in output_agent.items():
                    if key != 'action':  # action already recorded
                        transition.add_info(key, val)
                
                # Add transition to trajectory
                trajectory.add_transition(transition)
                
                # Terminate if episode finishes
                if done:
                    break
                    
                # Back up obs for next iteration to feed into agent
                obs = obs_next
            
            # Append trajectory to data
            D.append(trajectory)

        return D