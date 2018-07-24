import torch

from lagom.runner import Transition
from lagom.runner import Segment

from lagom.envs.vec_env import VecEnv


class SegmentRunner(object):
    """
    Batched data collection for an agent in one or multiple environments for a certain time steps. 
    It includes successive transitions (observation, action, reward, next observation, done) and 
    additional data useful for training the agent such as the action log-probabilities, policy entropies, 
    Q values etc.
    
    The collected data in each environment will be wrapped in an individual Segment object. Each call
    of the runner will return a list of Segment objects each with length same as the number of time steps.
    
    Note that we allow transitions coming from multiple episodes successively. 
    For example, for a Segment with length 4, it can have either of following cases:
    
    Let s_t^i be state at time step t in episode i and s_T^i be terminal state in episode i. 
    
    1. Part of single episode from initial observation: 
        s_0^1 -> s_1^1 -> s_2^1 -> s_3^1
    2. A complete episode:
        s_0^1 -> s_1^1 -> s_2^1 -> s_T^1
    3. Intermediate part of a single episode:
        s_5^1 -> s_6^1 -> s_7^1 -> s_8^1
    4. Two complete episodes:
        s_0^1 -> s_T^1 -> s_0^2 -> s_T^2
    5. Parts from two episodes:
        s_3^1 -> s_T^1 -> s_0^2 -> s_1^2
        
    Be aware that if the transitions coming from more than one episode, then the episodes should be
    in a successive order, and all preceding episodes but the last one should reach a terminal state
    before starting the new episode and each succeeding episode starts from initial observation. 
    
    In order to make such data collection possible, the environment must be of type VecEnv to support 
    batched data. VecEnv will continuously collect data in all environment, for each occurrence of 
    `done=True`, the environment will be automatically reset and continue. So if we want to collect data
    from initial observation in all environments, the method `reset` should be called. 
    
    The SegmentRunner is very general, for runner that only collects transitions from a single 
    episode (start from initial observation) one can use TrajectoryRunner instead. 
    """
    def __init__(self, agent, env, gamma):
        self.agent = agent
        self.env = env
        assert isinstance(self.env, VecEnv), 'The environment must be of type VecEnv. '
        self.gamma = gamma
        
        # Buffer for observation (continuous with next call)
        self.obs_buffer = None
        
    def __call__(self, T, reset=False):
        """
        Run the agent in the batched environments and collect all necessary data for given number of 
        time steps for each Segment (one Segment for each environment). 
        
        Note that we do not reset all environments for each call as it does in TrajectoryRunner. 
        An option `reset` is provided to decide if reset all environment before data collection.
        
        This is because for SegmentRunner, we often need to continuously collect a batched data 
        with small time steps, so each `__call__` will continuously collect data until `reset=True`.
        
        Args:
            T (int): Number of time steps
            reset (bool): Whether to reset all environments (in VecEnv). 
            
        Returns:
            D (list of Segment): list of collected segments. 
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
            else:
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
