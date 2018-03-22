from lagom.core.processor import CalcReturn


class Runner(object):
    """Data collection for an agent in an environment."""
    def __init__(self, agent, env, gamma):
        self.agent = agent
        self.env = env
        # Discount factor
        self.gamma = gamma
        
    def run(self, T, num_epi):
        """
        Run the agent in the environment and collect all necessary data
        
        Args:
            T (int): Number of time steps
            num_epi (int): Number of episodes
            
        Returns:
            batch_data (list of dict): Each dictionary indicates the data for one episode.
                                The keys of dictionary indicate different kinds of data.
        """
        batch_data = []
        for epi in range(num_epi):  # Iterate over the number of episodes
            # Dictionary for the data in current episode
            epi_data = {}
            # Initialize all necessary data
            epi_data['observations'] = []
            epi_data['actions'] = []
            epi_data['logprob_actions'] = []
            epi_data['state_values'] = []
            epi_data['rewards'] = []
            epi_data['returns'] = []
            epi_data['dones'] = []
            
            # Reset the environment
            obs = self.env.reset()
            # Record initial state
            epi_data['observations'].append(obs)
            
            for t in range(T):  # Iterate over the number of time steps
                # Agent chooses an action
                output_agent = self.agent.choose_action(self._make_input(obs))
                # Unpack dictionary of the output from the agent
                action = output_agent.get('action', None)
                logprob_action = output_agent.get('logprob_action', None)
                state_value = output_agent.get('state_value', None)
                # Record the output from the agent
                epi_data['actions'].append(action)
                epi_data['logprob_actions'].append(logprob_action)
                epi_data['state_values'].append(state_value)
                # Execute the action in the environment
                obs, reward, done, info = self.env.step(action)
                # Record data
                epi_data['observations'].append(obs)
                epi_data['rewards'].append(reward)
                epi_data['dones'].append(done)
                # Stop data collection once the episode terminated
                if done:
                    break
            # Calculate returns according to the rewards and gamma
            epi_data['returns'] = CalcReturn(self.gamma).process(epi_data['rewards'])
            
            # Record data for current episode
            batch_data.append(epi_data)
            
        return batch_data
    
    def _make_input(self, obs):
        """
        User-defined function to process the input data for the agent to choose action
        
        Args:
            obs (object): observations
            
        Returns:
            data (Variable): input data for the action selection
        
        """
        data = Variable(torch.FloatTensor(obs).unsqueeze(0))
        
        return data