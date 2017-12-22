from utils import calculate_return


class Runner(object):
    """Data collection for an agent in an environment."""
    def __init__(self, agent, env, gamma):
        self.agent = agent
        self.env = env
        
        self.gamma = gamma
        
    def run(self, num_step, num_epi):
        """
        Run the agent in the environment and collect all necessary data
        
        Args:
            num_step: Number of time steps
            num_epi: Number of episodes
            
        Returns:
            batch_data: A list of dictionaries. 
                        Each dictionary indicates the data for one episode.
                        The keys of dictionary indicate different kinds of data.
        """
        data_batch = []
        for epi in range(num_epi):  # Iterate over the number of episodes
            # Dictionary for the data in current episode
            epi_data = {}
            # Initialize all necessary data
            epi_data['states'] = []
            epi_data['rewards'] = []
            epi_data['returns'] = []
            epi_data['actions'] = []
            epi_data['logprob_actions'] = []
            epi_data['state_values'] = []
            epi_data['dones'] = []
            
            # Reset the environment
            state = self.env.reset()
            # Record state
            epi_data['states'].append(state)
            
            #done = False  # avoid Gym warning
            for t in range(num_step):  # Iterate over the number of time steps
                # Agent chooses an action
                output_agent = self.agent.choose_action(self._make_policy_input(state))
                # Unpack dictionary for the output from the agent
                action = output_agent.get('action', None)
                logprob_action = output_agent.get('log_prob', None)
                state_value = output_agent.get('state_value', None)
                # Record the output from the agent
                epi_data['actions'].append(action)
                epi_data['logprob_actions'].append(logprob_action)
                epi_data['state_values'].append(state_value)
                # Execute the action in the environment
                state, reward, done, info = self.env.step(action)
                # Record data
                epi_data['states'].append(state)
                epi_data['rewards'].append(reward)
                epi_data['dones'].append(done)
                # Stop data collection once the episode terminated
                if done:
                    break
            # Calculate returns according to the rewards and gamma
            epi_data['returns'] = calculate_return(epi_data['rewards'], self.gamma)
            
            ######
            # TODO: maybe erase the states after done, e.g. set to 0
            ######
            
            # Record data for current episode
            data_batch.append(epi_data)
            
        return data_batch
    
    def _make_policy_input(self, state):
        data = {}
        
        data['observation'] = state
        data['current_state'] = self.env.unwrapped.state
        if self.agent.policy.policy_type == 'goal':
            data['goal_state'] = self.env.goal_states
        
        return data