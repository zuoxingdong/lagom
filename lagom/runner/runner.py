import torch

from .transition import Transition
from .episode import Episode


class Runner(object):
    """
    Data collection for an agent in an environment.
    """
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
            batch (list of Episode): list of episodes, each is an object of Episode
        """
        batch = []
        for _ in range(num_epi):  # Iterate over the number of episodes
            # Create an episode object
            episode = Episode(self.gamma)
            
            # Reset the environment and returns initial state
            obs = self.env.reset()
            
            for t in range(T):  # Iterate over the number of time steps
                # Action selection by agent
                output_agent = self.agent.choose_action(self._make_input(obs))
                
                # Unpack output from agent
                action = output_agent['action']
                
                # Action execution in the environment
                obs_next, reward, done, info = self.env.step(action.item())  # item() retrieve raw data
                
                # Create a transition
                transition = Transition(s=obs, a=action, r=reward, s_next=obs_next)
                # Record more information about the transition
                transition.add_info('done', done)
                for key, val in output_agent.items():  # record other information from agent
                    if key != 'action':  # action already recorded
                        transition.add_info(key, val)
                
                # Add transition to the episode
                episode.add_transition(transition)
                
                # Terminate when episode finishes
                if done:
                    break
                    
                # Update obs after transition, for next iteration to feed into agent
                obs = obs_next
            
            # Append episode to the batch
            batch.append(episode)

        return batch
    
    def _make_input(self, obs):
        """
        User-defined function to process the input data for the agent to choose action
        
        Args:
            obs (object): observations
            
        Returns:
            data (Tensor): input data for the action selection
        
        """
        data = torch.Tensor(obs).unsqueeze(0)
        
        return data