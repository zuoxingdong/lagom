import numpy as np

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    """
    Engine for training policy gradient for some number of iterations
    """
    def train(self):
        # Set network as training mode
        self.agent.policy.network.train()
        
        for i in range(self.config['train_iter']):  # training iterations
            # Collect a list of trajectories
            D = self.runner(N=self.config['N'], T=self.config['T'])
            
            # Train agent with given data
            out_agent = self.agent.learn(D)
            
            # Unpack agent's learning outputs
            total_loss = out_agent['total_loss'].item()  # item saves memory
            
            # Unpack useful information from trajectory list
            all_returns = [trajectory.all_returns[0] for trajectory in D]
            all_discounted_returns = [trajectory.all_discounted_returns[0] for trajectory in D]
            
            # Loggins
            if i == 0 or (i+1) % self.config['log_interval'] == 0:
                print('-'*50)
                print(f'Training iteration: {i+1}')
                print('-'*50)
                
                print(f'Total loss: {total_loss}')
                print(f'Number of trajectories: {len(D)}')
                print(f'Average Return: {np.mean(all_returns)}')
                print(f'Average Discounted Return: {np.mean(all_discounted_returns)}')
                print(f'Std Return: {np.std(all_returns)}')
                print(f'Min Return: {np.min(all_returns)}')
                print(f'Max Return: {np.max(all_returns)}')
                
               
                print('-'*50)
        
    def eval(self):
        pass
        
    