import numpy as np

import torch

from lagom.engine import BaseEngine


class Engine(BaseEngine):
    """
    Engine for training policy gradient for some number of iterations
    """
    def train(self):
        # Training output
        train_output = {}
        train_output['returns'] = []
        
        # Set network as training mode
        self.agent.policy.network.train()
        
        for i in range(self.config['train_iter']):  # training iterations
            # Collect a list of trajectories
            D = self.runner(N=self.config['N'], T=self.config['T'])
            
            # Train agent with given data
            out_agent = self.agent.learn(D)
            
            # Unpack agent's learning outputs
            # Use items() to save memory
            loss = out_agent['loss'].item()
            policy_loss = torch.stack(out_agent['batch_policy_loss']).mean().item()
            if 'batch_value_loss' in out_agent:
                value_loss = torch.stack(out_agent['batch_value_loss']).mean().item()
            entropy_loss = torch.stack(out_agent['batch_entropy_loss']).mean().item()
            
            # Unpack useful information from trajectory list
            all_returns = [trajectory.all_returns[0] for trajectory in D]
            all_discounted_returns = [trajectory.all_discounted_returns[0] for trajectory in D]
            
            # Record training output for each iteration
            train_output['returns'].append(np.mean(all_returns))
            
            # Loggins
            if i == 0 or (i+1) % self.config['log_interval'] == 0:
                print('-'*50)
                print(f'Training iteration: {i+1}')
                print('-'*50)
                
                if 'current_lr' in out_agent:
                    print(f'Current lr: {out_agent["current_lr"]}')
                print(f'Loss: {loss}')
                print(f'Policy loss: {policy_loss}')
                if 'value_loss' in locals():  # if value_loss is defined above within train()
                    print(f'Value loss: {value_loss}')
                print(f'Entropy loss: {entropy_loss}')
                print(f'Number of trajectories: {len(D)}')
                print(f'Average Return: {np.mean(all_returns)}')
                print(f'Average Discounted Return: {np.mean(all_discounted_returns)}')
                print(f'Std Return: {np.std(all_returns)}')
                print(f'Min Return: {np.min(all_returns)}')
                print(f'Max Return: {np.max(all_returns)}')
                
                print('-'*50)
                
        return train_output
        
    def eval(self):
        pass
