


class VPGAgent(BaseAgent):
    r"""Vanilla Policy Gradient (VPG) with value network (baseline), no bootstrapping to estimate value function. """
    def __init__(self, config, device, policy, optimizer, **kwargs):
        self.policy = policy
        self.optimizer = optimizer
        
        super().__init__(config, device, **kwargs)
        
        # accumulated trained timesteps
        self.total_T = 0
        
    def choose_action(self, obs, info={}):
        if not torch.is_tensor(obs):
            obs = np.asarray(obs)
            assert obs.ndim >= 2, f'expected at least 2-dim for batched data, got {obs.ndim}'
            obs = torch.from_numpy(obs).float().to(self.device)
            
        if self.policy.recurrent and self.info['reset_rnn_states']:
            self.policy.reset_rnn_states(batch_size=obs.size(0))
            self.info['reset_rnn_states'] = False  # Done, turn off
            
        out_policy = self.policy(obs, 
                                 out_keys=['action', 'action_logprob', 'state_value', 
                                           'entropy', 'perplexity'], 
                                 info=info)
        
        return out_policy
        
    def learn(self, D, info={}):
        batch_policy_loss = []
        batch_entropy_loss = []
        batch_value_loss = []
        batch_total_loss = []
        
        for trajectory in D:
            logprobs = trajectory.all_info('action_logprob')
            entropies = trajectory.all_info('entropy')
            Qs = trajectory.all_discounted_returns
            
            # Standardize: encourage/discourage half of performed actions
            if self.config['agent.standardize_Q']:
                Qs = Standardize()(Qs).tolist()
            
            # State values
            Vs = trajectory.all_info('V_s')
            final_V = trajectory.transitions[-1].V_s_next
            final_done = trajectory.transitions[-1].done
            
            # Advantage estimates
            As = [Q - V.item() for Q, V in zip(Qs, Vs)]
            if self.config['agent.standardize_adv']:
                As = Standardize()(As).tolist()
            
            # Estimate policy gradient for all time steps and record all losses
            policy_loss = []
            entropy_loss = []
            value_loss = []
            for logprob, entropy, A, Q, V in zip(logprobs, entropies, As, Qs, Vs):
                policy_loss.append(-logprob*A)
                entropy_loss.append(-entropy)
                value_loss.append(F.mse_loss(V, torch.tensor(Q).view_as(V).to(V.device)))
            if final_done:  # learn terminal state value as zero
                value_loss.append(F.mse_loss(final_V, torch.tensor(0.0).view_as(V).to(V.device)))    
            
            # Average losses over all time steps
            policy_loss = torch.stack(policy_loss).mean()
            entropy_loss = torch.stack(entropy_loss).mean()
            value_loss = torch.stack(value_loss).mean()
            
            # Calculate total loss
            entropy_coef = self.config['agent.entropy_coef']
            value_coef = self.config['agent.value_coef']
            total_loss = policy_loss + value_coef*value_loss + entropy_coef*entropy_loss
            
            # Record all losses
            batch_policy_loss.append(policy_loss)
            batch_entropy_loss.append(entropy_loss)
            batch_value_loss.append(value_loss)
            batch_total_loss.append(total_loss)
                
        # Average loss over list of Trajectory
        policy_loss = torch.stack(batch_policy_loss).mean()
        entropy_loss = torch.stack(batch_entropy_loss).mean()
        value_loss = torch.stack(batch_value_loss).mean()
        loss = torch.stack(batch_total_loss).mean()
            
        # Train with estimated policy gradient
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config['agent.max_grad_norm'] is not None:
            clip_grad_norm_(parameters=self.policy.network.parameters(), 
                            max_norm=self.config['agent.max_grad_norm'], 
                            norm_type=2)
        
        if hasattr(self, 'lr_scheduler'):
            if 'train.iter' in self.config:  # iteration-based
                self.lr_scheduler.step()
            elif 'train.timestep' in self.config:  # timestep-based
                self.lr_scheduler.step(self.total_T)
            else:
                raise KeyError('expected `train.iter` or `train.timestep` in config, but got none of them')
        
        self.optimizer.step()
        
        # Accumulate trained timesteps
        self.total_T += sum([trajectory.T for trajectory in D])
        
        out = {}
        out['loss'] = loss.item()
        out['policy_loss'] = policy_loss.item()
        out['entropy_loss'] = entropy_loss.item()
        out['value_loss'] = value_loss.item()
        if hasattr(self, 'lr_scheduler'):
            out['current_lr'] = self.lr_scheduler.get_lr()

        return out
    
    def save(self, f):
        self.policy.network.save(f)
    
    def load(self, f):
        self.policy.network.load(f)
