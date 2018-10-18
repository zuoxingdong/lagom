
    @property
    def all_bootstrapped_returns(self):
        r"""Return a list of accumulated returns (no discount, gamma=1.0) with bootstrapping
        for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            Q_t = r_t + r_{t+1} + \dots + r_T + V(s_{T+1})
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
        
    @property
    def all_bootstrapped_discounted_returns(self):
        r"""Return a list of discounted returns with bootstrapping for all time steps. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)`, it computes
        
        .. math::
            Q_t = r_t + \gamma r_{t+1} + \dots + \gamma^{T - t} r_T + \gamma^{T - t + 1} V(s_{T+1})
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
    
    @property
    def all_V(self):
        r"""Return a list of all state values in the history and the value for final state separately.
        The final state is a pair of itself and ``done`` indicating whether the episode terminates. 
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
            
            It returns Tensor dtype, used for backprop to train value function. It does not set
            zero value for terminal state !
            
        """
        raise NotImplementedError
    
    @property
    def all_TD(self):
        r"""Return a list of all TD errors in the history. 
        
        Formally, suppose we have all rewards :math:`(r_1, \dots, r_T)` and all state
        values :math:`(V(s_1), \dots, V(s_T), V(s_{T+1}))`, it computes
        
        .. math::
            \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
            
        """
        raise NotImplementedError
    
    def all_GAE(self, gae_lambda):
        r"""Return a list of all `generalized advantage estimates`_ (GAE) in the history.
        
        .. note::
        
            The state value for terminal state is set as zero !
        
        .. note::
        
            This behaves differently for :class:`Trajectory` and :class:`Segment`. 
            
        .. note::
        
            It returns raw values instead of Tensor dtype, not to be used for backprop. 
        
        .. _generalized advantage estimates:
            https://arxiv.org/abs/1506.02438
        """
        raise NotImplementedError
        
        
        
        

    def _rewards_with_bootstrapping(self):
        # Get last state value and last done
        last_V = self.transitions[-1].V_s_next
        last_done = self.transitions[-1].done
        # Get raw value if Tensor dtype
        if torch.is_tensor(last_V):
            last_V = last_V.item()
        assert isinstance(last_V, float), f'expected float dtype, got {type(last_V)}'
        
        # Set zero value if terminal state
        if last_done:
            last_V = 0.0
            
        return self.all_r + [last_V]
    
    @property
    def all_bootstrapped_returns(self):
        bootstrapped_rewards = self._rewards_with_bootstrapping()
        
        out = ExpFactorCumSum(1.0)(bootstrapped_rewards)
        # Take out last one, because it is just last state value itself
        out = out[:-1]
        
        return out
        
    @property
    def all_bootstrapped_discounted_returns(self):
        bootstrapped_rewards = self._rewards_with_bootstrapping()
        
        out = ExpFactorCumSum(self.gamma)(bootstrapped_rewards)
        # Take out last one, because it is just last state value itself
        out = out[:-1]
        
        return out
    
    @property
    def all_V(self):
        final = [self.transitions[-1].V_s_next, self.transitions[-1].done]
        return [transition.V_s for transition in self.transitions], final
    
    @property
    def all_TD(self):
        # Get all rewards
        all_r = np.array(self.all_r)
        
        # Get all state values with raw values if Tensor dtype
        all_V = self.all_V
        all_V = all_V[0] + [all_V[1][0]]  # unpack to state values from first to last
        all_V = np.array([v.item() if torch.is_tensor(v) else v for v in all_V])
        # Set last state value as zero if terminal state
        if self.all_done[-1]:
            all_V[-1] = 0.0
        
        # Unpack state values into current and next time step
        all_V_s = all_V[:-1]
        all_V_s_next = all_V[1:]
        
        # Calculate TD error
        all_TD = all_r + self.gamma*all_V_s_next - all_V_s
        
        return all_TD.astype(np.float32).tolist()
    
    def all_GAE(self, gae_lambda):
        # TODO: implement it + add to test_runner
        raise NotImplementedError

        
        
        
        
        
        

    @property
    def all_bootstrapped_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_bootstrapped_returns for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_bootstrapped_discounted_returns(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_bootstrapped_discounted_returns for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out
    
    @property
    def all_V(self):
        out, all_final = zip(*[trajectory.all_V for trajectory in self.trajectories])
        
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = list(chain.from_iterable(out))
        
        return out, all_final
    
    @property
    def all_TD(self):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_TD for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out

    def all_GAE(self, gae_lambda):
        # Use itertools.chain().from_iterable, similar reason with doc in `transitions(self)`
        out = [trajectory.all_GAE(gae_lambda) for trajectory in self.trajectories]
        out = list(chain.from_iterable(out))
        
        return out