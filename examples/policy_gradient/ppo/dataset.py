import torch
from torch.utils import data

from lagom.history.metrics import final_state_from_segment
from lagom.history.metrics import gae_from_segment


class Dataset(data.Dataset):
    def __init__(self, config, D, policy):
        assert isinstance(D, list)
        self.D = D
        self.policy = policy
        
        self.states = []
        self.logprobs = []
        self.As = []
        self.Vs = []
        self.Qs = []
        
        for segment in D:
            all_s, all_final = segment.all_s
            self.states += all_s
            
            logprobs = segment.all_info('action_logprob')
            logprobs = [logprob.detach() for logprob in logprobs]
            self.logprobs += logprobs
            
            all_Vs = [traj.all_info('V') for traj in segment.trajectories]
            final_states = final_state_from_segment(segment)
            final_states = torch.tensor(final_states).float().to(policy.device)
            all_V_last = self.policy(final_states)['V'].cpu().detach().numpy()
            As = gae_from_segment(segment, all_Vs, all_V_last, config['algo.gamma'], config['algo.gae_lam'])
            self.As += As
            
            Vs = segment.all_info('V')
            Vs = [V.detach() for V in Vs]
            self.Vs += Vs
            
            Qs = [A + V.item() for A, V in zip(As, Vs)]
            self.Qs += Qs
        
        self.N = len(self.states)
        assert len(self.logprobs) == self.N
        assert len(self.As) == self.N
        assert len(self.Vs) == self.N
        assert len(self.Qs) == self.N
        
        self.states = torch.tensor(self.states).float().detach()
        self.logprobs = torch.stack(self.logprobs).cpu().detach()
        self.As = torch.tensor(self.As).float().detach()
        self.Vs = torch.cat(self.Vs).cpu().detach()
        self.Qs = torch.tensor(self.Qs).float().detach()
        
        assert len(self.states.shape) == 2
        assert all([len(x.shape) == 1 for x in [self.logprobs, self.As, self.Vs, self.Qs]])
        
    def __len__(self):
        return self.N
        
    def __getitem__(self, i):
        return self.states[i], self.logprobs[i], self.As[i], self.Vs[i], self.Qs[i]
