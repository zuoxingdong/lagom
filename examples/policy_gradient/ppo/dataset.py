from torch.utils import data

from lagom.history.metrics import final_state_from_segment
from lagom.history.metrics import gae_from_segment


class Dataset(data.Dataset):
    def __init__(self, env_spec, observations, actions, logprobs, entropies, all_Vs, Qs, As):
        self.env_spec = env_spec
        # eliminate the last observation, to have consistent shape with others
        self.observations = observations[:, :-1, ...]
        self.actions = actions
        self.logprobs = logprobs.detach().cpu().numpy()
        self.entropies = entropies.detach().cpu().numpy()
        self.all_Vs = all_Vs.detach().cpu().numpy()
        self.Qs = Qs.detach().cpu().numpy()
        self.As = As.detach().cpu().numpy()
        
        N, T = self.logprobs.shape
        self.total_N = N*T
        
        # rolling batch [N, T, ...] to [N*T, ...]
        obs_shape = self.env_spec.observation_space.shape
        action_shape = self.env_spec.action_space.shape
        self.observations = self.observations.reshape([self.total_N, *obs_shape])
        self.actions = self.actions.reshape([self.total_N, *action_shape])
        self.logprobs = self.logprobs.flatten()
        self.entropies = self.entropies.flatten()
        self.all_Vs = self.all_Vs.flatten()
        self.Qs = self.Qs.flatten()
        self.As = self.As.flatten()
        
        assert self.observations.shape[0] == self.total_N
        assert self.actions.shape[0] == self.total_N
        assert all([item.shape == (self.total_N,) for item in [self.logprobs, self.entropies, 
                                                               self.all_Vs, self.Qs, self.As]])
        
    def __len__(self):
        return self.total_N
        
    def __getitem__(self, i):
        return self.observations[i], self.actions[i], self.logprobs[i], self.entropies[i], self.all_Vs[i], self.Qs[i], self.As[i]
