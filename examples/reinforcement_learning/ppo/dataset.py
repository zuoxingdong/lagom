from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, D, logprobs, entropies, Vs, Qs, As):
        self.D = D
        self.observations = D.batch_observations
        self.actions = D.batch_actions
        self.logprobs = logprobs.detach().cpu().numpy()
        self.entropies = entropies.detach().cpu().numpy()
        self.Vs = Vs.detach().cpu().numpy()
        self.Qs = Qs.detach().cpu().numpy()
        self.As = As.detach().cpu().numpy()
        
        # rolling batch [N, T, ...] to [N*T, ...]
        self.observations = self.observations.reshape([len(self), *D.env.observation_space.shape])
        self.actions = self.actions.reshape([len(self), *D.env.action_space.shape])
        self.logprobs = self.logprobs.flatten()
        self.entropies = self.entropies.flatten()
        self.Vs = self.Vs.flatten()
        self.Qs = self.Qs.flatten()
        self.As = self.As.flatten()
        
        assert self.observations.shape[0] == len(self)
        assert self.actions.shape[0] == len(self)
        assert all([item.shape == (len(self),) for item in [self.logprobs, self.entropies, 
                                                            self.Vs, self.Qs, self.As]])

    def __len__(self):
        return self.D.N*self.D.T
        
    def __getitem__(self, i):
        return self.observations[i], self.actions[i], self.logprobs[i], self.entropies[i], self.Vs[i], self.Qs[i], self.As[i]
