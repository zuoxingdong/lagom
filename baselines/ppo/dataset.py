import numpy as np

from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, D, logprobs, entropies, Vs, Qs, As):
        self.observations = np.concatenate([np.concatenate(traj.observations[:-1], 0) for traj in D], 0).astype(np.float32)
        self.actions = np.concatenate([traj.numpy_actions for traj in D], 0).astype(np.float32)
        self.logprobs = logprobs
        self.entropies = entropies
        self.Vs = Vs
        self.Qs = Qs
        self.As = As
        
        assert self.actions.shape[0] == len(self)
        assert all([item.shape == (len(self),) for item in [self.logprobs, self.entropies, 
                                                            self.Vs, self.Qs, self.As]])

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, i):
        D = (self.observations[i], self.actions[i], self.logprobs[i], 
             self.entropies[i], self.Vs[i], self.Qs[i], self.As[i])
        return D
