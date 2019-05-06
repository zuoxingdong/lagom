import numpy as np

from torch.utils import data
from lagom.utils import numpify


class Dataset(data.Dataset):
    def __init__(self, D, logprobs, entropies, Vs, Qs, As):
        self.observations = np.concatenate([np.concatenate(traj.observations[:-1], 0) for traj in D], 0).astype(np.float32)
        self.actions = np.concatenate([traj.numpy_actions for traj in D], 0).astype(np.float32)
        self.logprobs = numpify(logprobs, 'float32')
        self.entropies = numpify(entropies, 'float32')
        self.Vs = numpify(Vs, 'float32')
        self.Qs = numpify(Qs, 'float32')
        self.As = numpify(As, 'float32')
        
        assert self.actions.shape[0] == len(self)
        assert all([item.shape == (len(self),) for item in [self.logprobs, self.entropies, 
                                                            self.Vs, self.Qs, self.As]])

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, i):
        D = (self.observations[i], self.actions[i], self.logprobs[i], 
             self.entropies[i], self.Vs[i], self.Qs[i], self.As[i])
        return D
