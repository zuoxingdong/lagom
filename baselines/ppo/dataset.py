import numpy as np
from torch.utils import data


class Dataset(data.Dataset):
    def __init__(self, D, logprobs, entropies, Vs, Qs, As):
        self.D = D
        self.observations = np.concatenate([np.stack(traj.observations[:-1], 0) for traj in D], 0).astype(np.float32)
        self.actions = np.concatenate([np.stack(traj.actions, 0) for traj in D], 0).astype(np.float32)
        self.logprobs = logprobs.detach()
        self.entropies = entropies.detach()
        self.Vs = Vs.detach()
        self.Qs = Qs.detach()
        self.As = As.detach()
        assert self.observations.shape[0] == len(self) and self.actions.shape[0] == len(self)
        assert all([i.shape == (len(self),) for i in [self.logprobs, self.entropies, self.Vs, self.Qs, self.As]])

    def __len__(self):
        return sum([traj.T for traj in self.D])

    def __getitem__(self, i):
        D = (self.observations[i], self.actions[i], self.logprobs[i], 
             self.entropies[i], self.Vs[i], self.Qs[i], self.As[i])
        return D
