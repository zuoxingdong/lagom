import numpy as np

from lagom.transform import geometric_cumsum


def returns(rewards, gamma):
    return geometric_cumsum(gamma, rewards).tolist()[0]


def get_returns(D, gamma):
    out = np.zeros((D.N, D.T), dtype=np.float32)
    for n in range(D.N):
        y = []
        for r in D.r[n]:
            y += returns(r, gamma)
        out[n, :len(y)] = y
    return out
