import numpy as np

from lagom.transform import ExpFactorCumSum


def returns(rewards, gamma):
    f = ExpFactorCumSum(gamma)
    return f(rewards).tolist()[0]


def get_returns(D, gamma):
    out = np.zeros((D.N, D.T), dtype=np.float32)
    for n in range(D.N):
        y = []
        for r in D.r[n]:
            y += returns(r, gamma)
        out[n, :len(y)] = y
    return out
