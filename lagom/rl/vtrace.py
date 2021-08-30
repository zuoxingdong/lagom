import numpy as np

from lagom.utils import numpify

from .td import td0_error


def vtrace(behavior_logprobs, target_logprobs, gamma, Rs, Vs, last_V, reach_terminal, clip_rho=1.0, clip_pg_rho=1.0):
    behavior_logprobs = numpify(behavior_logprobs, np.float32)
    target_logprobs = numpify(target_logprobs, np.float32)
    Rs = numpify(Rs, np.float32)
    Vs = numpify(Vs, np.float32)
    last_V = numpify(last_V, np.float32)
    assert all([item.ndim == 1 for item in [behavior_logprobs, target_logprobs, Rs, Vs]])
    assert np.isscalar(gamma)

    rhos = np.exp(target_logprobs - behavior_logprobs)
    clipped_rhos = np.minimum(clip_rho, rhos)
    cs = np.minimum(1.0, rhos)
    deltas = clipped_rhos*td0_error(gamma, Rs, Vs, last_V, reach_terminal)

    vs_minus_V = []
    total = 0.0
    for delta_t, c_t in zip(deltas[::-1], cs[::-1]):
        total = delta_t + gamma*c_t*total
        vs_minus_V.append(total)
    vs_minus_V = np.asarray(vs_minus_V)[::-1]

    vs = vs_minus_V + Vs
    vs_next = np.append(vs[1:], (1. - reach_terminal)*last_V)
    clipped_pg_rhos = np.minimum(clip_pg_rho, rhos)
    As = clipped_pg_rhos*(Rs + gamma*vs_next - Vs)
    return vs, As
