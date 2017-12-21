import numpy as np

def calculate_return(rewards, gamma):
    returns = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma*R
        returns.insert(0, R)
        
    return returns

def normalize_vec(vec):
    # Ensure vector is of type numpy array
    vec = np.array(vec)
    
    # Exclude all later zeros, avoid too small values for big sparse vector
    #idx_zero = np.where(vec == 0)[0][0]
    #vec_nonzero = vec[:idx_zero]
    
    #vec[:idx_zero] = (vec_nonzero - vec_nonzero.mean())/(vec_nonzero.std() + np.finfo(np.float32).eps)
    
    return (vec - vec.mean())/(vec.std() + np.finfo(np.float32).eps)
