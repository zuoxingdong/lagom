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
    
    return (vec - vec.mean())/(vec.std() + np.finfo(np.float32).eps)
