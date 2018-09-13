import numpy as np

from lagom.envs.spaces import Box
from .wrapper import ObservationWrapper


class FlattenDictWrapper(ObservationWrapper):
    r"""Observation wrapper that flattens a dictionary observation into an array based on selected keys. 
    
    In addition, it adapts the observation space from :class:`Dict` to :class:`Box`. 
    
    Example::
    
        >>> env = gym.make('FetchPush-v1')
        >>> env = GymWrapper(env)
        >>> env.observation_space
        Dict('achieved_goal: Box(3,)', 'desired_goal: Box(3,)', 'observation: Box(25,)')

        >>> env.reset()
        {'observation': array([ 1.34193113e+00,  7.48903354e-01,  4.13631365e-01,  1.38820802e+00,
                 6.54782758e-01,  4.24702091e-01,  4.62768933e-02, -9.41205963e-02,
                 1.10707256e-02, -2.06467383e-06,  1.87039390e-03, -8.82449686e-08,
                 1.35761490e-07,  3.97194063e-15, -2.96711930e-07, -5.50441164e-05,
                 4.69134018e-05,  5.03907166e-08, -7.75241793e-08,  2.51278755e-18,
                 2.94776838e-07,  5.50428586e-05, -8.56360696e-08,  5.26545712e-07,
                 6.00978493e-05]),
         'achieved_goal': array([1.38820802, 0.65478276, 0.42470209]),
         'desired_goal': array([1.43909209, 0.68916859, 0.42469975])}
         
        >>> env = FlattenDictWrapper(env, ['observation', 'desired_goal'])
        >>> env.observation_space
        Box(28,)
        
        >>> env.reset()
        array([ 1.34193110e+00,  7.48903334e-01,  4.13631380e-01,  1.36456788e+00,
                6.42616391e-01,  4.24702078e-01,  2.26368085e-02, -1.06286958e-01,
                1.10707255e-02, -2.06467394e-06,  1.87039387e-03, -8.82449669e-08,
                1.35761496e-07,  3.97194078e-15, -2.96711931e-07, -5.50441182e-05,
                4.69134029e-05,  5.03907174e-08, -7.75241773e-08,  2.51278755e-18,
                2.94776839e-07,  5.50428595e-05, -8.56360671e-08,  5.26545705e-07,
                6.00978492e-05,  1.45042038e+00,  6.73659265e-01,  4.24699754e-01],
              dtype=float32)

    """
    def __init__(self, env, keys):
        r"""Initialize the wrapper. 
        
        Args:
            env (Env): environment object
            keys (list): a list of selected keys to flatten the observation. 
        """
        super().__init__(env)
        
        self.keys = keys
        
        # Sanity check that all subspaces should be Box
        spaces = self.env.observation_space.spaces
        assert all([isinstance(space, Box) for space in spaces.values()])
        
        # Calculate dimensionality
        shape = (int(np.sum([spaces[key].flat_dim for key in self.keys])), )
        
        # Create new observation space
        self._observation_space = Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32)
        
    def process_observation(self, observation):
        assert isinstance(observation, dict), f'expected dict dtype, got {type(observation)}'
        return np.concatenate([observation[key].ravel() for key in self.keys]).astype(np.float32)
    
    @property
    def observation_space(self):
        return self._observation_space
