import numpy as np

from lagom.envs.spaces import Box

from .wrapper import ObservationWrapper


class FrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner. 
    
    For example, if the number os stacks is 4, then returned observation constains
    the most recent 4 observations. For environment 'Pendulum-v0', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [3, 4]. 
    
    .. note::
    
        Each call :meth:`step`, the new observation is augmented to the stacked buffer
        and the oldest one is removed. 
    
    .. note::
    
        The observation space must be :class:`Box` type. If one uses :class:`Dict`
        as observation space, it should apply :class:`FlattenDictWrapper` at first. 
    
    Example::
    
        >>> env = make_gym_env(env_id='CartPole-v1', seed=1)
        >>> env = FrameStack(env, num_stack=4)
        >>> env
        <FrameStack, <GymWrapper, <TimeLimit<CartPoleEnv<CartPole-v1>>>>>
        
        >>> env.observation_space
        Box(4, 4)
        
        >>> env.reset()
        array([[ 0.03073904,  0.        ,  0.        ,  0.        ],
               [ 0.00145001,  0.        ,  0.        ,  0.        ],
               [-0.03088818,  0.        ,  0.        ,  0.        ],
               [-0.03131252,  0.        ,  0.        ,  0.        ]],
              dtype=float32)
              
        >>> env.step(env.action_space.sample())
        (array([[ 0.03076804,  0.03073904,  0.        ,  0.        ],
                [-0.19321568,  0.00145001,  0.        ,  0.        ],
                [-0.03151444, -0.03088818,  0.        ,  0.        ],
                [ 0.25146705, -0.03131252,  0.        ,  0.        ]],
               dtype=float32), 1.0, False, {})
    
    """
    def __init__(self, env, num_stack):
        r"""Initialize the wrapper. 
        
        Args:
            env (Env): environment object
            num_stack (int): number of stacks
        """
        super().__init__(env)
        
        self.num_stack = num_stack
        
        assert isinstance(self.env.observation_space, Box), 'must be Box type'
        
        # Create a new observation space
        low = np.repeat(self.env.observation_space.low[..., np.newaxis], self.num_stack, axis=-1)
        high = np.repeat(self.env.observation_space.high[..., np.newaxis], self.num_stack, axis=-1)
        dtype = self.env.observation_space.dtype
        self._observation_space = Box(low=low, high=high, dtype=dtype)
        
        # Initialize the buffer for stacked observation
        self.stack_buffer = np.zeros(self._observation_space.shape, dtype=dtype)
        
    def reset(self):
        # Clean up all stacked observation
        self.stack_buffer.fill(0.0)
        
        # Call reset in original environment
        return super().reset()

    def process_observation(self, observation):
        # Shift the oldest observation to the front
        self.stack_buffer  = np.roll(self.stack_buffer, shift=1, axis=-1)
        # Replace the front as new observation
        self.stack_buffer[..., 0] = observation
        
        return self.stack_buffer
        
    @property
    def observation_space(self):
        return self._observation_space
