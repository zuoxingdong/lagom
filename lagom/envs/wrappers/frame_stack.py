import numpy as np

from gym.spaces import Box
from gym import ObservationWrapper


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
    
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first. 
    
    Example::
    
        >>> from lagom.envs import make_gym_env
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
    
    Args:
            env (Env): environment object
            num_stack (int): number of stacks
    
    """
    def __init__(self, env, num_stack):
        super().__init__(env)
        assert isinstance(self.observation_space, Box), 'must be Box type'
        self.num_stack = num_stack
        
        # Create a new observation space
        low = np.repeat(self.observation_space.low[..., np.newaxis], self.num_stack, axis=-1)
        high = np.repeat(self.observation_space.high[..., np.newaxis], self.num_stack, axis=-1)
        dtype = self.observation_space.dtype
        self.observation_space = Box(low=low, high=high, dtype=dtype)
        
        # Initialize the buffer for stacked observation
        self.stack_buffer = np.zeros(self.observation_space.shape, dtype=dtype)
        
    def reset(self, **kwargs):
        self.stack_buffer.fill(0.0)
        return super().reset(**kwargs)

    def observation(self, observation):
        # Shift the oldest observation to the front
        self.stack_buffer = np.roll(self.stack_buffer, shift=1, axis=-1)
        # Replace the front as new observation
        self.stack_buffer[..., 0] = observation
        
        return self.stack_buffer
