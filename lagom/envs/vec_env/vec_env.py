import numpy as np

from abc import ABC
from abc import abstractmethod

from lagom.core.plotter import GridImage
from lagom.core.plotter import ImageViewer


class VecEnv(ABC):
    r"""Base class for all asynchronous, vectorized environments. 
    
    A vectorized environment handles batched data with its sub-environments. Each observation
    returned from vectorized environment is a batch of observations for each sub-environment. 
    And :meth:`step` is expected to receive a batch of actions for each sub-environment. 
    
    .. note::
    
        All sub-environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported. 
        
    See docstring in :class:`Env` for more details about common APIs. 
    
    The subclass should implement at least the following:

    - :meth:`step_async`
    - :meth:`step_wait`
    - :meth:`reset`
    - :meth:`get_images`
    - :meth:`close_extras`
    - :meth:`T`
    - :meth:`max_episode_reward`
    - :meth:`reward_range`
    
    """
    def __init__(self, list_make_env, observation_space, action_space):
        r"""Initialize the vectorized environment. 
        
        .. note::
        
            The random seeds for all environments should be handled within each make_env function. 
            And it should not handled here in general, because of APIs for parallelized environments.
        
        Args:
            list_make_env (list): a list of functions each returns an instantiated enviroment. 
            observation_space (Space): observation space of the environment
            action_space (Space): action space of the environment
        """
        self.list_make_env = list_make_env
        self.num_env = len(self.list_make_env)
        
        self._observation_space = observation_space
        self._action_space = action_space
        
        # Some settings
        self.closed = False
        self.viewer = None  # rendering
        
    @abstractmethod
    def step_async(self, actions):
        r"""Ask all the environments to take a step with a list of actions, each for one environment. 
        
        .. note::
        
            Call :meth:`step_wait` to get the step execution results. 
        
        .. note::
        
            Do not call this function when it is already pending. 
        
        Args:
            actions (list): a list of actions, each for one environment. 
        """
        pass
        
    @abstractmethod
    def step_wait(self):
        r"""Wait for the jobs in :meth:`step_async` to finish and return all results. 
        
        Returns
        -------
        observations : list
            a list of observations, each returned from one environment after executing the given action. 
        rewards : list
            a list of scalar rewards, each returned from one environment. 
        dones : list
            a list of booleans indicating whether the episode terminates, each returned from one environment. 
        infos : list
            a list of dictionaries of additional informations, each returned from one environment. 
        """
        pass
        
    def step(self, actions):
        r"""Take an action in each environment for one time step through the environments' dynamics. 
        
        It firstly calls :meth:`step_async` to distribute actions to all environments and then
        calls :meth:`step_wait` to get all the results. 
        
        See docstrings in :meth:`step_async` and :meth:`step_wait` for more details. 
        """
        assert len(actions) == self.num_env, f'expected length {self.num_env}, got {len(actions)}'
        # Send actions to all environments asynchronously
        self.step_async(actions)
        
        # wait to receive all results and return them
        return self.step_wait()
    
    @abstractmethod
    def reset(self):
        r"""Reset all the environments and return a list of initial observations from each environment. 
        
        .. warning::
        
            If :meth:`step_async` is still working, then it will be aborted. 
        
        Returns
        -------
        observations : list
            a list of initial observations from all environments. 
        """
        pass
    
    def render(self, mode='human'):
        r"""Render all the environments. 
        
        It firstly retrieve RGB images from all environments and use :class:`GridImage`
        to make a grid of them as a single image. Then it either returns the image array
        or display the image to the screen by using :class:`ImageViewer`. 
        
        See docstring in :class:`Env` for more detais about rendering. 
        """
        # Get images from all environments with shape [N, H, W, C]
        imgs = self.get_images()
        imgs = np.stack(imgs)
        # Make a grid of images
        grid = GridImage(ncol=5, padding=5, pad_value=0)
        imgs = imgs.transpose(0, 3, 1, 2)  # to shape [N, C, H, W]
        grid.add(imgs)
        gridimg = np.asarray(grid())
        gridimg = gridimg.transpose(0, 2, 3, 1)  # back to shape [N, H, W, C]
        
        # render the grid of image
        if mode == 'human':
            self.get_viewer()(gridimg)
        elif mode == 'rgb_array':
            return gridimg
        else:
            raise ValueError(f'expected human or rgb_array, got {mode}')
        
    @abstractmethod
    def get_images(self):
        r"""Returns a batched RGB array with shape [N, H, W, C] from all environments. 
        
        Returns
        -------
        imgs : ndarray
            a batched RGB array with shape [N, H, W, C]
        """
        pass
    
    def get_viewer(self):
        r"""Returns an instantiated :class:`ImageViewer`. 
        
        Returns
        -------
        viewer : ImageViewer
            an image viewer
        """
        if self.viewer is None:  # create viewer is not existed
            self.viewer = ImageViewer(max_width=500)  # set a max width here
            
        return self.viewer
    
    @abstractmethod
    def close_extras(self):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        pass
    
    def close(self):
        r"""Close all environments. 
        
        It closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``. 
        
        .. warning::
        
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is useful for parallelized environments. 
        
        .. note::
        
            This will be automatically called when garbage collected or program exited. 
            
        """
        if self.closed:  # already closed
            return
        
        # Close viewer if existed
        if self.viewer is not None:
            self.viewer.close()
        
        # Close extra
        self.close_extras()

        # Set closed flag
        self.closed = True
    
    @property
    def unwrapped(self):
        r"""Unwrap this vectorized environment. 
        
        Useful for sequential wrappers applied, it can access information from the original 
        vectorized environment. 
        """
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self
        
    @property
    def observation_space(self):
        r"""Returns a :class:`Space` object to define the observation space. """
        return self._observation_space
    
    @property
    def action_space(self):
        r"""Returns a :class:`Space` object to define the action space. """
        return self._action_space
    
    @property
    @abstractmethod
    def T(self):
        r"""Maximum horizon of the environment, if available. """
        pass
    
    @property
    @abstractmethod
    def max_episode_reward(self):
        r"""Maximum episodic rewards of the environment, if available. """
        pass
    
    @property
    @abstractmethod
    def reward_range(self):
        r"""Returns a tuple of min and max possible rewards. 
        
        .. note::
        
            By default, it could be infinite range i.e. ``(-float('inf'), float('inf'))``. 
            One can also set a narrower range. 
        
        """
        pass
    
    def __repr__(self):
        return f'<{self.__class__.__name__}: {self.list_make_env[0].keywords["env_id"]}, n: {self.num_env}>'

    
class VecEnvWrapper(VecEnv):
    r"""Wraps the vectorized environment to allow a modular transformation. 
    
    This class is the base class for all wrappers for vectorized environments. The subclass
    could override some methods to change the behavior of the original vectorized environment
    without touching the original code. 
    
    .. note::
    
        Don't forget to call ``super().__init__(venv)`` if the subclass overrides :meth:`__init__`.
        
    The subclass should implement at least the following:

    - :meth:`step_wait`
    - :meth:`reset`
    - :meth:`close_extras`
    
    """
    def __init__(self, venv):
        assert isinstance(venv, VecEnv), f'expected VecEnv type, got {type(venv)}'
        
        self.venv = venv
        
        # Call VecEnv constructor to make some settings as direct attributes in this wrapper class
        super().__init__(list_make_env=venv.list_make_env, 
                         observation_space=venv.observation_space, 
                         action_space=venv.action_space)
        
    def step_async(self, actions):
        self.venv.step_async(actions)
        
    @abstractmethod
    def step_wait(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    def get_images(self):
        self.venv.get_images()
    
    @abstractmethod
    def close_extras(self):
        pass
    
    @property
    def T(self):
        return self.venv.T
    
    @property
    def max_episode_reward(self):
        return self.venv.max_episode_reward

    @property
    def reward_range(self):
        return self.venv.reward_range
