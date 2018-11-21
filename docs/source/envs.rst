lagom.envs
===================================

.. automodule:: lagom.envs
.. currentmodule:: lagom.envs

Spaces
----------------

.. currentmodule:: lagom.envs.spaces

.. autoclass:: Space
    :members:
    
.. autoclass:: Discrete
    :members:
    
.. autoclass:: Box
    :members:
    
.. autoclass:: Tuple
    :members:
    
.. autoclass:: Dict
    :members:
    
.. autofunction:: convert_gym_space

Environment
---------------

.. currentmodule:: lagom.envs

.. autoclass:: Env
    :members:
    
.. autoclass:: EnvSpec
    :members:
    
.. autoclass:: GoalEnv
    :members:
    
.. autofunction:: make_gym_env

.. autofunction:: make_envs

.. autofunction:: make_vec_env

Wrappers
--------------

.. currentmodule:: lagom.envs.wrappers

.. autoclass:: Wrapper
    :members:
    
.. autoclass:: ObservationWrapper
    :members:
    
.. autoclass:: ActionWrapper
    :members:
    
.. autoclass:: RewardWrapper
    :members:
    
.. autoclass:: GymWrapper
    :members:

.. autoclass:: FlattenObservation
    :members:

.. autoclass:: FlattenDictWrapper
    :members:
    
.. autoclass:: FrameStack
    :members:
    
.. autoclass:: RewardScale
    :members:

Vectorized Environment
--------------------------

.. currentmodule:: lagom.envs.vec_env

.. autoclass:: VecEnv
    :members:
    
.. autoclass:: SerialVecEnv
    :members:
    
.. autoclass:: ParallelVecEnv
    :members:
    
.. autoclass:: VecEnvWrapper
    :members:
    
.. autoclass:: VecStandardize
    :members:
    
.. autoclass:: VecClipAction
    :members:
    
.. autoclass:: VecMonitor
    :members:
    
.. autoclass:: CloudpickleWrapper
    :members:
    
.. autofunction:: get_wrapper
