lagom.envs
===================================

.. automodule:: lagom.envs
.. currentmodule:: lagom.envs

Spaces
----------------

.. autoclass:: lagom.envs.spaces.Space
    :members:
    
.. autoclass:: lagom.envs.spaces.Discrete
    :members:
    
.. autoclass:: lagom.envs.spaces.Box
    :members:
    
.. autoclass:: lagom.envs.spaces.Product
    :members:
    
.. autoclass:: lagom.envs.spaces.Dict
    :members:

Environment
---------------

.. autoclass:: Env
    :members:
    
.. autoclass:: EnvSpec
    :members:
    
.. autoclass:: GoalEnv
    :members:
    
.. autoclass:: GymEnv
    :members:
    
.. autofunction:: make_gym_env

.. autofunction:: make_envs

Wrappers
--------------

.. autoclass:: lagom.envs.wrappers.Wrapper
    :members:
    
.. autoclass:: lagom.envs.wrappers.ObservationWrapper
    :members:
    
.. autoclass:: lagom.envs.wrappers.ActionWrapper
    :members:
    
.. autoclass:: lagom.envs.wrappers.RewardWrapper
    :members:
    
.. autoclass:: lagom.envs.wrappers.StackObservation
    :members:
    
.. autoclass:: lagom.envs.wrappers.SparseReward
    :members:
    
.. autoclass:: lagom.envs.wrappers.PartialFlattenDict
    :members:
    
.. autoclass:: lagom.envs.wrappers.FlattenObservation
    :members:

Vectorized Environment
--------------------------

.. autoclass:: lagom.envs.vec_env.VecEnv
    :members:
    
.. autoclass:: lagom.envs.vec_env.SerialVecEnv
    :members:
    
.. autoclass:: lagom.envs.vec_env.ParallelVecEnv
    :members:
    
.. autoclass:: lagom.envs.vec_env.VecEnvWrapper
    :members:
    
.. autoclass:: lagom.envs.vec_env.StandardizeVecEnv
    :members:
    
.. autoclass:: lagom.envs.vec_env.CloudpickleWrapper
    :members: