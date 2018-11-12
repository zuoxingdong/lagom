lagom.history: History
===================================

.. automodule:: lagom.history
.. currentmodule:: lagom.history

.. autoclass:: BaseHistory
    :members:
    
.. autoclass:: Transition
    :members:
    
.. autoclass:: Trajectory
    :members:

.. autoclass:: Segment
    :members:
    
.. autoclass:: History
    :members:
    
.. autoclass:: BatchEpisode
    :members:
    
.. autoclass:: BatchSegment
    :members:

Metrics
----------------

.. currentmodule:: lagom.history.metrics

.. autofunction:: terminal_state_from_trajectory

.. autofunction:: terminal_state_from_segment

.. autofunction:: final_state_from_trajectory

.. autofunction:: final_state_from_segment

.. autofunction:: bootstrapped_returns_from_trajectory

.. autofunction:: bootstrapped_returns_from_segment

.. autofunction:: td0_target

.. autofunction:: td0_target_from_trajectory

.. autofunction:: td0_target_from_segment

.. autofunction:: td0_error

.. autofunction:: td0_error_from_trajectory

.. autofunction:: td0_error_from_segment

.. autofunction:: gae

.. autofunction:: gae_from_trajectory

.. autofunction:: gae_from_segment
