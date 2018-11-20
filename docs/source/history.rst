lagom.history: History
===================================

.. automodule:: lagom.history
.. currentmodule:: lagom.history

.. autoclass:: BaseHistory
    :members:
    
.. autoclass:: BatchEpisode
    :members:
    
.. autoclass:: BatchSegment
    :members:

Metrics
----------------

.. currentmodule:: lagom.history.metrics

.. autofunction:: terminal_state_from_episode

.. autofunction:: terminal_state_from_segment

.. autofunction:: final_state_from_episode

.. autofunction:: final_state_from_segment

.. autofunction:: returns_from_episode

.. autofunction:: returns_from_segment

.. autofunction:: bootstrapped_returns_from_episode

.. autofunction:: bootstrapped_returns_from_segment

.. autofunction:: td0_target_from_episode

.. autofunction:: td0_target_from_segment

.. autofunction:: td0_error_from_episode

.. autofunction:: td0_error_from_segment

.. autofunction:: gae_from_episode

.. autofunction:: gae_from_segment
