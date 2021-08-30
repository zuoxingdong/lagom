lagom.nn: Neural Network Modules
===================================

.. automodule:: lagom.nn
.. currentmodule:: lagom.nn

.. autoclass:: Module
    :members:

.. autofunction:: bound_logvar

.. autofunction:: ortho_init
    
.. autofunction:: make_fc

.. autofunction:: make_cnn

.. autofunction:: make_transposed_cnn

.. autoclass:: MDNHead
    :members:

Recurrent Neural Networks
-----------------------------
.. autoclass:: LayerNormLSTMCell
    :members:
    
.. autoclass:: LSTMLayer
    :members:
    
.. autoclass:: StackedLSTM
    :members:
    
.. autofunction:: make_lnlstm

RL components
-----------------------------    
.. autoclass:: CategoricalHead
    :members:
    
.. autoclass:: DiagGaussianHead
    :members:
