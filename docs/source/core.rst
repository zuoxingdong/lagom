lagom.core
===================================

.. automodule:: lagom.core
.. currentmodule:: lagom.core

Evolution Strategies (ES)
-------------------------------

.. currentmodule:: lagom.core.es

ES infrastructure
~~~~~~~~~~~~~~~~~

.. autoclass:: BaseES
    :members:
    
.. autoclass:: BaseESMaster
    :members:
    
.. autoclass:: BaseESWorker
    :members:
    
.. autoclass:: ESOptimizer
    :members:
    
ES algorithms
~~~~~~~~~~~~~

.. autoclass:: CMAES
    :members:
    
.. autoclass:: OpenAIES
    :members:

Test functions
~~~~~~~~~~~~~~

.. autoclass:: lagom.core.es.test_functions.BaseTestFunction
    :members:
    
.. autoclass:: lagom.core.es.test_functions.Rastrigin
    :members:
    
.. autoclass:: lagom.core.es.test_functions.Sphere
    :members:
    
.. autoclass:: lagom.core.es.test_functions.HolderTable
    :members:
    
.. autoclass:: lagom.core.es.test_functions.StyblinskiTang
    :members:
    
Multiprocessing
-----------------------

.. currentmodule:: lagom.core.multiprocessing

.. autoclass:: BaseMaster
    :members:
    
.. autoclass:: BaseWorker
    :members:
    
.. autoclass:: BaseIterativeMaster
    :members:
    
Networks
----------------------

.. currentmodule:: lagom.core.networks

.. autofunction:: ortho_init

.. autoclass:: BaseNetwork
    :members:
    
.. autofunction:: make_fc

.. autofunction:: make_cnn

.. autofunction:: make_transposed_cnn

.. autofunction:: make_rnncell
        
.. autoclass:: BaseMDN
    :members:
    
.. autoclass:: BaseVAE
    :members:
    
.. autoclass:: BaseRNN
    :members:

Plotter
----------------------

.. currentmodule:: lagom.core.plotter

.. autoclass:: BasePlot
    :members:
    
.. autoclass:: CurvePlot
    :members:
    
.. autoclass:: GridImage
    :members:
    
.. autoclass:: ImageViewer
    :members:
    
Policies
----------------------

.. currentmodule:: lagom.core.policies

.. autoclass:: BasePolicy
    :members:
    
.. autoclass:: RandomPolicy
    :members:
    
.. autoclass:: CategoricalPolicy
    :members:
    
.. autoclass:: GaussianPolicy
    :members:
    
Transformations
----------------------

.. currentmodule:: lagom.core.transform

.. autoclass:: BaseTransform
    :members:
    
.. autoclass:: Centralize
    :members:

.. autoclass:: Clip
    :members: 
    
.. autoclass:: ExpFactorCumSum
    :members:    
    
.. autoclass:: InterpCurve
    :members:

.. autoclass:: Normalize
    :members:
    
.. autoclass:: RankTransform
    :members:

.. autoclass:: RunningMeanStd
    :members:
    
.. autoclass:: SmoothFilter
    :members:

.. autoclass:: Standardize
    :members:
    
.. autoclass:: LinearSchedule
    :members:

Utils
----------------------

.. currentmodule:: lagom.core.utils

