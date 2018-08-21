.. lagom documentation master file, created by
   sphinx-quickstart on Fri Jul 27 16:10:10 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

lagom
=====
    | *inte för mycket och inte för lite, enkelhet är bäst*
    | *not too much and not too little, simplicity is often the best*

.. raw:: html

  <embed>
    <a href="https://github.com/zuoxingdong/lagom"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png"></a>
  </embed>
  
*lagom is a light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.*

lagom balances between the flexibility and the userability when developing reinforcement learning (RL) algorithms. The library is built on top of `PyTorch`_ and provides modular tools to quickly prototype RL algorithms. However, we do not go overboard, because going too low level is rather time consuming and prone to potential bugs, while going too high level degrades the flexibility which makes it difficult to try out some crazy ideas. 

.. _`PyTorch`: https://pytorch.org/

We are continuously making lagom more 'self-contained' to run experiments quickly. Now, it internally supports base classes for multiprocessing (master-worker framework) to parallelize (e.g. experiments and evolution strategies). It also supports hyperparameter search by defining configurations either as grid search or random search. 

One of the main pipelines to use lagom can be done as following:

    1. Define environment and RL agent
    2. User runner to collect data for agent
    3. Define algorithm to train agent
    4. Define experiment and configurations. 

A graphical illustration is coming soon. 


.. toctree::
    :maxdepth: 1
    :caption: Installation
    
    setup
    install
    
.. toctree::
    :maxdepth: 1
    :caption: Tutorials
    
.. toctree::
    :maxdepth: 1
    :caption: Examples
    
    examples_vae
    examples_mdn
    examples_es
    examples_pg
    
.. toctree::
    :maxdepth: 1
    :caption: lagom API
    
    lagom
    lagom.agents <agents>
    lagom.contrib <contrib>
    lagom.core <core>
    lagom.engine <engine>
    lagom.envs <envs>
    lagom.experiment <experiment>
    lagom.runner <runner>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
