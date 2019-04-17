# lagom
<!--- <img src='doc/img/infrastructure.png' width='300'> --->

[![Build Status](https://travis-ci.org/zuoxingdong/lagom.svg?branch=master)](https://travis-ci.org/zuoxingdong/lagom)
[![CircleCI](https://circleci.com/gh/zuoxingdong/lagom.svg?style=svg)](https://circleci.com/gh/zuoxingdong/lagom)
[![Documentation Status](https://readthedocs.org/projects/lagom/badge/?version=latest)](https://lagom.readthedocs.io/en/latest/?badge=latest)

**lagom is a light PyTorch infrastructure to quickly prototype reinforcement learning algorithms.** [Lagom](https://sv.wikipedia.org/wiki/Lagom) is a 'magic' word in Swedish, *"inte för mycket och inte för lite, enkelhet är bäst"*, meaning *"not too much and not too little, simplicity is often the best"*. lagom is the philosophy on which this library was designed. 

**Contents of this document**

- [Basics](#basics)
- [Installation](#installation)
    - [Install dependencies](#install-dependencies)
    - [Install lagom](#install-lagom)
- [Getting Started](#getting-started)
- [Documentation](#documentation)
- [Examples](#examples)
- [Test](#test)
- [What's new](#What's-new)
- [Reference](#reference)

# Basics

`lagom` balances between the flexibility and the usability when developing reinforcement learning (RL) algorithms. The library is built on top of [PyTorch](https://pytorch.org/) and provides modular tools to quickly prototype RL algorithms. However, it does not go overboard, because too low level is often time consuming and prone to potential bugs, while too high level degrades the flexibility which makes it difficult to try out some crazy ideas fast. 

We are continuously making `lagom` more 'self-contained' to set up and run experiments quickly. It internally supports base classes for multiprocessing ([master-worker framework](https://en.wikipedia.org/wiki/Master/slave_(technology))) for parallelization (e.g. experiments and evolution strategies). It also supports hyperparameter search by defining configurations either as grid search or random search. 

A common pipeline to use `lagom` can be done as following:
1. Define your [RL agent](lagom/agent.py)
2. Define your [environment](lagom/envs)
3. Define your [engine](lagom/engine.py) for training and evaluating the agent in the environment.
4. Define your [Configurations](lagom/experiment/config.py) for hyperparameter search
5. Define `run(config, seed, device)` for your experiment pipeline
6. Call `run_experiment(run, config, seeds, num_worker)` to parallelize your experiments

A graphical illustration is coming soon. 

# Installation

## Install dependencies
Run the following command to install [all required dependencies](./requirements.txt):

```bash
pip install -r requirements.txt
```

Note that it is highly recommanded to use an Miniconda environment:

```bash
conda create -n lagom python=3.7
```

We also provide some bash scripts in [scripts/](scripts/) directory to automatically set up the conda environment and dependencies.

## Install lagom

Run the following commands to install lagom from source:

```bash
git clone https://github.com/zuoxingdong/lagom.git
cd lagom
pip install -e .
```

Installing from source allows to flexibly modify and adapt the code as you pleased, this is very convenient for research purpose which often needs fast prototyping. 

# Getting Started

Detailed tutorials is coming soon. For now, it is recommended to have a look in [examples/](examples/) or the source code. 

# Documentation

The documentation hosted by ReadTheDocs is available online at [http://lagom.readthedocs.io](http://lagom.readthedocs.io)

# Examples

We are continuously providing [examples/](examples/) to use lagom. 

# Test

We are using [pytest](https://docs.pytest.org) for tests. Feel free to run via

```bash
pytest test -v
```

# What's new

- 2019-03-04 (v0.0.3)
    - Much easier and cleaner APIs

- 2018-11-04 (v0.0.2)
    - More high-level API designs
    - More unit tests

- 2018-09-20 (v0.0.1)
    - Initial release

# Reference

This repo is inspired by [OpenAI Gym](https://github.com/openai/gym/), [OpenAI baselines](https://github.com/openai/baselines), [OpenAI Spinning Up](https://github.com/openai/spinningup)

Please use this bibtex if you want to cite this repository in your publications:

    @misc{lagom,
          author = {Zuo, Xingdong},
          title = {lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms},
          year = {2018},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/lagom}},
        }
