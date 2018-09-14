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
- [Roadmap](#roadmap)
- [Reference](#reference)

# Basics

`lagom` balances between the flexibility and the userability when developing reinforcement learning (RL) algorithms. The library is built on top of [PyTorch](https://pytorch.org/) and provides modular tools to quickly prototype RL algorithms. However, we do not go overboard, because going too low level is rather time consuming and prone to potential bugs, while going too high level degrades the flexibility which makes it difficult to try out some crazy ideas. 

We are continuously making `lagom` more 'self-contained' to run experiments quickly. Now, it internally supports base classes for multiprocessing ([master-worker framework](https://en.wikipedia.org/wiki/Master/slave_(technology))) to parallelize (e.g. experiments and evolution strategies). It also supports hyperparameter search by defining configurations either as grid search or random search. 

One of the main pipelines to use `lagom` can be done as following:
1. Define environment and RL agent
2. User runner to collect data for agent
3. Define algorithm to train agent
4. Define experiment and configurations. 

A graphical illustration is coming soon. 

# Installation

## Install dependencies
Run the following command to install [all the dependencies](./requirements.txt):

```bash
pip install -r requirements.txt
```

Note that it is highly recommanded to use an Anaconda environment:

```bash
conda create -n lagom python=3.7
```

We also provide some bash scripts in [scripts/](scripts/) directory to automatically set up the conda environment and dependencies.

## Install lagom

Run the following command to install from source:

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

We shall continuously provide [examples/](examples/) to use lagom. 

# Test

We are using [pytest](https://docs.pytest.org) for tests. Feel free to run via

```bash
pytest test -v
```

# Roadmap

## Core
    - Readthedocs Documentation
    - Tutorials
## More standard RL baselines
    - TRPO/PPO
    - ACKTR
    - DDPG
    - ACER
    - Q-Prop
    - DQN: Rainbow
    - ES: PEPG/NES
## More standard networks
    - Monte Carlo Dropout/Concrete Dropout
## Misc
    - VecEnv: similar to that of OpenAI baseline
    - Support pip install
    - Technical report

# Reference

This repo is inspired by [OpenAI Gym](https://github.com/openai/gym/), [OpenAI rllab](https://github.com/rll/rllab), [OpenAI baselines](https://github.com/openai/baselines), [RLPyTorch](https://github.com/pytorch/ELF/tree/master/src_py/rlpytorch), [TensorForce](https://github.com/reinforceio/tensorforce) [Intel Coach](https://github.com/NervanaSystems/coach) and [Dopamine](https://github.com/google/dopamine)

Please use this bibtex if you want to cite this repository in your publications:

    @misc{lagom,
          author = {Zuo, Xingdong},
          title = {lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms},
          year = {2018},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/lagom}},
        }
