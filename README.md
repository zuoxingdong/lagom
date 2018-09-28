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

`lagom` balances between the flexibility and the usability when developing reinforcement learning (RL) algorithms. The library is built on top of [PyTorch](https://pytorch.org/) and provides modular tools to quickly prototype RL algorithms. However, it does not go overboard, because too low level is often time consuming and prone to potential bugs, while too high level degrades the flexibility which makes it difficult to try out some crazy ideas fast. 

We are continuously making `lagom` more 'self-contained' to set up and run experiments quickly. It internally supports base classes for multiprocessing ([master-worker framework](https://en.wikipedia.org/wiki/Master/slave_(technology))) for parallelization (e.g. experiments and evolution strategies). It also supports hyperparameter search by defining configurations either as grid search or random search. 

A common pipeline to use `lagom` can be done as following:
1. Define [environment](lagom/envs) and [agent](lagom/agents) (mainly for RL)
2. Use [runner](lagom/runner) to collect data (trajectories or segments) for agent
3. Define [engine](lagom/engine) for training and evaluating the agent
4. Define [algorithm](lagom/base_algo.py)
5. Define [experiment](lagom/experiment) and [configurations](lagom/experiment/configurator.py)

A graphical illustration is coming soon. 

# Installation

## Install dependencies
Run the following command to install [all required dependencies](./requirements.txt):

```bash
pip install -r requirements.txt
```

Note that it is highly recommanded to use an Anaconda environment:

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

# Roadmap

## Core
    - Tutorials
## More standard RL baselines
    - IMPALA
    - TRPO/PPO
    - NAF/GAE
    - ACKTR
    - DDPG
    - ACER
    - Q-Prop
    - DQN: DQN/Rainbow
    - ES: NES
## More standard networks
    - Monte Carlo Dropout/Concrete Dropout
## Misc
    - Support pip install
    - Technical report

# Reference

This repo is inspired by [OpenAI Gym](https://github.com/openai/gym/), [garage](https://github.com/rlworkgroup/garage), [OpenAI baselines](https://github.com/openai/baselines), [Intel Coach](https://github.com/NervanaSystems/coach) and [Dopamine](https://github.com/google/dopamine)

Please use this bibtex if you want to cite this repository in your publications:

    @misc{lagom,
          author = {Zuo, Xingdong},
          title = {lagom: A light PyTorch infrastructure to quickly prototype reinforcement learning algorithms},
          year = {2018},
          publisher = {GitHub},
          journal = {GitHub repository},
          howpublished = {\url{https://github.com/zuoxingdong/lagom}},
        }
