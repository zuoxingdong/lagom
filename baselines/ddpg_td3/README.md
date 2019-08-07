# Deep Deterministic Policy Gradients (DDPG) / Twin Delayed DDPG (TD3)

This is an implementation of both [DDPG](https://arxiv.org/abs/1509.02971) and [TD3](https://arxiv.org/abs/1802.09477).

One can choose to use it in [experiment.py](./experiment.py): `'agent.use_td3': True`

# Usage

Run the following command to start parallelized training:

```bash
python experiment.py
```

One could modify [experiment.py](./experiment.py) to quickly set up different configurations. 

# Results
<img src='logs/default/result.png' width='100%'>
