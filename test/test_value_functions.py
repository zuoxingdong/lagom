import torch

from lagom.networks import BaseNetwork

from lagom.value_functions import StateValueHead


def test_state_value_head():
    value_head = StateValueHead(None, None, 30)
    assert isinstance(value_head, BaseNetwork)
    assert value_head.feature_dim == 30
    x = value_head(torch.randn(3, 30))
    assert list(x.shape) == [3, 1]
