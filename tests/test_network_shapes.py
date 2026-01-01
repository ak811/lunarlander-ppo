import torch

from lunarlander_ppo.agent.networks import ActorCritic


def test_actorcritic_shapes():
    state_dim = 8
    action_dim = 4
    net = ActorCritic(state_dim, action_dim, hidden_dim=64)

    x = torch.randn(5, state_dim)
    logits, value = net(x)

    assert logits.shape == (5, action_dim)
    assert value.shape == (5,)
