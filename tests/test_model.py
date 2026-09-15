import numpy as np

from adaptive_memory.model import AdaptiveMLP


def test_forward_is_probability_distribution() -> None:
    model = AdaptiveMLP(4, 6, 3, seed=7)
    probabilities, hidden = model.forward(np.ones((5, 4)))
    assert probabilities.shape == (5, 3)
    assert hidden.shape == (5, 6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_baseline_does_not_decay_during_idle_period() -> None:
    model = AdaptiveMLP(4, 6, 3, mechanism="baseline", forgetting_rate=0.1, seed=7)
    before = model.weight_norm()
    model.apply_forgetting(20)
    assert model.weight_norm() == before


def test_reinforcement_protects_used_pathway() -> None:
    passive = AdaptiveMLP(4, 2, 3, mechanism="passive", forgetting_rate=0.1, seed=7)
    reinforced = passive.clone()
    reinforced.mechanism = "reinforced"
    reinforced.usage[:] = [1.0, 0.0]
    passive.apply_forgetting(5)
    reinforced.apply_forgetting(5)
    assert np.linalg.norm(reinforced.w1[:, 0]) > np.linalg.norm(passive.w1[:, 0])


def test_training_reduces_toy_loss() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] > 0).astype(int)
    model = AdaptiveMLP(4, 8, 2, seed=3)
    first = model.train_batch(x, y, 0.1).loss
    for _ in range(40):
        last = model.train_batch(x, y, 0.1).loss
    assert last < first
