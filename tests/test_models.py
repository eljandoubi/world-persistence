import pytest
import torch
from torch import nn

from worldpersist import models


@pytest.mark.parametrize("kind", ["a", "b", "c", "d", "e"])
def test_predict_shape_and_range(kind):
    model = models.make_model(kind, base=8).eval()
    hist = torch.rand(3, 4, 3, 32, 32)
    out = model.predict(hist, torch.tensor([-2.0, 0.0, 2.0]))
    assert out.shape == (3, 3, 32, 32)
    assert out.min() >= 0 and out.max() <= 1


def test_unet_output_shape():
    net = models.UNet(12, 3, cond=64, base=8)
    out = net(torch.rand(2, 12, 32, 32), torch.rand(2, 64))
    assert out.shape == (2, 3, 32, 32)


def test_parameter_budget_is_about_one_million():
    sizes = [models.count_params(models.make_model(k)) for k in "abcde"]
    assert all(1.0e6 < s < 1.3e6 for s in sizes)
    assert max(sizes) - min(sizes) < 0.02 * min(sizes)


def test_memoryless_model_ignores_older_frames():
    torch.manual_seed(0)
    hist = torch.rand(2, 4, 3, 32, 32)
    other = hist.clone()
    other[:, :3] = torch.rand(2, 3, 3, 32, 32)
    action = torch.zeros(2)
    memoryless, history = models.make_model("a", 8), models.make_model("b", 8)
    assert torch.equal(memoryless(hist, action), memoryless(other, action))
    assert not torch.allclose(history(hist, action), history(other, action))


def test_action_changes_the_output_once_the_film_layers_are_set():
    model = models.make_model("b", 8)
    hist = torch.rand(1, 4, 3, 32, 32)
    left, right = torch.tensor([-2.0]), torch.tensor([2.0])
    assert torch.equal(model(hist, left), model(hist, right))
    for m in model.modules():
        if isinstance(m, models.Block):
            nn.init.normal_(m.film.weight, std=0.5)
    assert not torch.allclose(model(hist, left), model(hist, right))


def test_flow_target_at_both_ends():
    x1, noise = torch.rand(4, 3, 8, 8), torch.randn(4, 3, 8, 8)
    start, v0 = models.flow_target(x1, noise, torch.zeros(4))
    end, v1 = models.flow_target(x1, noise, torch.ones(4))
    assert torch.allclose(start, noise) and torch.allclose(end, x1)
    assert torch.allclose(v0, x1 - noise) and torch.allclose(v1, x1 - noise)


class Oracle:
    """A velocity field that knows the data and the noise, and records its input."""

    def __init__(self, x1, noise):
        self.x1, self.noise, self.seen = x1, noise, []

    def velocity(self, x, t, hist, action):
        self.seen.append((x, hist))
        return self.x1 - self.noise


@pytest.mark.parametrize("t", [0.0, 1.0])
def test_flow_loss_vanishes_for_the_exact_velocity(t):
    target, hist = torch.rand(4, 3, 32, 32), torch.rand(4, 4, 3, 32, 32)
    noise = torch.randn(4, 3, 32, 32)
    oracle = Oracle(target * 2 - 1, noise)
    loss = models.flow_loss(
        oracle, hist, torch.zeros(4), target, noise=noise, t=torch.full((4,), t)
    )
    assert loss < 1e-10
    x_seen = oracle.seen[0][0]
    expected = noise if t == 0.0 else target * 2 - 1
    assert torch.allclose(x_seen, expected)


def test_flow_loss_is_positive_for_a_wrong_velocity():
    target, hist = torch.rand(4, 3, 32, 32), torch.rand(4, 4, 3, 32, 32)
    noise = torch.randn(4, 3, 32, 32)
    wrong = Oracle(target * 2 - 1 + 1.0, noise)
    loss = models.flow_loss(wrong, hist, torch.zeros(4), target, noise=noise)
    assert abs(loss.item() - 1.0) < 1e-5


def test_history_noise_is_added_only_when_asked():
    target, hist = torch.rand(2, 3, 32, 32), torch.rand(2, 4, 3, 32, 32)
    oracle = Oracle(target * 2 - 1, torch.zeros_like(target))
    models.flow_loss(oracle, hist, torch.zeros(2), target, history_noise=0.0)
    assert torch.allclose(oracle.seen[0][1], hist * 2 - 1)
    models.flow_loss(oracle, hist, torch.zeros(2), target, history_noise=0.1)
    spread = (oracle.seen[1][1] - (hist * 2 - 1)).abs().max()
    assert 0 < spread < 0.6


def test_flow_sampling_depends_on_the_generator_only():
    model = models.make_model("d", 8).eval()
    hist, action = torch.rand(2, 4, 3, 32, 32), torch.zeros(2)

    def sample(seed):
        gen = torch.Generator().manual_seed(seed)
        return model.predict(hist, action, gen)

    assert torch.equal(sample(1), sample(1))
    assert not torch.equal(sample(1), sample(2))
