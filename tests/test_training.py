import numpy as np
import pytest
import torch
from torch import nn

from worldpersist import models, sim, training

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def episodes():
    return sim.simulate(12, seed=21)


def test_sample_batch_follows_time_for_single_steps():
    n = 5
    frames = torch.arange(sim.LENGTH, dtype=torch.float32).repeat(n, 1)
    frames = frames[:, :, None, None, None].expand(n, sim.LENGTH, 3, 32, 32)
    actions = torch.arange(sim.LENGTH - 1, dtype=torch.float32).repeat(n, 1)
    hist, act, target = training.sample_batch("b", frames, actions, 64)
    last = hist[:, -1, 0, 0, 0]
    assert hist.shape == (64, 4, 3, 32, 32) and target.shape == (64, 3, 32, 32)
    assert torch.equal(target[:, 0, 0, 0], last + 1)
    assert torch.equal(hist[:, 0, 0, 0, 0], last - 3)
    assert torch.equal(act, last)
    assert target[:, 0, 0, 0].min() >= sim.CONTEXT and target.max() <= sim.LENGTH - 1


def test_sample_batch_for_unrolled_windows():
    n = 5
    frames = torch.arange(sim.LENGTH, dtype=torch.float32).repeat(n, 1)
    frames = frames[:, :, None, None, None].expand(n, sim.LENGTH, 3, 32, 32)
    actions = torch.arange(sim.LENGTH - 1, dtype=torch.float32).repeat(n, 1)
    hist, act, target = training.sample_batch("c", frames, actions, 64)
    last = hist[:, -1, 0, 0, 0]
    assert target.shape == (64, training.UNROLL, 3, 32, 32)
    steps = torch.arange(1, training.UNROLL + 1)
    assert torch.equal(target[:, :, 0, 0, 0], last[:, None] + steps)
    assert torch.equal(act, last[:, None] + steps - 1)
    assert target.max() <= sim.LENGTH - 1


class Identity(nn.Module):
    """Predicts that nothing changes, and counts its calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hist, action):
        self.calls += 1
        return hist[:, -1]


def test_unrolled_loss_feeds_back_the_prediction():
    torch.manual_seed(0)
    hist = torch.rand(3, 4, 3, 32, 32)
    targets = torch.rand(3, training.UNROLL, 3, 32, 32)
    model = Identity()
    loss = training.unrolled_loss(model, hist, torch.zeros(3, training.UNROLL), targets)
    expected = torch.stack(
        [((hist[:, -1] - targets[:, j]) ** 2).mean() for j in range(targets.shape[1])]
    ).mean()
    assert model.calls == training.UNROLL
    assert torch.allclose(loss, expected)


def test_mass_penalty():
    ep = sim.simulate(4, seed=5)
    frames = torch.as_tensor(ep.frames[:, 0])
    assert training.mass_penalty(frames, frames) == 0
    erased = frames.clone()
    erased[:, 0] = 0.0
    assert training.mass_penalty(erased, frames) > 0.01


def test_regression_loss_adds_the_penalty():
    ep = sim.simulate(4, seed=5)
    model = models.make_model("b", 8)
    hist = torch.as_tensor(ep.frames[:, :4])
    target = torch.as_tensor(ep.frames[:, 4])
    action = torch.as_tensor(ep.actions[:, 3])
    plain = training.regression_loss(model, hist, action, target)
    with_penalty = training.regression_loss(model, hist, action, target, lam=2.0)
    penalty = training.mass_penalty(model(hist, action), hist[:, -1])
    assert torch.allclose(with_penalty, plain + 2.0 * penalty)


def test_rollout_never_reads_future_frames(episodes):
    frames = torch.as_tensor(episodes.frames)
    actions = torch.as_tensor(episodes.actions)
    blind = frames.clone()
    blind[:, sim.CONTEXT :] = float("nan")
    for kind in "abd":
        model = models.make_model(kind, 8).eval()
        gen = torch.Generator().manual_seed(3) if model.flow else None
        out = training.rollout(model, blind, actions, gen)
        gen = torch.Generator().manual_seed(3) if model.flow else None
        full = training.rollout(model, frames, actions, gen)
        assert out.shape == frames.shape and torch.isfinite(out).all()
        assert torch.equal(out, full)
        assert torch.equal(out[:, : sim.CONTEXT], frames[:, : sim.CONTEXT])


@pytest.mark.parametrize("kind", ["a", "b", "c", "d", "e"])
def test_every_model_trains_for_a_few_steps(episodes, kind):
    model, log = training.train(
        kind, episodes, steps=3, seed=0, device=CPU, batch=8, base=8, lam=0.5
    )
    assert log["steps"] == 3 and np.isfinite(log["loss"]).all()
    assert not model.training


def test_loss_goes_down_on_a_small_set(episodes):
    _, log = training.train(
        "b", episodes, steps=120, seed=0, device=CPU, batch=16, lr=2e-3, base=8
    )
    assert log["loss"][-1] < 0.5 * log["loss"][0]


def test_training_is_deterministic_for_a_seed(episodes):
    runs = [
        training.train("b", episodes, steps=4, seed=7, device=CPU, batch=8, base=8)[0]
        for _ in range(2)
    ]
    for p, q in zip(runs[0].parameters(), runs[1].parameters(), strict=True):
        assert torch.equal(p, q)


def test_checkpoint_round_trip(tmp_path):
    model = models.make_model("b", 8).eval()
    training.save_model(model, "b", tmp_path / "m.pt", base=8)
    loaded = training.load_model(tmp_path / "m.pt", CPU)
    hist, action = torch.rand(2, 4, 3, 32, 32), torch.zeros(2)
    assert torch.equal(model(hist, action), loaded(hist, action))


class ZeroFlow:
    """A flow sampler with zero velocity, so its prediction is the starting noise."""

    flow, history, steps = True, sim.CONTEXT, 8

    def predict(self, hist, action, generator=None, noise=None):
        def velocity(x, t, hist, action):
            return torch.zeros_like(x)

        return models.euler_sample(velocity, hist, action, self.steps, generator, noise)


def test_episode_noise_depends_only_on_the_seed_and_the_episode():
    whole = training.episode_noise(3, 0, 6)
    assert whole.shape == (6, sim.LENGTH - sim.CONTEXT, 3, sim.SIZE, sim.SIZE)
    assert torch.equal(training.episode_noise(3, 2, 3), whole[2:5])
    assert torch.equal(training.episode_noise(3, 4, 1)[0], whole[4])
    assert not torch.equal(training.episode_noise(4, 0, 6), whole)
    assert not torch.equal(whole[0], whole[1])
    assert not torch.equal(whole[0, 0], whole[0, 1])


def test_noise_follows_the_episode_and_the_step_whatever_the_batch(episodes):
    sampler = ZeroFlow()
    whole = training.predict_episodes(sampler, episodes, CPU, batch=12, seed=5)
    parts = training.predict_episodes(sampler, episodes, CPU, batch=5, seed=5)
    other = training.predict_episodes(sampler, episodes, CPU, batch=12, seed=6)
    assert torch.equal(whole, parts) and not torch.equal(whole, other)
    noise = training.episode_noise(5, 0, len(episodes.frames))
    assert torch.equal(whole[:, sim.CONTEXT :], ((noise + 1) / 2).clamp(0, 1))


def test_batching_does_not_change_a_flow_models_rollouts(episodes):
    model = models.make_model("d", 8).eval()
    whole = training.predict_episodes(model, episodes, CPU, batch=12, seed=5)
    parts = training.predict_episodes(model, episodes, CPU, batch=5, seed=5)
    other = training.predict_episodes(model, episodes, CPU, batch=12, seed=6)
    assert torch.allclose(whole, parts, atol=1e-4)
    assert not torch.allclose(whole, other, atol=1e-2)  # the noise does matter


def test_evaluate_model_returns_all_tables(episodes):
    model = models.make_model("b", 8).eval()
    result = training.evaluate_model(model, episodes, CPU, batch=12)
    tables = {"reappearance", "control", "matched", "paired", "drift_abs", "drift"}
    assert set(result) == tables | {"mse"}
    assert set(result["mse"]) == {"1", "4", "8", "16"}
