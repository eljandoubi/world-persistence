import numpy as np
import pytest
import torch
from torch import nn

from worldpersist import frugal, models, sim, training

pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")


def tuned_model(kind, seed=0):
    """A small model whose FiLM layers are non-zero, so the action matters."""
    torch.manual_seed(seed)
    model = models.make_model(kind, base=8).eval()
    for module in model.modules():
        if isinstance(module, models.Block):
            nn.init.normal_(module.film.weight, std=0.2)
    return model


@pytest.mark.parametrize("kind", ["a", "b", "d"])
def test_onnx_output_equals_torch(kind, tmp_path):
    model = tuned_model(kind)
    path = tmp_path / "step.onnx"
    frugal.export_onnx(model, path)
    session = frugal.make_session(path)
    for batch in (1, 3):
        inputs = frugal.example_inputs(model, batch=batch, seed=batch)
        inputs["action"] = torch.tensor([-2.0, 0.0, 2.0])[:batch]
        with torch.no_grad():
            expected = frugal.Step(model)(*inputs.values()).numpy()
        got = session.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
        assert got.shape == expected.shape
        assert np.abs(got - expected).max() < 1e-4


@pytest.mark.parametrize("kind", ["b", "d"])
def test_onnx_predictor_rolls_out_like_the_model(kind, tmp_path):
    model = tuned_model(kind)
    path = tmp_path / "step.onnx"
    frugal.export_onnx(model, path)
    predictor = frugal.OnnxPredictor(path, model)
    ep = sim.simulate(4, seed=9)
    frames, actions = torch.as_tensor(ep.frames), torch.as_tensor(ep.actions)
    gen = torch.Generator().manual_seed(1)
    expected = training.rollout(model, frames, actions, gen)
    gen = torch.Generator().manual_seed(1)
    got = training.rollout(predictor, frames, actions, gen)
    assert (got - expected).abs().max() < 1e-3


@pytest.mark.parametrize("kind", ["b", "d"])
def test_onnx_and_torch_agree_when_the_noise_is_the_same(kind, tmp_path):
    model = tuned_model(kind)
    path = tmp_path / "step.onnx"
    frugal.export_onnx(model, path)
    predictor = frugal.OnnxPredictor(path, model)
    ep, cpu = sim.simulate(6, seed=9), torch.device("cpu")
    expected = training.predict_episodes(model, ep, cpu, batch=6, seed=2)
    got = training.predict_episodes(predictor, ep, cpu, batch=2, seed=2)
    assert (got - expected).abs().max() < 1e-3


def test_latency_reports_ordered_percentiles(tmp_path):
    model = tuned_model("b")
    path = tmp_path / "step.onnx"
    frugal.export_onnx(model, path)
    result = frugal.latency(path, model, runs=6, warmup=1)
    assert 0 < result["p10_ms"] <= result["median_ms"] <= result["p90_ms"]
    assert result["runs"] == 6 and result["threads"] == 1


def test_calibration_inputs_match_the_graph(tmp_path):
    ep = sim.simulate(10, seed=4)
    for kind in ("a", "d"):
        model = tuned_model(kind)
        data = frugal.calibration_inputs(model, ep, n=7)
        assert list(data) == list(frugal.example_inputs(model))
        assert all(len(v) == 7 for v in data.values())
        assert data["hist"].shape[1] == model.history


def test_int8_model_is_smaller_and_close(tmp_path):
    model = tuned_model("b")
    ep = sim.simulate(10, seed=4)
    fp32, int8 = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    frugal.export_onnx(model, fp32)
    frugal.quantize(fp32, int8, frugal.calibration_inputs(model, ep, n=16))
    assert frugal.size_mb(int8) < frugal.size_mb(fp32)
    inputs = {k: v.numpy() for k, v in frugal.example_inputs(model, batch=2).items()}
    full = frugal.make_session(fp32).run(None, inputs)[0]
    small = frugal.make_session(int8).run(None, inputs)[0]
    # u8s8 kernels round differently on CPUs without VNNI, so the bound is loose
    assert np.isfinite(small).all()
    assert np.abs(full - small).mean() < 0.3 * np.abs(full).mean()


def test_pruning_reaches_the_target_and_masks_stay_fixed():
    ep = sim.simulate(8, seed=2)
    model = tuned_model("b")
    assert frugal.sparsity(model) < 0.01
    frugal.prune_(model, 0.5)
    tuned, _ = training.train(
        "b", ep, steps=3, seed=0, device=torch.device("cpu"), batch=8, model=model
    )
    frugal.make_permanent(tuned)
    assert abs(frugal.sparsity(tuned) - 0.5) < 0.01
    assert not any(
        hasattr(layer, "weight_mask") for layer in frugal.weight_layers(tuned)
    )
