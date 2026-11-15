"""ONNX export, static int8 quantisation, CPU latency and magnitude pruning."""

import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from . import models, training
from .sim import SIZE


class Step(nn.Module):
    """The network that runs once per generated frame, with plain tensor inputs."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.velocity(*inputs) if self.model.flow else self.model(*inputs)


def example_inputs(model, batch=1, seed=0):
    """Random inputs for the one-step network, in the order the graph takes them."""
    gen = torch.Generator().manual_seed(seed)
    inputs = {}
    if model.flow:
        inputs["x"] = torch.randn(batch, 3, SIZE, SIZE, generator=gen)
        inputs["t"] = torch.full((batch,), 0.5)
    inputs["hist"] = torch.rand(batch, model.history, 3, SIZE, SIZE, generator=gen)
    inputs["action"] = torch.zeros(batch)
    return inputs


def export_onnx(model, path):
    """Write the one-step network as fp32 ONNX with a dynamic batch dimension."""
    inputs = example_inputs(model)
    names = list(inputs)
    torch.onnx.export(
        Step(model).eval(),
        tuple(inputs.values()),
        str(path),
        input_names=names,
        output_names=["out"],
        dynamic_axes={name: {0: "batch"} for name in [*names, "out"]},
        opset_version=17,
        dynamo=False,
    )


def make_session(path, threads=1):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxPredictor:
    """Rolls out like a torch model but runs an ONNX network on the CPU."""

    def __init__(self, path, like, threads=1):
        self.session = make_session(path, threads)
        self.names = list(example_inputs(like))
        self.flow, self.history = like.flow, like.history
        self.steps = getattr(like, "steps", 1)

    def run(self, *inputs):
        feed = {n: x.numpy() for n, x in zip(self.names, inputs, strict=True)}
        return torch.from_numpy(self.session.run(None, feed)[0])

    def velocity(self, x, t, hist, action):
        return self.run(x, t, hist, action)

    def predict(self, hist, action, generator=None, noise=None):
        hist = hist[:, -self.history :]
        if self.flow:
            return models.euler_sample(
                self.velocity, hist, action, self.steps, generator, noise
            )
        return self.run(hist, action).clamp(0, 1)


def latency(path, like, runs=200, warmup=20, threads=1):
    """Milliseconds of one network call at batch 1 and the load during the runs."""
    session = make_session(path, threads)
    feed = {n: x.numpy() for n, x in example_inputs(like).items()}
    for _ in range(warmup):
        session.run(None, feed)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, feed)
        times.append((time.perf_counter() - start) * 1000)
    p10, p50, p90 = np.percentile(times, [10, 50, 90])
    return {
        "p10_ms": round(float(p10), 3),
        "median_ms": round(float(p50), 3),
        "p90_ms": round(float(p90), 3),
        "runs": runs,
        "threads": threads,
        "loadavg": round(os.getloadavg()[0], 2),
    }


def calibration_inputs(like, episodes, n=200, seed=0):
    """One-step inputs from n training-style windows of the episodes, by input name."""
    torch.manual_seed(seed)
    frames, actions = (
        torch.as_tensor(episodes.frames),
        torch.as_tensor(episodes.actions),
    )
    hist, action, target = training.sample_batch("b", frames, actions, n)
    hist = hist[:, -like.history :]
    inputs = {}
    if like.flow:
        x1 = target * 2 - 1
        t = torch.rand(n)
        inputs["x"], _ = models.flow_target(x1, torch.randn_like(x1), t)
        inputs["t"] = t
        hist = hist * 2 - 1
    inputs["hist"], inputs["action"] = hist, action
    return {name: value.numpy() for name, value in inputs.items()}


def quantize(src, dst, calibration):
    """Static int8 quantisation (QDQ); calibration maps input names to arrays."""
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            count = len(next(iter(calibration.values())))
            self.feeds = iter(
                [
                    {k: v[i : i + 1] for k, v in calibration.items()}
                    for i in range(count)
                ]
            )

        def get_next(self):
            return next(self.feeds, None)

    prepared = f"{dst}.prep"
    quant_pre_process(str(src), prepared, skip_symbolic_shape=True)
    quantize_static(
        prepared,
        str(dst),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    os.remove(prepared)


def size_mb(path):
    return round(os.path.getsize(path) / 1e6, 3)


def weight_layers(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d | nn.Linear)]


def prune_(model, amount):
    """Zero the smallest-magnitude weights of every conv and linear layer."""
    for layer in weight_layers(model):
        prune.l1_unstructured(layer, "weight", amount=amount)


def make_permanent(model):
    """Bake the pruning masks into the weights."""
    for layer in weight_layers(model):
        if prune.is_pruned(layer):
            prune.remove(layer, "weight")


def sparsity(model):
    """Share of conv and linear weights that are exactly zero."""
    layers = weight_layers(model)
    zeros = sum(int((layer.weight == 0).sum()) for layer in layers)
    return zeros / sum(layer.weight.numel() for layer in layers)
