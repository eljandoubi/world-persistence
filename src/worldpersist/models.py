"""Small action-conditioned UNets: regression predictors and a flow-matching one."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .sim import CONTEXT, SIZE

ACTION_SCALE = 2.0
TIME_DIM = 32


class Block(nn.Module):
    """Residual block; an embedding scales and shifts the second normalisation."""

    def __init__(self, cin, cout, cond):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.film = nn.Linear(cond, 2 * cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.conv2(F.silu(h)) + self.skip(x)


class UNet(nn.Module):
    """Three resolutions (32, 16, 8 px), about 1M parameters at base width 32."""

    def __init__(self, in_ch, out_ch, cond=64, base=32):
        super().__init__()
        c1, c2, c3 = base, 2 * base, 4 * base
        self.inp = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.down1 = Block(c1, c1, cond)
        self.down2 = Block(c1, c2, cond)
        self.down3 = Block(c2, c3, cond)
        self.mid = nn.ModuleList([Block(c3, c3, cond), Block(c3, c3, cond)])
        self.up2 = Block(c3 + c2, c2, cond)
        self.up1 = Block(c2 + c1, c1, cond)
        self.norm = nn.GroupNorm(8, c1)
        self.out = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x, emb):
        h1 = self.down1(self.inp(x), emb)
        h2 = self.down2(F.avg_pool2d(h1, 2), emb)
        h3 = self.down3(F.avg_pool2d(h2, 2), emb)
        for block in self.mid:
            h3 = block(h3, emb)
        u2 = self.up2(torch.cat([F.interpolate(h3, scale_factor=2.0), h2], dim=1), emb)
        u1 = self.up1(torch.cat([F.interpolate(u2, scale_factor=2.0), h1], dim=1), emb)
        return self.out(F.silu(self.norm(u1)))


def mlp(inp, width):
    return nn.Sequential(nn.Linear(inp, width), nn.SiLU(), nn.Linear(width, width))


def time_features(t):
    """Sine and cosine features of t in [0, 1] at frequencies from 1 to 100."""
    freqs = torch.exp(
        torch.linspace(0.0, math.log(100.0), TIME_DIM // 2, device=t.device)
    )
    angles = t[:, None] * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class Regressor(nn.Module):
    """Predicts the next frame from the last `history` frames and the bar action."""

    flow = False

    def __init__(self, history=CONTEXT, base=32, cond=64):
        super().__init__()
        self.history = history
        self.act = mlp(1, cond)
        self.net = UNet(3 * history, 3, cond, base)

    def forward(self, hist, action):
        hist = hist[:, -self.history :]
        emb = self.act(action[:, None] / ACTION_SCALE)
        return hist[:, -1] + self.net(hist.flatten(1, 2), emb)

    @torch.no_grad()
    def predict(self, hist, action, generator=None, noise=None):
        return self(hist, action).clamp(0, 1)


class Flow(nn.Module):
    """Conditional flow matching for the next frame, sampled with Euler steps."""

    flow = True

    def __init__(self, history=CONTEXT, base=32, cond=64, steps=8):
        super().__init__()
        self.history, self.steps = history, steps
        self.act = mlp(1, cond)
        self.time = mlp(TIME_DIM, cond)
        self.net = UNet(3 + 3 * history, 3, cond, base)

    def velocity(self, x, t, hist, action):
        """Velocity at x and time t; frames are in [-1, 1] and hist is signed too."""
        emb = self.act(action[:, None] / ACTION_SCALE) + self.time(time_features(t))
        hist = hist[:, -self.history :].flatten(1, 2)
        return self.net(torch.cat([x, hist], dim=1), emb)

    @torch.no_grad()
    def predict(self, hist, action, generator=None, noise=None):
        return euler_sample(self.velocity, hist, action, self.steps, generator, noise)


def euler_sample(velocity, hist, action, steps, generator=None, noise=None):
    """Integrate a velocity field from noise to a frame in [0, 1] with Euler steps.

    The starting noise is `noise` when given and is drawn from `generator` otherwise.
    """
    hist = hist * 2 - 1
    if noise is None:
        noise = torch.randn(
            len(hist), 3, SIZE, SIZE, generator=generator, device=hist.device
        )
    x = noise.to(hist.device)
    for i in range(steps):
        t = torch.full((len(x),), i / steps, device=x.device)
        x = x + velocity(x, t, hist, action) / steps
    return ((x + 1) / 2).clamp(0, 1)


def flow_target(x1, noise, t):
    """Point at time t on the straight path from noise to data, and its velocity."""
    t = t[:, None, None, None]
    return (1 - t) * noise + t * x1, x1 - noise


def flow_loss(model, hist, action, target, history_noise=0.1, noise=None, t=None):
    """Flow matching loss; the history frames get noise of a random level."""
    x1, hist = target * 2 - 1, hist * 2 - 1
    sigma = history_noise * torch.rand(len(hist), 1, 1, 1, 1, device=hist.device)
    hist = hist + sigma * torch.randn_like(hist)
    noise = torch.randn_like(x1) if noise is None else noise
    t = torch.rand(len(x1), device=x1.device) if t is None else t
    x, velocity = flow_target(x1, noise, t)
    return F.mse_loss(model.velocity(x, t, hist, action).float(), velocity)


def make_model(kind, base=32):
    """A is memoryless, B, C and E use the four-frame history, D is the flow model."""
    if kind == "a":
        return Regressor(history=1, base=base)
    if kind in ("b", "c", "e"):
        return Regressor(history=CONTEXT, base=base)
    if kind == "d":
        return Flow(history=CONTEXT, base=base)
    raise ValueError(f"unknown model {kind!r}")


def count_params(model):
    return sum(p.numel() for p in model.parameters())
