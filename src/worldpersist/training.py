"""Training, rollouts and checkpoints for the five predictors."""

import math
import time

import torch
import torch.nn.functional as F

from . import metrics
from .models import count_params, flow_loss, make_model
from .sim import CONTEXT, LENGTH, SIZE

UNROLL = 4


def mass_penalty(pred, prev):
    """Squared change of each colour's mass between the previous and predicted frame."""
    return (
        ((metrics.masses(pred) - metrics.masses(prev)) / metrics.NOMINAL).pow(2).mean()
    )


def regression_loss(model, hist, action, target, lam=0.0):
    """Pixel MSE of the next frame, plus lam times the mass penalty."""
    pred = model(hist, action).float()
    loss = F.mse_loss(pred, target)
    if lam > 0:
        loss = loss + lam * mass_penalty(pred, hist[:, -1])
    return loss


def unrolled_loss(model, hist, actions, targets):
    """Feed the model its own clamped predictions for every step and score each one."""
    total = 0.0
    for j in range(targets.shape[1]):
        pred = model(hist, actions[:, j]).float()
        total = total + F.mse_loss(pred, targets[:, j])
        hist = torch.cat([hist[:, 1:], pred.clamp(0, 1)[:, None]], dim=1)
    return total / targets.shape[1]


def sample_batch(kind, frames, actions, batch):
    """Random training windows: history, actions and targets (UNROLL targets for c)."""
    device = frames.device
    ep = torch.randint(0, len(frames), (batch,), device=device)
    back = torch.arange(-CONTEXT, 0, device=device)
    if kind == "c":
        start = torch.randint(CONTEXT, LENGTH - UNROLL + 1, (batch,), device=device)
        at = start[:, None] + torch.arange(UNROLL, device=device)
        hist = frames[ep[:, None], start[:, None] + back]
        return (
            hist.float(),
            actions[ep[:, None], at - 1],
            frames[ep[:, None], at].float(),
        )
    t = torch.randint(CONTEXT, LENGTH, (batch,), device=device)
    hist = frames[ep[:, None], t[:, None] + back]
    return hist.float(), actions[ep, t - 1], frames[ep, t].float()


def lr_factor(step, steps, warmup):
    """Linear warm-up, then cosine decay down to 5 percent of the peak rate."""
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))


def train(
    kind,
    episodes,
    steps,
    seed,
    device,
    batch=128,
    lr=1e-3,
    lam=0.0,
    base=32,
    model=None,
):
    """Train one predictor for a fixed number of steps; returns the model and a log.

    Every step runs the network on `batch` frames: model c uses batch // UNROLL
    windows of UNROLL steps, so all models see the same amount of network work.
    """
    torch.manual_seed(seed)
    cuda = device.type == "cuda"
    dtype = torch.float16 if cuda else torch.float32
    frames = torch.as_tensor(episodes.frames).to(device, dtype)
    actions = torch.as_tensor(episodes.actions).to(device)
    model = (make_model(kind, base) if model is None else model).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    warmup = min(100, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_factor(s, steps, warmup)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=cuda)
    windows = max(1, batch // UNROLL) if kind == "c" else batch
    every = max(1, steps // 20)
    curve, running = [], torch.zeros((), device=device)
    start = time.time()
    for step in range(steps):
        hist, act, target = sample_batch(kind, frames, actions, windows)
        with torch.autocast(device.type, dtype=torch.float16, enabled=cuda):
            if kind == "c":
                loss = unrolled_loss(model, hist, act, target)
            elif kind == "d":
                loss = flow_loss(model, hist, act, target)
            else:
                loss = regression_loss(model, hist, act, target, lam)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        running += loss.detach()
        if (step + 1) % every == 0:
            curve.append(round(running.item() / every, 6))
            running.zero_()
    if cuda:
        torch.cuda.synchronize()
    log = {
        "steps": steps,
        "batch": batch,
        "lr": lr,
        "lam": lam,
        "params": count_params(model),
        "seconds": round(time.time() - start, 1),
        "loss": curve,
    }
    return model.eval(), log


@torch.no_grad()
def rollout(model, frames, actions, generator=None, noise=None):
    """Roll out from the first CONTEXT frames; later frames are never read.

    `noise` (episodes, rollout steps, 3, SIZE, SIZE) gives a flow model its starting
    noise at each step; without it the noise comes from `generator`.
    """
    seq = [frames[:, t] for t in range(CONTEXT)]
    for t in range(CONTEXT, LENGTH):
        hist = torch.stack(seq[-CONTEXT:], dim=1)
        step = None if noise is None else noise[:, t - CONTEXT]
        seq.append(model.predict(hist, actions[:, t - 1], generator, step))
    return torch.stack(seq, dim=1)


def episode_noise(seed, first, count):
    """Flow noise for episodes first to first + count - 1, one slice per rollout step.

    Each episode has its own CPU generator, seeded with seed * 1000003 + the episode
    number, so its noise does not depend on how episodes are batched or on the device.
    """
    steps = LENGTH - CONTEXT
    out = torch.empty(count, steps, 3, SIZE, SIZE)
    for i in range(count):
        gen = torch.Generator().manual_seed(seed * 1_000_003 + first + i)
        out[i] = torch.randn(steps, 3, SIZE, SIZE, generator=gen)
    return out


@torch.no_grad()
def predict_episodes(model, episodes, device, batch=250, seed=0, amp=True):
    """Roll the model out on every episode and return the frames on the CPU.

    A flow model gets `episode_noise(seed, ...)`, so an episode's result does not
    depend on the batch size, the device or which other episodes are scored with it.
    """
    half = amp and device.type == "cuda"
    out = []
    for i in range(0, len(episodes.frames), batch):
        frames = torch.as_tensor(episodes.frames[i : i + batch]).to(device)
        actions = torch.as_tensor(episodes.actions[i : i + batch]).to(device)
        noise = episode_noise(seed, i, len(frames)).to(device) if model.flow else None
        with torch.autocast(device.type, dtype=torch.float16, enabled=half):
            out.append(rollout(model, frames, actions, noise=noise).float().cpu())
    return torch.cat(out)


def evaluate_model(model, episodes, device, batch=250, seed=0, amp=True):
    """Persistence metrics of a model rolled out on a set of episodes."""
    frames = predict_episodes(model, episodes, device, batch, seed, amp)
    return metrics.evaluate(frames, episodes)


def save_model(model, kind, path, base=32):
    torch.save({"kind": kind, "base": base, "state": model.state_dict()}, path)


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = make_model(ckpt["kind"], ckpt["base"])
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval()
