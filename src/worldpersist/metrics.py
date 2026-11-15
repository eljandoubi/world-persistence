"""Persistence metrics: colour masks, centroids, reappearance error, mass drift."""

import numpy as np
import torch

from .sim import CONTEXT, SIZE, SQUARE

DEAD = 0.1
MIN_MASS = 5.0
RADIUS = 2.0
MISS = float(SIZE)
HORIZONS = (1, 4, 8, 16)
LENGTHS = (1, 2, 3, 4, 5, 6)
NOMINAL = float(SQUARE * SQUARE)


def colour_masks(frames):
    """Soft mask per colour: channel above the grey level, small values dropped."""
    frames = frames.clamp(0, 1)
    above = frames - frames.amin(dim=-3, keepdim=True)
    return (above - DEAD).clamp_min(0) / (1 - DEAD)


def masses(frames):
    """Pixel mass of each colour, shape (..., 3)."""
    return colour_masks(frames).sum(dim=(-2, -1))


def centroids(frames):
    """Centroid (x, y) and mass of each colour; pixel centres sit at i + 0.5."""
    mask = colour_masks(frames)
    mass = mask.sum(dim=(-2, -1))
    grid = torch.arange(SIZE, dtype=mask.dtype, device=mask.device) + 0.5
    cx = (mask.sum(dim=-2) * grid).sum(dim=-1) / mass.clamp_min(1e-6)
    cy = (mask.sum(dim=-1) * grid).sum(dim=-1) / mass.clamp_min(1e-6)
    return torch.stack([cx, cy], dim=-1), mass


def occlusion_events(covered):
    """Hidden runs that end with the square fully visible again.

    Rows are (episode, colour, run length k, first fully visible frame). Runs that
    break off into another hidden run first, and reappearances inside the context
    frames, are skipped.
    """
    hidden = covered >= 1 - 1e-9
    visible = covered <= 1e-9
    events = []
    for e, c in np.ndindex(covered.shape[0], covered.shape[2]):
        h, v = hidden[e, :, c], visible[e, :, c]
        t = 0
        while t < len(h):
            if not h[t]:
                t += 1
                continue
            start = t
            while t < len(h) and h[t]:
                t += 1
            after = t
            while after < len(h) and not (h[after] or v[after]):
                after += 1
            if after < len(h) and v[after] and after >= CONTEXT:
                events.append((e, c, t - start, after))
    return np.array(events, dtype=np.int64).reshape(-1, 4)


def control_events(covered, events):
    """Pair each event with another colour that stays fully visible around its frame.

    Events with no such colour are dropped. Rows are (episode, colour, k, frame).
    """
    visible = covered <= 1e-9
    rows = []
    for e, c, k, r in events:
        window = visible[e, max(r - 7, 0) : r + 1]
        others = [o for o in range(window.shape[1]) if o != c and window[:, o].all()]
        if others:
            rows.append((e, others[0], k, r))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def reappearance(pred, pos, events):
    """Pixel distance between detected and true centre at each reappearance.

    A colour with less than MIN_MASS detected is a miss and gets the error MISS.
    """
    e, c, _, r = (torch.as_tensor(col) for col in events.T)
    centre, mass = centroids(pred[e, r])
    rows = torch.arange(len(e))
    centre, mass = centre[rows, c], mass[rows, c]
    true = torch.as_tensor(pos[e.numpy(), r.numpy(), c.numpy()] + SQUARE / 2)
    error = (centre - true.to(centre.dtype)).norm(dim=-1)
    found = mass >= MIN_MASS
    return torch.where(
        found, error, torch.full_like(error, MISS)
    ).numpy(), found.numpy()


def summarize(error, found, k):
    """Median error, share within RADIUS and miss share for each run length k."""
    table = {}
    groups = [(str(n), k == n) for n in LENGTHS] + [("all", np.isin(k, LENGTHS))]
    for name, rows in groups:
        err, hit = error[rows], found[rows]
        table[name] = {
            "n": int(rows.sum()),
            "median": round(float(np.median(err)), 4) if len(err) else None,
            "share": round(float((err < RADIUS).mean()), 4) if len(err) else None,
            "miss": round(float((~hit).mean()), 4) if len(err) else None,
        }
    return table


def evaluate(pred, ep):
    """Persistence numbers for rolled-out frames of shape (n, LENGTH, 3, SIZE, SIZE).

    The control entry scores a square that stayed visible next to each event.
    """
    pred = pred.clamp(0, 1)
    truth = torch.as_tensor(ep.frames)
    events = occlusion_events(ep.covered)
    error, found = reappearance(pred, ep.pos, events)
    controls = control_events(ep.covered, events)
    control_error, control_found = reappearance(pred, ep.pos, controls)
    at = [CONTEXT - 1 + h for h in HORIZONS]
    drift = (masses(pred[:, at]) - masses(truth[:, at])) / NOMINAL
    mse = ((pred[:, at] - truth[:, at]) ** 2).mean(dim=(2, 3, 4))

    def by_horizon(values):
        return {
            str(h): round(float(values[:, i].mean()), 6) for i, h in enumerate(HORIZONS)
        }

    return {
        "reappearance": summarize(error, found, events[:, 2]),
        "control": summarize(control_error, control_found, controls[:, 2]),
        "drift_abs": by_horizon(drift.abs().mean(dim=-1)),
        "drift": by_horizon(drift.mean(dim=-1)),
        "mse": by_horizon(mse),
    }
