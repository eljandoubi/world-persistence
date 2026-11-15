"""Episode-clustered bootstrap for per-event outcomes."""

import numpy as np


def episode_bootstrap(episode, values, draws=2000, seed=0):
    """Means of per-event `values` over episodes resampled with replacement.

    `values` has shape (events, columns). The events of one episode move together, so
    the spread of the draws reflects the choice of test episodes and not of single
    events. Returns an array of shape (draws, columns).
    """
    values = np.asarray(values, dtype=float).reshape(len(episode), -1)
    ids, index = np.unique(episode, return_inverse=True)
    sums = np.zeros((len(ids), values.shape[1]))
    np.add.at(sums, index, values)
    counts = np.bincount(index).astype(float)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, draws, 250):
        pick = rng.integers(0, len(ids), size=(min(250, draws - start), len(ids)))
        out.append(sums[pick].sum(axis=1) / counts[pick].sum(axis=1)[:, None])
    return np.concatenate(out)


def interval(draws, level=0.95):
    """Percentile interval of bootstrap draws along the first axis: (low, high)."""
    tail = 50 * (1 - level)
    return np.percentile(draws, [tail, 100 - tail], axis=0)
