"""Occlusion benchmark: three coloured squares behind a grey bar the action moves."""

from dataclasses import dataclass, fields

import numpy as np

SIZE = 32
SQUARE = 5
BAR_W = 8
BAR_MAX = SIZE - BAR_W
POS_MAX = SIZE - SQUARE
LENGTH = 20
CONTEXT = 4
ACTIONS = (-2, 0, 2)
HARD_ACTIONS = (-4, -2, 0, 2, 4)
SPEED = 1.2
HARD_SPEED = 1.8
MIN_SPEED = 0.4
SWITCH = 0.2
BAR_VALUE = 0.5
SPLITS = {
    "train": (4000, 11, False),
    "val": (500, 12, False),
    "test": (1000, 13, False),
    "hard": (1000, 14, True),
}


@dataclass
class Episodes:
    """Frames and actions for the models, true state for scoring only."""

    frames: np.ndarray  # (n, LENGTH, 3, SIZE, SIZE) float32 in [0, 1]
    actions: np.ndarray  # (n, LENGTH - 1) float32, bar move from frame t to t + 1
    pos: np.ndarray  # (n, LENGTH, 3, 2) float64, top-left corner (x, y) of a square
    bar: np.ndarray  # (n, LENGTH) int64, left edge of the bar
    covered: np.ndarray  # (n, LENGTH, 3) float64, share of a square under the bar

    @property
    def hidden(self) -> np.ndarray:
        return self.covered >= 1.0 - 1e-9

    @property
    def visible(self) -> np.ndarray:
        return self.covered <= 1e-9

    def head(self, n: int) -> "Episodes":
        return Episodes(*(getattr(self, f.name)[:n] for f in fields(self)))


def covered_share(x, bar):
    """Share of a square with left edge x that lies under a bar with left edge bar."""
    overlap = np.minimum(x + SQUARE, bar + BAR_W) - np.maximum(x, bar)
    return np.clip(overlap, 0.0, SQUARE) / SQUARE


def advance(pos, vel):
    """Move the squares one frame and reflect them elastically on the walls."""
    pos = pos + vel
    low, high = pos < 0, pos > POS_MAX
    pos = np.where(low, -pos, np.where(high, 2 * POS_MAX - pos, pos))
    return pos, np.where(low | high, -vel, vel)


def render(pos, bar):
    """Draw each square in its own channel with exact pixel coverage, bar on top."""
    px = np.arange(SIZE)
    cover = np.minimum(pos[..., None] + SQUARE, px + 1) - np.maximum(pos[..., None], px)
    cover = np.clip(cover, 0.0, 1.0)  # (..., 3, 2, SIZE): columns for x, rows for y
    frames = np.einsum("...cj,...ci->...cji", cover[..., 1, :], cover[..., 0, :])
    under_bar = (px >= bar[..., None]) & (px < bar[..., None] + BAR_W)
    frames = np.where(under_bar[..., None, None, :], BAR_VALUE, frames)
    return frames.astype(np.float32)


def simulate(n, seed, hard=False) -> Episodes:
    """Draw n episodes. The hard variant has faster squares and a faster bar."""
    rng = np.random.default_rng(seed)
    top = HARD_SPEED if hard else SPEED
    moves = np.array(HARD_ACTIONS if hard else ACTIONS)

    pos = np.empty((n, LENGTH, 3, 2))
    pos[:, 0] = rng.uniform(0, POS_MAX, size=(n, 3, 2))
    vel = rng.uniform(-top, top, size=(n, 3, 2))
    slow = np.abs(vel).max(axis=-1) <= MIN_SPEED
    while slow.any():
        vel[slow] = rng.uniform(-top, top, size=(int(slow.sum()), 2))
        slow = np.abs(vel).max(axis=-1) <= MIN_SPEED
    for t in range(1, LENGTH):
        pos[:, t], vel = advance(pos[:, t - 1], vel)

    actions = np.empty((n, LENGTH - 1), dtype=np.int64)
    actions[:, 0] = rng.choice(moves, size=n)
    for t in range(1, LENGTH - 1):
        redraw = rng.random(n) < SWITCH
        actions[:, t] = np.where(redraw, rng.choice(moves, size=n), actions[:, t - 1])
    bar = np.empty((n, LENGTH), dtype=np.int64)
    bar[:, 0] = rng.integers(0, BAR_MAX + 1, size=n)
    for t in range(LENGTH - 1):
        bar[:, t + 1] = np.clip(bar[:, t] + actions[:, t], 0, BAR_MAX)

    frames = np.empty((n, LENGTH, 3, SIZE, SIZE), dtype=np.float32)
    for i in range(0, n, 250):
        frames[i : i + 250] = render(pos[i : i + 250], bar[i : i + 250])
    covered = covered_share(pos[..., 0], bar[..., None])
    return Episodes(frames, actions.astype(np.float32), pos, bar, covered)


def load_split(name, n=None) -> Episodes:
    """One of the fixed splits (train, val, test, hard); n draws a smaller set."""
    size, seed, hard = SPLITS[name]
    return simulate(size if n is None else n, seed, hard)
