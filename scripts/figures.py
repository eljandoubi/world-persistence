"""Figures: a rollout strip across one occlusion, and the reappearance curve."""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from worldpersist import metrics, sim, training

NAMES = {"a": "A", "b": "B", "c": "C", "d": "D", "e": "E"}
COLOURS = {
    "a": "#888888",
    "b": "#1f77b4",
    "c": "#2ca02c",
    "d": "#d62728",
    "e": "#9467bd",
}


def strip(args):
    """Truth and every model across the first test occlusion of k hidden frames."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = sim.load_split("test", args.eval_episodes).head(250)
    events = [
        ev for ev in metrics.occlusion_events(episodes.covered) if ev[2] == args.k
    ]
    if not events:
        raise SystemExit(f"no occlusion of {args.k} frames in the first 250 episodes")
    e, c, k, r = events[0]
    first, last = max(r - k - 2, 0), min(r + 2, sim.LENGTH - 1)
    rows = {"truth": torch.as_tensor(episodes.frames)}
    for name in NAMES:
        model = training.load_model(pathlib.Path(args.ckpt) / f"{name}_s0.pt", device)
        rows[NAMES[name]] = training.predict_episodes(model, episodes, device)
    times = list(range(first, last + 1))
    fig, axes = plt.subplots(len(rows), len(times), figsize=(0.95 * len(times), 6.2))
    for i, (label, frames) in enumerate(rows.items()):
        for j, t in enumerate(times):
            ax = axes[i, j]
            ax.imshow(frames[e, t].permute(1, 2, 0).numpy(), interpolation="nearest")
            x, y = episodes.pos[e, t, c] + sim.SQUARE / 2 - 0.5
            ax.plot(x, y, marker="+", color="white", markersize=7, mew=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title("context" if t < sim.CONTEXT else f"t={t}", fontsize=7)
            if j == 0:
                ax.set_ylabel(label, fontsize=8)
    fig.suptitle(
        f"test episode {e}, colour {'RGB'[c]}: hidden for {k} frames, visible again at "
        f"t={r} ('+' marks the true centre)",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = pathlib.Path(args.out) / "rollout_strip.png"
    fig.savefig(path, dpi=110)
    print(path, "episode", int(e), "event", int(k), int(r))


def curve(args):
    """Share of reappearances within 2 px against occlusion length, test and hard."""
    runs = {}
    for path in sorted((pathlib.Path(args.out) / "runs").glob("*_s*.json")):
        run = json.loads(path.read_text())
        if run["name"] in NAMES:
            runs.setdefault(run["name"], []).append(run)
    lengths = [str(n) for n in metrics.LENGTHS]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, split in zip(axes, ("test", "hard"), strict=True):
        for name, group in runs.items():
            share = np.array(
                [
                    [r["metrics"][split]["reappearance"][k]["share"] for k in lengths]
                    for r in group
                ],
                dtype=float,
            )
            spread = share.std(axis=0, ddof=1) if len(group) > 1 else None
            ax.errorbar(
                range(1, len(lengths) + 1),
                100 * share.mean(axis=0),
                yerr=None if spread is None else 100 * spread,
                label=NAMES[name],
                color=COLOURS[name],
                marker="o",
                markersize=3,
                capsize=2,
                linewidth=1,
            )
        ax.set_title(split, fontsize=9)
        ax.set_xlabel("frames hidden (k)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reappear within 2 px (%)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = pathlib.Path(args.out) / "reappearance_curve.png"
    fig.savefig(path, dpi=110)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("figure", choices=["strip", "curve"])
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default="work/ckpt")
    p.add_argument("--k", type=int, default=3, help="hidden frames of the strip event")
    p.add_argument("--eval-episodes", type=int)
    args = p.parse_args()
    plt.switch_backend("Agg")
    {"strip": strip, "curve": curve}[args.figure](args)


if __name__ == "__main__":
    main()
