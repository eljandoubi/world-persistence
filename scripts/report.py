"""Print the README tables from results/ (mean and standard deviation over seeds)."""

import argparse
import json
import pathlib

import numpy as np

from worldpersist import stats

NAMES = {
    "static": "static copy",
    "a": "A memoryless",
    "b": "B history",
    "c": "C unrolled",
    "d": "D flow",
    "e": "E mass penalty",
}
LENGTHS = ["1", "2", "3", "4", "5", "6", "all"]
HORIZONS = ["1", "4", "8", "16"]
REFERENCE = "b"


def load_runs(out):
    runs = {}
    baseline = out / "baseline.json"
    if baseline.exists():
        runs["static"] = [
            {"name": "static", "metrics": json.loads(baseline.read_text())}
        ]
    for path in sorted((out / "runs").glob("*_s*.json")):
        run = json.loads(path.read_text())
        if run["name"] in NAMES:
            runs.setdefault(run["name"], []).append(run)
    return runs


def cell(values, scale=1.0, digits=1):
    if any(v is None for v in values):
        return "n/a"
    arr = np.array(values, dtype=float) * scale
    text = f"{arr.mean():.{digits}f}"
    return text + (f" ± {arr.std(ddof=1):.{digits}f}" if len(arr) > 1 else "")


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def reappearance(runs, split, field, scale, digits, key="reappearance"):
    """One row per model, one column per occlusion length k."""
    first = next(iter(runs.values()))[0]["metrics"][split][key]
    counts = ["events"] + [first[k]["n"] for k in LENGTHS]
    rows = [counts]
    for name, label in NAMES.items():
        if name not in runs:
            continue
        cells = [
            cell(
                [r["metrics"][split][key][k][field] for r in runs[name]], scale, digits
            )
            for k in LENGTHS
        ]
        rows.append([label] + cells)
    return table(["model"] + [k if k == "all" else f"k={k}" for k in LENGTHS], rows)


def horizons(runs, split, field, scale, digits):
    rows = []
    for name, label in NAMES.items():
        if name in runs:
            values = [
                cell([r["metrics"][split][field][h] for r in runs[name]], scale, digits)
                for h in HORIZONS
            ]
            rows.append([label] + values)
    return table(["model"] + [f"h={h}" for h in HORIZONS], rows)


def interval_text(point, draws, digits=1):
    """A difference in points with its 95 percent interval from bootstrap draws."""
    low, high = stats.interval(draws) * 100
    return f"{point * 100:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})"


def matched_hits(runs, split):
    """Hits on the events that have a visible control, for every model that stored them.

    Returns the episode and run length of each event and, per model, the hits of the
    reappearing colour and of the control colour as arrays of shape (seeds, events).
    """
    episode = k = None
    data = {}
    for name in NAMES:
        pairs = [r["metrics"][split].get("paired") for r in runs.get(name, [])]
        if not pairs or any(p is None for p in pairs):
            continue
        if episode is None:
            episode, k = np.array(pairs[0]["episode"]), np.array(pairs[0]["k"])
        if any(p["episode"] != episode.tolist() or p["k"] != k.tolist() for p in pairs):
            raise SystemExit(f"the runs of {name} list other events than the others")
        data[name] = tuple(
            np.array([p[key] for p in pairs], dtype=float)
            for key in ("target", "control")
        )
    return episode, k, data


def paired(runs, split):
    """The reappearing colour and the visible control scored on the same events.

    The gap is control minus reappearing colour. The interval comes from resampling
    test episodes, with the seeds of a model averaged first; the spread after the
    plus-minus sign is the standard deviation over training seeds.
    """
    episode, k, data = matched_hits(runs, split)
    if not data:
        return None
    names = list(data)
    mean = {n: (t.mean(axis=0), c.mean(axis=0)) for n, (t, c) in data.items()}
    columns = np.stack([v for n in names for v in mean[n]], axis=1)
    draws = stats.episode_bootstrap(episode, columns)
    target = {n: (mean[n][0].mean(), draws[:, 2 * i]) for i, n in enumerate(names)}
    control = {n: (mean[n][1].mean(), draws[:, 2 * i + 1]) for i, n in enumerate(names)}
    gap = {
        n: (control[n][0] - target[n][0], control[n][1] - target[n][1]) for n in names
    }

    rows = []
    for n in names:
        seeds = [data[n][0].mean(axis=1), data[n][1].mean(axis=1)]
        rows.append(
            [NAMES[n]]
            + [cell(s, 100.0, 0) for s in seeds]
            + [cell(seeds[1] - seeds[0], 100.0, 1), interval_text(*gap[n])]
        )
    first = table(
        ["model", "reappearing colour (%)", "visible control (%)", "gap (points)"]
        + ["gap, 95% interval"],
        rows,
    )

    versus = []
    for n in names:
        if n != REFERENCE and REFERENCE in data:
            parts = [
                [t[n][0] - t[REFERENCE][0], t[n][1] - t[REFERENCE][1]]
                for t in (target, control, gap)
            ]
            versus.append([NAMES[n]] + [interval_text(*p) for p in parts])
    head = ["model", "reappearing colour", "visible control", "gap"]
    second = table(head, versus) if versus else None

    by_k = [["events"] + [int((k == j).sum()) for j in range(1, 7)] + [len(k)]]
    for n in names:
        diff = data[n][1] - data[n][0]
        cells = [
            cell(diff[:, k == j].mean(axis=1), 100.0, 1) if (k == j).any() else "n/a"
            for j in range(1, 7)
        ]
        by_k.append([NAMES[n]] + cells + [cell(diff.mean(axis=1), 100.0, 1)])
    third = table(["model"] + [f"k={j}" for j in range(1, 7)] + ["all"], by_k)
    return first, second, third


def training(runs):
    rows = []
    for name, label in NAMES.items():
        if name in runs and "train" in runs[name][0]:
            logs = [r["train"] for r in runs[name]]
            rows.append(
                [
                    label,
                    logs[0]["params"],
                    logs[0]["steps"],
                    cell([g["seconds"] for g in logs], 1.0, 0),
                    cell([g["loss"][-1] for g in logs], 1000.0, 2),
                    len(logs),
                ]
            )
    header = ["model", "params", "steps", "seconds", "final loss x1000", "seeds"]
    return table(header, rows)


def penalty(out):
    """Validation scores of the mass penalty weights, with model B as weight 0."""
    path = out / "lambda.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    rows = [
        [
            lam,
            f"{row['val_drift_abs'] * 100:.2f}",
            f"{row['val_share'] * 100:.0f}",
            f"{row['val_mse_16'] * 1000:.2f}",
            "yes" if float(lam) == info["chosen"] else "",
        ]
        for lam, row in info["candidates"].items()
    ]
    plain = out / "runs" / "b_s0.json"
    if plain.exists():
        val = json.loads(plain.read_text())["metrics"]["val"]
        drift = np.mean([val["drift_abs"][h] for h in ("4", "8", "16")])
        share = val["reappearance"]["all"]["share"]
        mse = val["mse"]["16"]
        rows.insert(
            0,
            [
                "0 (B)",
                f"{drift * 100:.2f}",
                f"{share * 100:.0f}",
                f"{mse * 1000:.2f}",
                "",
            ],
        )
    header = [
        "weight",
        "val drift (%)",
        "val share < 2 px (%)",
        "val mse@16 x1000",
        "chosen",
    ]
    return table(header, rows)


def pooled(result):
    """Share within 2 px, median error, drift and mse at h=16, over all lengths.

    A list of results, one per sampling seed, is averaged.
    """
    results = result if isinstance(result, list) else [result]
    values = np.array(
        [
            [
                r["reappearance"]["all"]["share"] * 100,
                r["reappearance"]["all"]["median"],
                r["drift_abs"]["16"] * 100,
                r["mse"]["16"] * 1000,
            ]
            for r in results
        ]
    ).mean(axis=0)
    return [
        f"{values[0]:.0f}",
        f"{values[1]:.2f}",
        f"{values[2]:.1f}",
        f"{values[3]:.2f}",
    ]


def precision_pairs(info):
    """int8 against fp32 on the same events and the same noise, per model."""
    rows = []
    for kind, row in info["models"].items():
        found = row.get("precision")
        if found is None:
            return None
        fp32 = np.array(found["hits"]["fp32"], dtype=float)
        int8 = np.array(found["hits"]["int8"], dtype=float)
        episode = np.array(found["event"]["episode"])
        columns = np.stack([fp32.mean(axis=0), int8.mean(axis=0)], axis=1)
        draws = stats.episode_bootstrap(episode, columns)
        lost = ((fp32 == 1) & (int8 == 0)).sum(axis=1).mean()
        gained = ((fp32 == 0) & (int8 == 1)).sum(axis=1).mean()
        rows.append(
            [
                NAMES[kind],
                row["sub_episodes"],
                fp32.shape[1],
                len(found["seeds"]),
                f"{fp32.mean() * 100:.1f}",
                f"{int8.mean() * 100:.1f}",
                interval_text(int8.mean() - fp32.mean(), draws[:, 1] - draws[:, 0]),
                f"{lost:.1f}",
                f"{gained:.1f}",
            ]
        )
    head = ["model", "episodes", "events", "noise seeds", "fp32 share (%)"]
    head += ["int8 share (%)", "int8 minus fp32 (points)", "lost", "gained"]
    return table(head, rows)


def frugal(out):
    """Precision and pruning tables from results/frugal.json."""
    path = out / "frugal.json"
    if not path.exists():
        return None
    info = json.loads(path.read_text())
    score = ["share < 2 px (%)", "median error (px)", "drift@16 (%)", "mse@16 x1000"]
    precision, sparse = [], []
    for kind, row in info["models"].items():
        calls = 8 if kind == "d" else 1
        for variant in ("fp32", "int8"):
            part = row[variant]
            ms = part["latency"]["median_ms"]
            lead = [NAMES[kind], variant, part["mb"], f"{ms:.2f}", f"{ms * calls:.1f}"]
            scored = row["precision"][variant] if "precision" in row else part["sub"]
            precision.append(lead + [row["sub_episodes"]] + pooled(scored))
        first = row["fp32"]
        ms = first["latency"]["median_ms"]
        sparse.append([NAMES[kind], 0, "none", f"{ms:.2f}"] + pooled(first["test"]))
        for part in row["pruned"].values():
            ms = part["latency"]["median_ms"]
            lead = [
                NAMES[kind],
                f"{part['sparsity'] * 100:.0f}",
                info["finetune_steps"],
            ]
            sparse.append(lead + [f"{ms:.2f}"] + pooled(part["test"]))
    head = ["model", "weights", "size (MB)", "ms per call", "ms per frame", "episodes"]
    return (
        table(head + score, precision),
        table(
            ["model", "sparsity (%)", "finetune steps", "ms per call"] + score, sparse
        ),
        precision_pairs(info),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    out = pathlib.Path(p.parse_args().out)
    runs = load_runs(out)
    for split in ("test", "hard"):
        print(f"\n### Reappearance within 2 px (%), {split}\n")
        print(reappearance(runs, split, "share", 100.0, 0))
        print(f"\n### Median reappearance error (px), {split}\n")
        print(reappearance(runs, split, "median", 1.0, 1))
        print(f"\n### Visible control within 2 px (%), {split}\n")
        print(reappearance(runs, split, "share", 100.0, 0, key="control"))
        same = paired(runs, split)
        if same:
            print(f"\n### Same events: reappearing colour and control (%), {split}\n")
            print(same[0])
            if same[1]:
                print(f"\n### Same events: difference from model B (points), {split}\n")
                print(same[1])
            print(f"\n### Same events: control minus reappearing colour, {split}\n")
            print(same[2])
        print(f"\n### Mass drift (% of a square, absolute), {split}\n")
        print(horizons(runs, split, "drift_abs", 100.0, 1))
        print(f"\n### Mass drift (% of a square, signed), {split}\n")
        print(horizons(runs, split, "drift", 100.0, 1))
        print(f"\n### Pixel MSE x1000, {split}\n")
        print(horizons(runs, split, "mse", 1000.0, 2))
    print("\n### Training\n")
    print(training(runs))
    lam = penalty(out)
    if lam:
        print("\n### Mass penalty weight (validation)\n")
        print(lam)
    tables = frugal(out)
    if tables:
        print("\n### Frugal inference: precision\n")
        print(tables[0])
        print("\n### Frugal inference: pruning, full test set\n")
        print(tables[1])
        if tables[2]:
            print("\n### Frugal inference: int8 against fp32, same events and noise\n")
            print(tables[2])


if __name__ == "__main__":
    main()
