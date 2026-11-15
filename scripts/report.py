"""Print the README tables from results/ (mean and standard deviation over seeds)."""

import argparse
import json
import pathlib

import numpy as np

NAMES = {
    "a": "A memoryless",
    "b": "B history",
    "c": "C unrolled",
    "d": "D flow",
    "e": "E mass penalty",
}
LENGTHS = ["1", "2", "3", "4", "5", "6", "all"]
HORIZONS = ["1", "4", "8", "16"]


def load_runs(out):
    runs = {}
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


def training(runs):
    rows = []
    for name, label in NAMES.items():
        if name in runs:
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
    header = [
        "weight",
        "val drift (%)",
        "val share < 2 px (%)",
        "val mse@16 x1000",
        "chosen",
    ]
    return table(header, rows)


def pooled(result):
    """Share within 2 px, median error, drift and mse at h=16, over all lengths."""
    total = result["reappearance"]["all"]
    return [
        f"{total['share'] * 100:.0f}",
        f"{total['median']:.2f}",
        f"{result['drift_abs']['16'] * 100:.1f}",
        f"{result['mse']['16'] * 1000:.2f}",
    ]


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
            precision.append(lead + [row["sub_episodes"]] + pooled(part["sub"]))
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


if __name__ == "__main__":
    main()
