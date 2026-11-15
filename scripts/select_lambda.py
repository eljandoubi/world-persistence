"""Pick the mass penalty weight of model e on the validation set (seed 0 runs)."""

import argparse
import json
import pathlib
import shutil

HORIZONS = ("4", "8", "16")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default="work/ckpt")
    args = p.parse_args()
    out, ckpt = pathlib.Path(args.out), pathlib.Path(args.ckpt)

    candidates = {}
    for path in sorted((out / "runs").glob("e_lam*_s0.json")):
        run = json.loads(path.read_text())
        val = run["metrics"]["val"]
        drift = sum(val["drift_abs"][h] for h in HORIZONS) / len(HORIZONS)
        candidates[run["train"]["lam"]] = {
            "val_drift_abs": round(drift, 6),
            "val_share": val["reappearance"]["all"]["share"],
            "val_mse_16": val["mse"]["16"],
        }
    chosen = min(candidates, key=lambda lam: candidates[lam]["val_drift_abs"])
    weights = ckpt / f"e_lam{chosen:g}_s0.pt"
    if not weights.exists():
        raise SystemExit(f"missing weights {weights}: train model e first")
    summary = {
        "criterion": "mean absolute mass drift on validation at horizons 4, 8 and 16",
        "candidates": {f"{lam:g}": row for lam, row in sorted(candidates.items())},
        "chosen": chosen,
    }
    (out / "lambda.json").write_text(json.dumps(summary, indent=1))

    run = json.loads((out / "runs" / f"e_lam{chosen:g}_s0.json").read_text())
    run["name"] = "e"
    (out / "runs" / "e_s0.json").write_text(json.dumps(run, indent=1))
    shutil.copy(weights, ckpt / "e_s0.pt")
    print("chosen lambda", chosen)


if __name__ == "__main__":
    main()
