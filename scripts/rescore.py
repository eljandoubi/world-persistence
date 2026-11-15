"""Score saved weights again and rewrite the metrics of every run, keeping the logs."""

import argparse
import json
import pathlib

import torch

from worldpersist import sim, training


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default="work/ckpt")
    p.add_argument("--eval-episodes", type=int)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored = {s: sim.load_split(s, args.eval_episodes) for s in ("val", "test", "hard")}
    for path in sorted((pathlib.Path(args.out) / "runs").glob("*_s*.json")):
        run = json.loads(path.read_text())
        weights = pathlib.Path(args.ckpt) / f"{run['name']}_s{run['seed']}.pt"
        model = training.load_model(weights, device)
        run["metrics"] = {
            s: training.evaluate_model(model, e, device) for s, e in scored.items()
        }
        path.write_text(json.dumps(run, indent=1))
        print(path, flush=True)


if __name__ == "__main__":
    main()
