"""Train predictors and score them; one JSON per run goes to results/runs/."""

import argparse
import json
import pathlib

import torch

from worldpersist import sim, training


def run(kind, name, lam, seed, args, train_set, scored, device):
    """Train one model, save its weights and score it on every split."""
    model, log = training.train(
        kind,
        train_set,
        args.steps,
        seed,
        device,
        batch=args.batch,
        lr=args.lr,
        lam=lam,
        base=args.base,
    )
    weights = pathlib.Path(args.ckpt) / f"{name}_s{seed}.pt"
    training.save_model(model, kind, weights, args.base)
    metrics = {s: training.evaluate_model(model, e, device) for s, e in scored.items()}
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return {
        "name": name,
        "model": kind,
        "seed": seed,
        "gpu": gpu,
        "train": log,
        "metrics": metrics,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["a", "b", "c", "d", "e"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--lams", nargs="+", type=float, default=[1.0])
    p.add_argument("--name", help="run name; default is the model letter")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base", type=int, default=32)
    p.add_argument("--train-episodes", type=int)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default="work/ckpt")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    runs = pathlib.Path(args.out) / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    train_set = sim.load_split("train", args.train_episodes)
    scored = {s: sim.load_split(s, args.eval_episodes) for s in ("val", "test", "hard")}

    for kind in args.models:
        for lam in args.lams if kind == "e" else [0.0]:
            name = args.name or (f"e_lam{lam:g}" if kind == "e" else kind)
            for seed in args.seeds:
                path = runs / f"{name}_s{seed}.json"
                if path.exists():
                    print("skip", path, flush=True)
                    continue
                result = run(kind, name, lam, seed, args, train_set, scored, device)
                path.write_text(json.dumps(result, indent=1))
                total = result["metrics"]["test"]["reappearance"]["all"]
                seconds = result["train"]["seconds"]
                print(path, f"{seconds}s", "test share", total["share"], flush=True)


if __name__ == "__main__":
    main()
