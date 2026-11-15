"""Frugal inference study: ONNX size and latency, int8 quantisation, pruning."""

import argparse
import copy
import json
import pathlib

import numpy as np
import torch

from worldpersist import frugal, metrics, models, sim, training

SPARSITY = (0.0, 0.3, 0.5, 0.7)
CPU = torch.device("cpu")


def best_regressor(runs):
    """Regression model (a, b, c or e) with the best validation share within 2 px."""
    best, best_share = None, -1.0
    for name in ("a", "b", "c", "e"):
        files = sorted(runs.glob(f"{name}_s*.json"))
        shares = [json.loads(f.read_text())["metrics"]["val"] for f in files]
        shares = [v["reappearance"]["all"]["share"] for v in shares]
        if shares and np.mean(shares) > best_share:
            best, best_share = name, float(np.mean(shares))
    return best


def score_onnx(path, like, episodes, batch=50):
    """Persistence metrics of an ONNX network rolled out on the CPU."""
    predictor = frugal.OnnxPredictor(path, like, threads=2)
    return metrics.evaluate(
        training.predict_episodes(predictor, episodes, CPU, batch), episodes
    )


def study(kind, args, device, data, work, lam):
    """Size, latency and persistence of one model as fp32, int8 and pruned."""
    model = training.load_model(
        pathlib.Path(args.ckpt) / f"{kind}_s{args.seed}.pt", device
    )
    cpu = copy.deepcopy(model).cpu()
    sub = data["test"].head(args.sub_flow if model.flow else args.sub)
    row = {"params": models.count_params(model), "sub_episodes": len(sub.frames)}

    fp32 = work / f"{kind}_fp32.onnx"
    frugal.export_onnx(cpu, fp32)
    row["fp32"] = {
        "mb": frugal.size_mb(fp32),
        "latency": frugal.latency(fp32, cpu, args.runs),
        "test": training.evaluate_model(model, data["test"], device, amp=False),
        "sub": training.evaluate_model(model, sub, device, amp=False),
    }
    print(kind, "fp32", row["fp32"]["latency"]["median_ms"], "ms", flush=True)

    int8 = work / f"{kind}_int8.onnx"
    frugal.quantize(fp32, int8, frugal.calibration_inputs(cpu, data["val"], 200))
    row["int8"] = {
        "mb": frugal.size_mb(int8),
        "latency": frugal.latency(int8, cpu, args.runs),
        "sub": score_onnx(int8, cpu, sub),
    }
    print(kind, "int8", row["int8"]["latency"]["median_ms"], "ms", flush=True)

    row["pruned"] = {}
    for amount in SPARSITY:
        pruned = copy.deepcopy(model)
        if amount:
            frugal.prune_(pruned, amount)
        pruned, log = training.train(
            kind,
            data["train"],
            args.ft_steps,
            args.seed,
            device,
            lr=args.ft_lr,
            lam=lam,
            model=pruned,
        )
        frugal.make_permanent(pruned)
        path = work / f"{kind}_p{amount:g}.onnx"
        frugal.export_onnx(copy.deepcopy(pruned).cpu(), path)
        row["pruned"][f"{amount:g}"] = {
            "sparsity": round(frugal.sparsity(pruned), 4),
            "latency": frugal.latency(path, cpu, args.runs),
            "test": training.evaluate_model(pruned, data["test"], device, amp=False),
            "finetune_seconds": log["seconds"],
        }
        print(
            kind, "pruned", amount, row["pruned"][f"{amount:g}"]["sparsity"], flush=True
        )
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regressor", default="auto", help="a, b, c, e or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sub", type=int, default=200, help="test episodes for int8, regressor"
    )
    p.add_argument(
        "--sub-flow", type=int, default=100, help="test episodes for int8, flow"
    )
    p.add_argument("--ft-steps", type=int, default=600)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--runs", type=int, default=200)
    p.add_argument("--train-episodes", type=int)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--out", default="results")
    p.add_argument("--ckpt", default="work/ckpt")
    p.add_argument("--work", default="work/onnx")
    args = p.parse_args()

    out, work = pathlib.Path(args.out), pathlib.Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regressor = (
        best_regressor(out / "runs") if args.regressor == "auto" else args.regressor
    )
    lam_file = out / "lambda.json"
    lam = json.loads(lam_file.read_text())["chosen"] if lam_file.exists() else 0.0
    data = {
        "train": sim.load_split("train", args.train_episodes),
        "val": sim.load_split("val", args.eval_episodes),
        "test": sim.load_split("test", args.eval_episodes),
    }
    result = {
        "seed": args.seed,
        "regressor": regressor,
        "finetune_steps": args.ft_steps,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "models": {},
    }
    for kind in (regressor, "d"):
        result["models"][kind] = study(
            kind, args, device, data, work, lam if kind == "e" else 0.0
        )
    (out / "frugal.json").write_text(json.dumps(result, indent=1))
    print(out / "frugal.json")


if __name__ == "__main__":
    main()
