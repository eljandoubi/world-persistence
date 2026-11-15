"""Score the trivial predictor that repeats the last context frame."""

import argparse
import json
import pathlib

import torch

from worldpersist import metrics, sim


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results")
    out = pathlib.Path(p.parse_args().out)
    result = {}
    for split in ("val", "test", "hard"):
        episodes = sim.load_split(split)
        frames = torch.as_tensor(episodes.frames).clone()
        frames[:, sim.CONTEXT :] = frames[:, sim.CONTEXT - 1 : sim.CONTEXT]
        result[split] = metrics.evaluate(frames, episodes)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline.json").write_text(json.dumps(result, indent=1))
    print(out / "baseline.json")


if __name__ == "__main__":
    main()
