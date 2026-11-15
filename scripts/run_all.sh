#!/bin/bash
# Full study on one GPU. Stages: baseline, train, frugal, figures (default: all of them).
# STEPS sets the number of optimiser steps per model (default 3000). OUT and CKPT set
# the folders for results and weights (defaults results and work/ckpt). A model run is
# skipped only when its result file and its weights both exist and match the settings.
# The extra stage rescore re-scores saved weights after a change to the metrics.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
STEPS=${STEPS:-3000}
OUT=${OUT:-results}
CKPT=${CKPT:-work/ckpt}
LAMS="0.001 0.01 0.1 1 10"
dirs=(--out "$OUT" --ckpt "$CKPT")

baseline() {
    python scripts/baseline.py --out "$OUT"
}

train() {
    python scripts/train.py --models a b c d --seeds 0 1 2 --steps "$STEPS" "${dirs[@]}"
    python scripts/train.py --models e --lams $LAMS --seeds 0 --steps "$STEPS" "${dirs[@]}"
    python scripts/select_lambda.py "${dirs[@]}"
    lam=$(python -c "import json; print(json.load(open('$OUT/lambda.json'))['chosen'])")
    python scripts/train.py --models e --lams "$lam" --name e --seeds 1 2 \
        --steps "$STEPS" "${dirs[@]}"
}

rescore() {
    python scripts/rescore.py "${dirs[@]}"
}

frugal() {
    python scripts/frugal.py "${dirs[@]}"
}

figures() {
    python scripts/figures.py strip "${dirs[@]}"
    python scripts/figures.py curve "${dirs[@]}"
}

for stage in "${@:-baseline train frugal figures}"; do
    for one in $stage; do
        "$one"
    done
done
