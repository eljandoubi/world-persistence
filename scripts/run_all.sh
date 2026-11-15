#!/bin/bash
# Full study on one GPU. Stages: baseline, train, frugal, figures (default: all of them).
# STEPS sets the number of optimiser steps per model (default 3000). OUT and CKPT set
# the folders for results and weights (defaults results and work/ckpt). PYTHON sets the
# interpreter (default python3). A model run is skipped only when its result file and
# its weights both exist and match the settings.
# The extra stage rescore re-scores saved weights after a change to the metrics.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
PYTHON=${PYTHON:-python3}
STEPS=${STEPS:-3000}
OUT=${OUT:-results}
CKPT=${CKPT:-work/ckpt}
LAMS="0.001 0.01 0.1 1 10"
dirs=(--out "$OUT" --ckpt "$CKPT")

baseline() {
    "$PYTHON" scripts/baseline.py --out "$OUT"
}

train() {
    "$PYTHON" scripts/train.py --models a b c d --seeds 0 1 2 --steps "$STEPS" "${dirs[@]}"
    "$PYTHON" scripts/train.py --models e --lams $LAMS --seeds 0 --steps "$STEPS" "${dirs[@]}"
    "$PYTHON" scripts/select_lambda.py "${dirs[@]}"
    lam=$("$PYTHON" -c "import json; print(json.load(open('$OUT/lambda.json'))['chosen'])")
    "$PYTHON" scripts/train.py --models e --lams "$lam" --name e --seeds 1 2 \
        --steps "$STEPS" "${dirs[@]}"
}

rescore() {
    "$PYTHON" scripts/rescore.py "${dirs[@]}"
}

frugal() {
    "$PYTHON" scripts/frugal.py "${dirs[@]}"
}

figures() {
    "$PYTHON" scripts/figures.py strip "${dirs[@]}"
    "$PYTHON" scripts/figures.py curve "${dirs[@]}"
}

for stage in "${@:-baseline train frugal figures}"; do
    for one in $stage; do
        "$one"
    done
done
