#!/bin/bash
# Full study on one GPU. Stages: baseline, train, frugal, figures (default: all of them).
# STEPS sets the number of optimiser steps per model (default 3000).
# The extra stage rescore re-scores saved weights after a change to the metrics.
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=src
STEPS=${STEPS:-3000}

baseline() {
    python scripts/baseline.py
}

train() {
    python scripts/train.py --models a b c d --seeds 0 1 2 --steps "$STEPS"
    python scripts/train.py --models e --lams 0.1 1 10 --seeds 0 --steps "$STEPS"
    python scripts/select_lambda.py
    lam=$(python -c "import json; print(json.load(open('results/lambda.json'))['chosen'])")
    python scripts/train.py --models e --lams "$lam" --name e --seeds 1 2 --steps "$STEPS"
}

rescore() {
    python scripts/rescore.py
}

frugal() {
    python scripts/frugal.py
}

figures() {
    python scripts/figures.py strip
    python scripts/figures.py curve
}

for stage in "${@:-baseline train frugal figures}"; do
    for one in $stage; do
        "$one"
    done
done
