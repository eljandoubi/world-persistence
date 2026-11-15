import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TINY = ["--steps", "2", "--batch", "8", "--base", "8"]
TINY += ["--train-episodes", "16", "--eval-episodes", "12"]
TINY += ["--models", "b", "--seeds", "0"]


def train(tmp_path, *extra):
    """Run scripts/train.py on a tiny setup and return its output lines."""
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py")]
    cmd += ["--out", str(tmp_path / "results"), "--ckpt", str(tmp_path / "ckpt")]
    done = subprocess.run(cmd + TINY + list(extra), capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout.splitlines()


def test_a_result_without_weights_is_trained_again(tmp_path):
    train(tmp_path)
    weights = tmp_path / "ckpt" / "b_s0.pt"
    result = tmp_path / "results" / "runs" / "b_s0.json"
    assert weights.exists() and result.exists()
    weights.unlink()  # a checkout that has the result files but no weights
    lines = train(tmp_path)
    assert weights.exists()
    assert not any(line.startswith("skip") for line in lines)


def test_a_complete_run_is_skipped(tmp_path):
    train(tmp_path)
    result = tmp_path / "results" / "runs" / "b_s0.json"
    before = result.read_text()
    lines = train(tmp_path)
    assert lines and lines[0].startswith("skip")
    assert result.read_text() == before


def test_a_run_with_other_settings_is_trained_again(tmp_path):
    train(tmp_path)
    lines = train(tmp_path, "--steps", "3")
    result = json.loads((tmp_path / "results" / "runs" / "b_s0.json").read_text())
    assert result["train"]["steps"] == 3
    assert not any(line.startswith("skip") for line in lines)


def test_the_run_script_trains_every_weight_of_the_documented_grid():
    script = (ROOT / "scripts" / "run_all.sh").read_text()
    grid = re.search(r'^LAMS="([^"]+)"', script, re.MULTILINE).group(1).split()
    documented = json.loads((ROOT / "results" / "lambda.json").read_text())
    assert [float(w) for w in grid] == sorted(map(float, documented["candidates"]))
    assert "--lams $LAMS" in script


def test_the_run_script_needs_no_python_alias():
    lines = (ROOT / "scripts" / "run_all.sh").read_text().splitlines()
    code = [line for line in lines if not line.lstrip().startswith("#")]
    assert "PYTHON=${PYTHON:-python3}" in code
    calls = [line for line in code if re.search(r"scripts/\w+\.py| -c ", line)]
    assert calls and all('"$PYTHON"' in line for line in calls)
