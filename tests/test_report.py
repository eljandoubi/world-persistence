import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report", ROOT / "scripts" / "report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)

EPISODE = np.repeat(np.arange(30), 4).tolist()
K = (np.arange(120) % 6 + 1).tolist()


def run(target, control, episode=EPISODE):
    paired = {"episode": episode, "k": K, "target": target, "control": control}
    return {"metrics": {"test": {"paired": paired}}}


def hits(rate, seed):
    return (np.random.default_rng(seed).random(120) < rate).astype(int).tolist()


def test_the_gap_is_control_minus_reappearing_colour_on_the_same_events():
    target, control = hits(0.3, 1), hits(0.7, 2)
    runs = {"b": [run(target, control)] * 2}
    first, second, third = report.paired(runs, "test")
    gap = (np.mean(control) - np.mean(target)) * 100
    row = first.splitlines()[2]
    assert row.startswith("| B history |") and f"{gap:.1f} ± 0.0" in row
    assert "to" in row.split("|")[-2]  # the interval from resampling episodes
    assert third.splitlines()[3].count("±") == 7  # six run lengths and the total
    assert second is None  # only the reference model: nothing to compare it with


def test_differences_from_the_reference_model_come_with_intervals():
    runs = {
        "b": [run(hits(0.3, 1), hits(0.6, 2))],
        "c": [run(hits(0.5, 3), hits(0.8, 4))],
    }
    _, second, _ = report.paired(runs, "test")
    row = second.splitlines()[2]
    assert row.startswith("| C unrolled |") and row.count("(") == 3
    assert row.count("to") == 3


def test_runs_that_list_different_events_are_refused():
    other = EPISODE[:-1] + [99]
    runs = {
        "b": [run(hits(0.3, 1), hits(0.6, 2)), run(hits(0.3, 3), hits(0.6, 4), other)]
    }
    with pytest.raises(SystemExit):
        report.paired(runs, "test")


def test_models_without_stored_events_are_left_out():
    stored = run(hits(0.3, 1), hits(0.6, 2))
    old = {"metrics": {"test": {"reappearance": {}}}}
    first, _, _ = report.paired({"b": [stored], "c": [old]}, "test")
    assert "C unrolled" not in first and "B history" in first
    assert report.paired({"c": [old]}, "test") is None
