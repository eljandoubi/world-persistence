import importlib.util
import json
import pathlib
import sys

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


HEADING = "### Frugal inference: int8 against fp32, same events and noise"


def print_report(monkeypatch, capsys, tmp_path, frugal_tables):
    """Run report.main() with every table builder but frugal replaced by a stand-in."""
    for name in ("reappearance", "horizons", "training"):
        monkeypatch.setattr(report, name, lambda *args, **kwargs: "| STAND-IN |")
    monkeypatch.setattr(report, "load_runs", lambda out: {})
    monkeypatch.setattr(report, "paired", lambda runs, split: None)
    monkeypatch.setattr(report, "penalty", lambda out: None)
    monkeypatch.setattr(report, "frugal", lambda out: frugal_tables)
    monkeypatch.setattr(sys, "argv", ["report.py", "--out", str(tmp_path)])
    report.main()
    return capsys.readouterr().out


def test_the_int8_against_fp32_table_is_printed_once_under_its_heading(
    monkeypatch, capsys, tmp_path
):
    tables = ("| PRECISION |", "| PRUNING |", "| PAIRED_PRECISION_TABLE |")
    printed = print_report(monkeypatch, capsys, tmp_path, tables)
    assert printed.count(HEADING) == 1
    assert printed.count("| PAIRED_PRECISION_TABLE |") == 1
    assert printed.index(HEADING) < printed.index("| PAIRED_PRECISION_TABLE |")


def test_no_int8_against_fp32_heading_without_a_table(monkeypatch, capsys, tmp_path):
    tables = ("| PRECISION |", "| PRUNING |", None)
    printed = print_report(monkeypatch, capsys, tmp_path, tables)
    assert "| PRUNING |" in printed and HEADING not in printed


def scores(share):
    """A persistence summary with the fields the frugal tables read."""
    return {
        "reappearance": {"all": {"share": share, "median": 3.0}},
        "drift_abs": {"16": 0.25},
        "mse": {"16": 0.02},
    }


def frugal_info(matched):
    """A frugal.json for one model; `matched` is the block frugal.py writes."""
    part = {"mb": 4.8, "latency": {"median_ms": 4.5}, "sub": scores(0.3)}
    row = {
        "sub_episodes": 3,
        "fp32": dict(part, test=scores(0.5)),
        "int8": dict(part),
        "pruned": {},
    }
    if matched:
        row["precision"] = matched
    return {"finetune_steps": 600, "models": {"c": row}}


def matched_block(fp32, int8):
    """Two noise seeds, six events in three episodes, as frugal.py writes them."""
    return {
        "seeds": [0, 1],
        "fp32": [scores(0.5)] * 2,
        "int8": [scores(0.4)] * 2,
        "hits": {"fp32": fp32, "int8": int8},
        "event": {"episode": [0, 0, 1, 1, 2, 2], "k": [1, 2, 1, 1, 3, 1]},
    }


def cells(row):
    return [c.strip() for c in row.strip("|").split("|")]


def test_int8_against_fp32_counts_the_events_lost_and_gained():
    fp32 = [[1, 1, 0, 0, 1, 0]] * 2
    int8 = [[1, 0, 0, 0, 1, 0], [1, 0, 0, 1, 1, 0]]
    info = frugal_info(matched_block(fp32, int8))
    row = cells(report.precision_pairs(info).splitlines()[2])
    assert row[:5] == ["C unrolled", "3", "6", "2", "50.0"]
    assert row[5] == "41.7" and row[6].startswith("-8.3 (")
    assert row[7:] == ["1.0", "0.5"]  # events lost and gained, mean over noise seeds


def test_frugal_tables_carry_the_matched_comparison_when_the_results_have_it(
    tmp_path,
):
    block = matched_block([[1, 1, 0, 0, 1, 0]] * 2, [[1, 1, 0, 0, 1, 0]] * 2)
    (tmp_path / "frugal.json").write_text(json.dumps(frugal_info(block)))
    precision, _, same = report.frugal(tmp_path)
    assert cells(precision.splitlines()[3])[6] == "40"  # matched int8 score
    assert cells(same.splitlines()[2])[:4] == ["C unrolled", "3", "6", "2"]


def test_older_frugal_results_have_no_matched_comparison(tmp_path):
    (tmp_path / "frugal.json").write_text(json.dumps(frugal_info(None)))
    precision, _, same = report.frugal(tmp_path)
    assert cells(precision.splitlines()[3])[6] == "30"  # the earlier int8 score
    assert same is None


def write_run(path, val_share, test_share):
    scored = {"drift_abs": {h: 0.2 for h in ("4", "8", "16")}, "mse": {"16": 0.027}}
    metrics = {
        split: dict(scored, reappearance={"all": {"share": share}})
        for split, share in (("val", val_share), ("test", test_share))
    }
    path.write_text(json.dumps({"metrics": metrics}))


def test_penalty_table_marks_the_first_grid_and_shows_the_test_share(tmp_path):
    (tmp_path / "runs").mkdir()
    row = {"val_drift_abs": 0.2, "val_share": 0.24, "val_mse_16": 0.027}
    info = {"candidates": {"0.001": row, "10": row}, "chosen": 0.001}
    (tmp_path / "lambda.json").write_text(json.dumps(info))
    write_run(tmp_path / "runs" / "b_s0.json", 0.26, 0.2617)
    write_run(tmp_path / "runs" / "e_lam0.001_s0.json", 0.24, 0.2569)
    rows = [cells(line) for line in report.penalty(tmp_path).splitlines()[2:]]
    assert [r[:2] for r in rows] == [["0 (B)", ""], ["0.001", "added"], ["10", "first"]]
    assert [r[5] for r in rows] == ["26.2", "25.7", "n/a"]  # no run file for weight 10
    assert [r[6] for r in rows] == ["", "yes", ""]
