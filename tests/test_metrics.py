import numpy as np
import torch

from worldpersist import metrics, sim


def one_frame(pos, bar=24):
    return torch.as_tensor(sim.render(np.array([pos], dtype=float), np.array([bar])))[0]


def test_masks_pick_pure_colours_and_ignore_grey():
    frame = one_frame([[2.0, 3.0], [15.0, 20.0], [12.5, 8.0]])
    mask = metrics.colour_masks(frame)
    assert mask[0, 3:8, 2:7].min() == 1.0
    assert mask[0, :, 24:].max() == 0.0 and mask[1, :, 24:].max() == 0.0
    assert mask[0, 20:25, 15:20].max() == 0.0
    assert torch.allclose(metrics.masses(frame), torch.full((3,), 25.0), atol=1.0)


def test_centroid_reads_the_true_centre():
    for x, y in [(3.0, 4.0), (3.3, 9.7), (12.5, 20.2), (17.9, 1.1)]:
        frame = one_frame([[x, y], [0.0, 27.0], [8.0, 27.0]])
        centre, mass = metrics.centroids(frame)
        assert torch.allclose(centre[0], torch.tensor([x + 2.5, y + 2.5]), atol=0.1)
        assert mass[0] > 23


def test_noise_below_the_dead_zone_does_not_count():
    frame = torch.rand(3, 32, 32) * 0.05
    assert metrics.masses(frame).max() == 0.0


def test_events_on_hand_built_runs():
    covered = np.zeros((1, 20, 3))
    covered[0, 6:9, 0] = 1.0  # k = 3, one partial frame, then visible at 10
    covered[0, 9, 0] = 0.4
    covered[0, 2:4, 1] = 1.0  # k = 2, visible again at frame 4
    covered[0, 1:3, 2] = 1.0  # reappears at frame 3, inside the context
    events = metrics.occlusion_events(covered)
    assert events.tolist() == [[0, 0, 3, 10], [0, 1, 2, 4]]


def test_run_that_hides_again_before_it_is_visible_is_skipped():
    covered = np.zeros((1, 20, 3))
    covered[0, 5:7, 0] = 1.0
    covered[0, 7, 0] = 0.5
    covered[0, 8, 0] = 1.0
    assert metrics.occlusion_events(covered).tolist() == [[0, 0, 1, 9]]


def controls_of(covered):
    """Positions of the events that have a control, and the control rows."""
    keep, rows = metrics.control_events(covered, metrics.occlusion_events(covered))
    return keep.tolist(), rows.tolist()


def test_control_events_pair_each_event_with_a_visible_colour():
    covered = np.zeros((1, 20, 3))
    covered[0, 6:9, 0] = 1.0
    covered[0, 9, 0] = 0.4
    assert controls_of(covered) == ([0], [[0, 1, 3, 10]])
    covered[0, 8, 1] = 0.5  # colour 1 is covered inside the window of frames 3 to 10
    assert controls_of(covered) == ([0], [[0, 2, 3, 10]])
    covered[0, 2, 2] = 0.5  # before the window: colour 2 still counts
    assert controls_of(covered) == ([0], [[0, 2, 3, 10]])
    covered[0, 5, 2] = 0.5
    assert controls_of(covered) == ([], [])


def test_control_events_keep_the_index_of_the_event():
    covered = np.zeros((2, 20, 3))
    covered[:, 6:8, 0] = 1.0  # colour 0 hides in both episodes: events 0 and 2
    covered[0, 12:14, 1] = 1.0  # colour 1 hides in episode 0: event 1
    covered[0, 9, 2] = (
        0.5  # with colour 0 hidden at 7, nothing is visible around event 1
    )
    events = metrics.occlusion_events(covered)
    assert events[:, :2].tolist() == [[0, 0], [0, 1], [1, 0]]
    keep, rows = metrics.control_events(covered, events)
    assert keep.tolist() == [0, 2]
    assert rows[:, [0, 2, 3]].tolist() == events[keep][:, [0, 2, 3]].tolist()
    assert (rows[:, 1] != events[keep, 1]).all()


def test_events_exist_in_the_simulator():
    ep = sim.simulate(200, seed=1)
    events = metrics.occlusion_events(ep.covered)
    assert len(events) > 50
    assert events[:, 3].min() >= sim.CONTEXT and events[:, 2].min() >= 1


def test_ground_truth_scores_zero():
    ep = sim.simulate(150, seed=2)
    result = metrics.evaluate(torch.as_tensor(ep.frames), ep)
    total = result["reappearance"]["all"]
    assert total["n"] > 50
    assert total["median"] < 0.1 and total["share"] == 1.0 and total["miss"] == 0.0
    control = result["control"]["all"]
    assert 20 < control["n"] <= total["n"] and control["share"] == 1.0
    assert all(v == 0.0 for v in result["mse"].values())
    assert all(v < 1e-6 for v in result["drift_abs"].values())


def test_matched_scores_cover_the_events_that_have_a_control():
    ep = sim.simulate(150, seed=2)
    events = metrics.occlusion_events(ep.covered)
    keep, _ = metrics.control_events(ep.covered, events)
    lost = np.setdiff1d(np.arange(len(events)), keep)
    assert len(keep) > 20 and len(lost) > 20  # the simulator makes both kinds of event
    pred = torch.as_tensor(ep.frames).clone()
    for e, c, _, r in events[lost]:
        pred[e, r, c] = pred[e, r].amin(dim=0)  # drop the colour, keep the others
    result = metrics.evaluate(pred, ep)
    total, matched, control = (
        result[key]["all"] for key in ("reappearance", "matched", "control")
    )
    assert total["share"] < 0.9
    assert matched["share"] == control["share"] == 1.0
    assert matched["n"] == control["n"] < total["n"]


def test_paired_hits_agree_with_the_summaries():
    ep = sim.simulate(150, seed=2)
    pred = torch.as_tensor(ep.frames).clone()
    pred[:, sim.CONTEXT :] = pred[:, sim.CONTEXT - 1 : sim.CONTEXT]
    result = metrics.evaluate(pred, ep)
    pairs, n = result["paired"], result["matched"]["all"]["n"]
    assert {len(v) for v in pairs.values()} == {n}
    assert set(pairs["k"]) <= set(metrics.LENGTHS)
    assert round(sum(pairs["target"]) / n, 4) == result["matched"]["all"]["share"]
    assert round(sum(pairs["control"]) / n, 4) == result["control"]["all"]["share"]
    assert 0 < sum(pairs["target"]) < n and 0 < sum(pairs["control"]) < n


def test_shifted_copy_is_penalised():
    ep = sim.simulate(150, seed=2)
    shifted = torch.roll(torch.as_tensor(ep.frames), 3, dims=-1)
    result = metrics.evaluate(shifted, ep)
    total = result["reappearance"]["all"]
    assert abs(total["median"] - 3.0) < 0.1 and total["share"] < 0.05
    assert result["control"]["all"]["share"] < 0.05
    assert all(v > 0 for v in result["mse"].values())


def test_empty_prediction_is_all_misses():
    ep = sim.simulate(100, seed=3)
    result = metrics.evaluate(torch.zeros(ep.frames.shape), ep)
    total = result["reappearance"]["all"]
    assert total["miss"] == 1.0 and total["median"] == metrics.MISS
    assert result["drift"]["16"] < 0 < result["drift_abs"]["16"]


def test_dimmed_copy_loses_mass():
    ep = sim.simulate(100, seed=3)
    result = metrics.evaluate(0.5 * torch.as_tensor(ep.frames), ep)
    assert all(v < 0 for v in result["drift"].values())
    assert result["reappearance"]["all"]["share"] == 1.0
