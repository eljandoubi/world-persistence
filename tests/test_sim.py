import numpy as np

from worldpersist import sim


def test_simulate_is_deterministic():
    a, b = sim.simulate(8, seed=3), sim.simulate(8, seed=3)
    assert np.array_equal(a.frames, b.frames)
    assert np.array_equal(a.actions, b.actions)
    assert not np.array_equal(a.frames, sim.simulate(8, seed=4).frames)


def test_shapes_and_ranges():
    ep = sim.simulate(6, seed=0)
    assert ep.frames.shape == (6, 20, 3, 32, 32)
    assert ep.frames.dtype == np.float32
    assert ep.frames.min() >= 0 and ep.frames.max() <= 1
    assert set(np.unique(ep.actions)) <= {-2, 0, 2}
    assert ep.bar.min() >= 0 and ep.bar.max() <= sim.BAR_MAX
    assert ep.pos.min() >= 0 and ep.pos.max() <= sim.POS_MAX


def test_hard_split_uses_faster_bar():
    ep = sim.simulate(200, seed=0, hard=True)
    assert set(np.unique(ep.actions)) == {-4, -2, 0, 2, 4}


def test_speed_range_of_the_first_step():
    ep = sim.simulate(400, seed=1)
    step = np.abs(np.diff(ep.pos[:, :2], axis=1))[:, 0]  # (n, 3, 2)
    clear = (ep.pos[:, :2] > 2).all(axis=(1, 3)) & (ep.pos[:, :2] < 25).all(axis=(1, 3))
    step = step[clear]
    assert step.max() <= sim.SPEED + 1e-9
    assert (step.max(axis=-1) > sim.MIN_SPEED).all()


def test_actions_are_piecewise_constant():
    ep = sim.simulate(300, seed=2)
    switches = (np.diff(ep.actions, axis=1) != 0).mean()
    assert 0.05 < switches < sim.SWITCH


def test_bar_follows_actions_and_stays_inside():
    ep = sim.simulate(50, seed=5)
    expected = np.clip(ep.bar[:, :-1] + ep.actions, 0, sim.BAR_MAX)
    assert np.array_equal(ep.bar[:, 1:], expected)


def test_wall_bounce():
    pos, vel = sim.advance(np.array([[0.5, 10.0]]), np.array([[-1.2, 0.0]]))
    assert np.allclose(pos, [[0.7, 10.0]]) and np.allclose(vel, [[1.2, 0.0]])
    pos, vel = sim.advance(np.array([[10.0, 26.5]]), np.array([[0.0, 1.2]]))
    assert np.allclose(pos, [[10.0, 26.3]]) and np.allclose(vel, [[0.0, -1.2]])


def test_squares_keep_their_speed_between_bounces():
    pos, vel = np.array([[5.0, 5.0]]), np.array([[0.8, -0.6]])
    for _ in range(12):
        pos, vel = sim.advance(pos, vel)
        assert np.allclose(np.abs(vel), [[0.8, 0.6]])
        assert (pos >= 0).all() and (pos <= sim.POS_MAX).all()


def test_covered_share_on_hand_built_cases():
    bar = np.array([10, 10, 10, 10, 10, 10])
    x = np.array([10.0, 13.0, 13.5, 5.0, 3.0, 18.0])
    share = sim.covered_share(x, bar)
    assert np.allclose(share, [1.0, 1.0, 0.9, 0.0, 0.0, 0.0])


def test_hidden_and_visible_flags():
    ep = sim.simulate(100, seed=7)
    assert not (ep.hidden & ep.visible).any()
    assert ep.hidden.any() and ep.visible.any()
    inside = (ep.pos[..., 0] >= ep.bar[..., None]) & (
        ep.pos[..., 0] + sim.SQUARE <= ep.bar[..., None] + sim.BAR_W
    )
    assert np.array_equal(ep.hidden, inside)


def test_render_square_and_bar():
    pos = np.array([[[2.0, 3.0], [15.0, 20.0], [12.5, 8.0]]])  # one frame
    frames = sim.render(pos, np.array([24]))
    red, green, blue = frames[0]
    assert red[3:8, 2:7].min() == 1.0 and red[:, :24].sum() == 25.0
    assert green[20:25, 15:20].min() == 1.0
    assert np.isclose(blue[8:13, 12].sum(), 2.5)
    assert np.isclose(blue[:, :24].sum(), 25.0)
    assert np.allclose(frames[0, :, :, 24:], sim.BAR_VALUE)


def test_hidden_square_leaves_only_the_bar():
    pos = np.array([[[11.0, 5.0], [11.0, 9.0], [12.0, 20.0]]])
    frame = sim.render(pos, np.array([10]))[0]
    assert np.allclose(frame[:, :, 10:18], sim.BAR_VALUE)
    assert np.allclose(frame[:, :, :10], 0) and np.allclose(frame[:, :, 18:], 0)
