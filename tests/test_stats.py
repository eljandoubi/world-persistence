import numpy as np

from worldpersist import stats


def test_draws_centre_on_the_pooled_mean_of_events():
    episode = np.concatenate([np.arange(20), np.repeat(np.arange(20, 40), 9)])
    values = np.concatenate([np.ones(20), np.zeros(180)])
    draws = stats.episode_bootstrap(episode, values, draws=4000)
    assert draws.shape == (4000, 1)
    assert abs(draws.mean() - 0.1) < 0.02  # events weigh the same, not episodes


def test_events_of_one_episode_move_together():
    episode = np.repeat(np.arange(20), 10)
    values = (episode % 2).astype(float)  # every event of an episode agrees
    draws = stats.episode_bootstrap(episode, values, draws=2000)[:, 0]
    assert abs(draws.mean() - 0.5) < 0.02
    assert draws.std() > 0.08  # 200 independent events would give about 0.035


def test_columns_are_resampled_together_and_the_seed_fixes_the_draws():
    rng = np.random.default_rng(1)
    episode = rng.integers(0, 30, size=120)
    first = rng.random(120)
    draws = stats.episode_bootstrap(episode, np.stack([first, first], axis=1), 300)
    assert np.allclose(draws[:, 0], draws[:, 1])
    again = stats.episode_bootstrap(episode, np.stack([first, first], axis=1), 300)
    assert np.array_equal(draws, again)


def test_interval_brackets_the_estimate():
    episode = np.repeat(np.arange(40), 5)
    values = np.random.default_rng(2).random(200)
    draws = stats.episode_bootstrap(episode, values)
    low, high = stats.interval(draws)
    assert low[0] < values.mean() < high[0]
    assert np.isclose(
        ((draws[:, 0] >= low[0]) & (draws[:, 0] <= high[0])).mean(), 0.95, atol=0.01
    )
