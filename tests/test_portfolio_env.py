import pytest
import numpy as np
import pandas as pd
from environment.portfolio_env import PortfolioEnv


@pytest.fixture
def dummy_data(tmp_path):
    """Creates dummy data to simulate preprocessed input."""
    dates = pd.date_range("2020-01-01", periods=200)
    rng = np.random.RandomState(42)
    data = {
        "Adj Close_AAPL": np.linspace(100, 150, 200),
        "Adj Close_GOOG": np.linspace(200, 250, 200),
        "Volume_Ratio_AAPL": rng.rand(200),
        "Volume_Ratio_GOOG": rng.rand(200),
    }
    df = pd.DataFrame(data, index=dates)
    file_path = tmp_path / "dummy.parquet"
    df.to_parquet(file_path)
    return str(file_path)


@pytest.fixture
def dummy_data_with_sentiment(tmp_path):
    """Dummy data including sentiment columns."""
    dates = pd.date_range("2020-01-01", periods=200)
    rng = np.random.RandomState(42)
    data = {
        "Adj Close_AAPL": np.linspace(100, 150, 200),
        "Adj Close_GOOG": np.linspace(200, 250, 200),
        "Volume_Ratio_AAPL": rng.rand(200),
        "Volume_Ratio_GOOG": rng.rand(200),
        "Sentiment_AAPL": rng.uniform(-1, 1, 200),
        "Sentiment_GOOG": rng.uniform(-1, 1, 200),
    }
    df = pd.DataFrame(data, index=dates)
    file_path = tmp_path / "dummy_sent.parquet"
    df.to_parquet(file_path)
    return str(file_path)


def test_env_initialization(dummy_data):
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10)
    assert env.n_assets == 2
    assert env.price_data.shape[0] == 200
    assert env.price_data.shape[1] == 2
    assert env.action_space.shape == (3,)  # 2 assets + cash


def test_env_reset(dummy_data):
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == env.observation_space.shape[0]
    assert np.isclose(np.sum(env.current_weights), 1.0, atol=1e-3)


def test_step_returns_valid_output(dummy_data):
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10)
    obs, info = env.reset()
    action = np.array([0.3, 0.3, 0.4])
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert np.isfinite(reward)
    assert obs.shape == env.observation_space.shape
    assert isinstance(terminated, bool)
    assert isinstance(info, dict)


def test_action_processing(dummy_data):
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10, max_positions=1)
    obs, info = env.reset()
    raw_action = np.array([10.0, 1.0, 0.0])
    weights = env._process_action(raw_action)
    assert np.isclose(np.sum(weights), 1.0)
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    assert np.count_nonzero(weights > 0) <= 1


def test_log_returns(dummy_data):
    """Returns should be log-returns, not arithmetic."""
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10)
    prices = env.price_data.iloc[:3]
    expected = np.log(prices.iloc[1] / prices.iloc[0])
    actual = env.returns_data.iloc[1]
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-6)


def test_train_test_split_by_idx(dummy_data):
    """Env should respect start_idx / end_idx boundaries."""
    env = PortfolioEnv(
        data_path=dummy_data, lookback_window=10, start_idx=50, end_idx=150
    )
    assert env.valid_start_idx >= 50
    assert env.valid_end_idx <= 150

    for _ in range(20):
        env.reset()
        assert env.current_step >= env.valid_start_idx
        assert env.current_step < env.valid_end_idx


def test_train_test_split_by_date(dummy_data):
    """Env should respect start_date / end_date boundaries."""
    env = PortfolioEnv(
        data_path=dummy_data,
        lookback_window=10,
        start_date="2020-03-01",
        end_date="2020-05-01",
    )
    assert env.valid_start_idx > 10
    assert env.valid_end_idx < 199


def test_risk_bonus_weight_zero(dummy_data):
    """With risk_bonus_weight=0, reward should be pure return * scaling."""
    env = PortfolioEnv(
        data_path=dummy_data, lookback_window=10, risk_bonus_weight=0.0
    )
    obs, _ = env.reset()
    for _ in range(25):
        action = np.ones(env.action_space.shape[0])
        obs, reward, done, _, _ = env.step(action)
        if done:
            break


def test_sentiment_observation(dummy_data_with_sentiment):
    """Sentiment columns should add an extra feature per asset."""
    env_no_sent = PortfolioEnv(
        data_path=dummy_data_with_sentiment, lookback_window=10, use_sentiment=False
    )
    env_sent = PortfolioEnv(
        data_path=dummy_data_with_sentiment, lookback_window=10, use_sentiment=True
    )
    assert env_sent.n_features_per_asset == env_no_sent.n_features_per_asset + 1
    assert env_sent.observation_space.shape[0] > env_no_sent.observation_space.shape[0]
    assert env_sent.has_sentiment is True


def test_observation_clipping(dummy_data):
    """When normalize_observations=True, obs should be clipped to [-10, 10]."""
    env = PortfolioEnv(
        data_path=dummy_data, lookback_window=10, normalize_observations=True
    )
    obs, _ = env.reset()
    assert obs.max() <= 10.0
    assert obs.min() >= -10.0
