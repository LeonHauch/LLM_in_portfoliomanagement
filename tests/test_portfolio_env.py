import pytest
import numpy as np
import pandas as pd
from environment.portfolio_env import PortfolioEnv
import os

@pytest.fixture
def dummy_data(tmp_path):
    """Creates dummy data to simulate preprocessed input."""
    dates = pd.date_range("2020-01-01", periods=100)
    data = {
        "Adj Close_AAPL": np.linspace(100, 150, 100),
        "Adj Close_GOOG": np.linspace(200, 250, 100),
        "Volume_Ratio_AAPL": np.random.rand(100),
        "Volume_Ratio_GOOG": np.random.rand(100)
    }
    df = pd.DataFrame(data, index=dates)
    file_path = tmp_path / "dummy.parquet"
    df.to_parquet(file_path)
    return str(file_path)

def test_env_initialization(dummy_data):
    env = PortfolioEnv(data_path=dummy_data, lookback_window=10)
    assert env.n_assets == 2
    assert env.price_data.shape[0] == 100
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
    raw_action = np.array([10.0, 1.0, 0.0])  # Will be softmaxed
    weights = env._process_action(raw_action)
    assert np.isclose(np.sum(weights), 1.0)
    assert (weights >= 0).all()
    assert (weights <= 1).all()
    assert np.count_nonzero(weights > 0) <= 1  # max_positions=1
