# tests/test_eval_ppo.py
import os
import tempfile
import pytest
import numpy as np
from src.rl_agent import eval_ppo

@pytest.fixture
def dummy_paths(tmp_path):
    """Fixture to provide temporary model/data paths."""
    model_path = tmp_path / "dummy_model.zip"
    data_path = tmp_path / "dummy_data.parquet"

    # Create empty files (content doesn't matter due to monkeypatching)
    model_path.write_text("")
    data_path.write_text("")

    return str(model_path), str(data_path)

def test_model_loading_fails_without_file(dummy_paths):
    model_path, data_path = dummy_paths
    os.remove(model_path)
    with pytest.raises(FileNotFoundError):
        eval_ppo.PortfolioEvaluator(model_path, data_path)

def test_evaluate_and_backtest(monkeypatch, tmp_path, dummy_paths):
    model_path, data_path = dummy_paths

    # Dummy PPO model
    class DummyModel:
        device = "cpu"
        def predict(self, obs, deterministic=True):
            return np.array([0]), None

    monkeypatch.setattr("src.rl_agent.eval_ppo.PPO", type("PPO", (), {"load": staticmethod(lambda *a, **k: DummyModel())}))

    # Dummy environment for evaluate_episodes
    class DummyEvalEnv:
        def __init__(self, *a, **k): self.steps = 0
        def reset(self, seed=None): return np.zeros(3)
        def step(self, action):
            self.steps += 1
            done = self.steps >= 2
            info = {"portfolio_value": 100000 + self.steps, "weights": [0.5, 0.5]}
            return np.zeros(3), 1.0, done, info
        def close(self): pass

    # Dummy environment for backtest
    class DummyBacktestEnv(DummyEvalEnv):
        def reset(self, seed=None):
            return np.zeros(3), {}
        def step(self, action):
            self.steps += 1
            done = self.steps >= 3
            info = {"portfolio_value": 100000 + self.steps, "weights": [0.5, 0.5], "net_return": 0.01}
            return np.zeros(3), 1.0, False, False, info

    monkeypatch.setattr("environment.portfolio_env.PortfolioEnv", DummyEvalEnv)

    evaluator = eval_ppo.PortfolioEvaluator(model_path, data_path, results_dir=tmp_path)

    metrics = evaluator.evaluate_episodes(n_episodes=2, n_envs=1)
    assert "mean_reward" in metrics
    assert metrics["n_episodes"] == 2

    # Patch environment for backtest
    monkeypatch.setattr("environment.portfolio_env.PortfolioEnv", DummyBacktestEnv)
    backtest_results = evaluator.backtest()
    assert "total_return" in backtest_results
    assert backtest_results["final_value"] > backtest_results["initial_value"]
