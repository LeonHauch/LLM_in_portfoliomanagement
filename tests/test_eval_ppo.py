import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.rl_agent import eval_ppo

@pytest.fixture
def dummy_paths(tmp_path: eval_ppo.Path):
    """Fixture to provide a temporary model path and the real data path."""
    model_path = tmp_path / "models.sample_model.zip"
    
    # Correctly point to the real, valid data file
    data_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'data', 
        'sample', 
        'data_ppo_sample.parquet'
    )

    # Create empty model file (content doesn't matter due to monkeypatching)
    model_path.write_text("")

    return str(model_path), str(data_path)

def test_model_loading_fails_without_file(dummy_paths: tuple[str, str]):
    model_path, data_path = dummy_paths
    os.remove(model_path)
    with pytest.raises(FileNotFoundError):
        eval_ppo.PortfolioEvaluator(model_path, data_path)

def test_evaluate_and_backtest(monkeypatch: pytest.MonkeyPatch, tmp_path: eval_ppo.Path, dummy_paths: tuple[str, str]):
    model_path, data_path = dummy_paths

    # Dummy PPO model
    class DummyModel:
        device = "cpu"
        def predict(self, obs, deterministic=True):
            # --- FIX: Return a valid array of weights, not a scalar ---
            # The real PortfolioEnv expects an action array of a certain size.
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2]), None
            # --- END FIX ---

    monkeypatch.setattr("src.rl_agent.eval_ppo.PPO", type("PPO", (), {"load": staticmethod(lambda *a, **k: DummyModel())}))

    # Dummy environment for evaluate_episodes
    class DummyEvalEnv:
        def __init__(self, data_path=None): 
            self.steps = 0
            self.data_path = data_path
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
