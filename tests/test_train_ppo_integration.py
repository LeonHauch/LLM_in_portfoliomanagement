# tests/test_train_ppo_integration.py

import pytest
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from src.rl_agent.train_ppo import PortfolioTrainer

@pytest.mark.integration
def test_training_runs_and_saves(tmp_path):
    """
    Integration test for PortfolioTrainer:
    - Initializes with a sample dataset
    - Runs a short training session
    - Saves model + config + stats
    - Reloads model and evaluates
    """
    # --- Setup ---
    data_path = "data/sample/data_ppo_sample.parquet" 
    trainer = PortfolioTrainer(
        config_path=None,
        data_path=data_path
    )

    # Override config for short test
    trainer.config["training"]["total_timesteps"] = 500
    trainer.config["training"]["n_envs"] = 1
    trainer.config["saving"]["save_freq"] = 0
    trainer.config["evaluation"]["eval_freq"] = 0

    # Redirect output dirs to tmp_path
    trainer.model_dir = tmp_path / "models"
    trainer.log_dir = tmp_path / "logs"
    trainer.results_dir = tmp_path / "results"
    for d in [trainer.model_dir, trainer.log_dir, trainer.results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Train ---
    model = trainer.train()

    # --- Assertions ---
    final_model_path = trainer.model_dir / "final_model.zip"
    assert final_model_path.exists(), "Final model not saved"

    config_path = trainer.results_dir / "config.yaml"
    assert config_path.exists(), "Config not saved"

    stats_path = trainer.results_dir / "training_stats.json"
    assert stats_path.exists(), "Training stats not saved"

    # --- Reload and evaluate ---
    loaded_model = PPO.load(final_model_path, env=trainer.create_vec_env())
    mean_reward, std_reward = evaluate_policy(loaded_model, trainer.create_vec_env(), n_eval_episodes=1)

    assert isinstance(mean_reward, float)
    assert not np.isnan(mean_reward)


@pytest.mark.integration
def test_predict_after_training(tmp_path):
    """
    Ensure trained PPO model can predict valid actions.
    """
    data_path = "data/sample/data_ppo_sample.parquet"
    trainer = PortfolioTrainer(data_path=data_path)

    trainer.config["training"]["total_timesteps"] = 300
    trainer.config["training"]["n_envs"] = 1
    trainer.config["saving"]["save_freq"] = 0
    trainer.config["evaluation"]["eval_freq"] = 0

    trainer.model_dir = tmp_path / "models"
    trainer.log_dir = tmp_path / "logs"
    trainer.results_dir = tmp_path / "results"
    for d in [trainer.model_dir, trainer.log_dir, trainer.results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model = trainer.train()

    env = trainer.create_vec_env()
    obs = env.reset()
    action, _ = model.predict(obs)

    assert action.shape[-1] == env.action_space.shape[0]
    assert np.all(np.isfinite(action)), "Model produced invalid actions"


@pytest.mark.integration
def test_reproducibility(tmp_path):
    """
    Optional: with fixed seeds, PPO training should be reproducible.
    """
    data_path = "data/sample/data_ppo_sample.parquet"

    # First trainer
    trainer1 = PortfolioTrainer(data_path=data_path)
    trainer1.config["training"]["total_timesteps"] = 200
    trainer1.config["training"]["n_envs"] = 1
    trainer1.model_dir = tmp_path / "m1"; trainer1.model_dir.mkdir()
    trainer1.log_dir = tmp_path / "l1"; trainer1.log_dir.mkdir()
    trainer1.results_dir = tmp_path / "r1"; trainer1.results_dir.mkdir()

    # Second trainer
    trainer2 = PortfolioTrainer(data_path=data_path)
    trainer2.config = trainer1.config.copy()  # exact same config
    trainer2.seed = trainer1.seed
    trainer2.model_dir = tmp_path / "m2"; trainer2.model_dir.mkdir()
    trainer2.log_dir = tmp_path / "l2"; trainer2.log_dir.mkdir()
    trainer2.results_dir = tmp_path / "r2"; trainer2.results_dir.mkdir()

    model1 = trainer1.train()
    model2 = trainer2.train()

    env = trainer1.create_vec_env()
    obs = env.reset(seed=42)

    a1, _ = model1.predict(obs)
    a2, _ = model2.predict(obs)

    np.testing.assert_array_almost_equal(a1, a2, decimal=4)
