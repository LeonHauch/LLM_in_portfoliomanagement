# src/rl_agent/tune_ppo.py
"""Optuna-based hyperparameter tuning for PPO portfolio agent."""

import os
import sys
import argparse
import logging
import json
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from environment.portfolio_env import PortfolioEnv
from src.rl_agent.custom_policies import get_policy_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrialEvalCallback(BaseCallback):
    """Callback that reports intermediate results to Optuna for pruning."""

    def __init__(self, eval_env, trial, n_eval_episodes=3, eval_freq=5000):
        super().__init__()
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.eval_idx = 0
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
            rewards.append(ep_reward)

        mean_reward = float(np.mean(rewards))
        self.last_mean_reward = mean_reward
        self.trial.report(mean_reward, self.eval_idx)
        self.eval_idx += 1

        if self.trial.should_prune():
            raise optuna.TrialPruned()

        return True


def sample_ppo_params(trial: optuna.Trial) -> dict:
    """Sample PPO hyperparameters for a single Optuna trial."""
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_int("n_epochs", 3, 15)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 1.0)

    return {
        "learning_rate": lr,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_range": clip_range,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
    }


def create_env(data_path: str, env_config: dict, seed: int = 42):
    """Create a single PortfolioEnv."""
    env = PortfolioEnv(data_path=data_path, **env_config)
    env.reset(seed=seed)
    return env


class PPOTuner:
    """Optuna hyperparameter tuner for PPO portfolio agent."""

    def __init__(
        self,
        data_path: str,
        config_path: str = None,
        n_trials: int = 30,
        n_timesteps: int = 50_000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 3,
        seed: int = 42,
        study_name: str = "ppo_portfolio",
        storage: str = None,
    ):
        self.data_path = data_path
        self.n_trials = n_trials
        self.n_timesteps = n_timesteps
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.seed = seed
        self.study_name = study_name
        self.storage = storage

        # Load base config for env settings
        self.config = self._load_config(config_path)
        self.env_config = self.config.get("env", {
            "lookback_window": 60,
            "initial_capital": 100000.0,
            "transaction_cost": 0.001,
            "cash_weight": True,
            "normalize_observations": True,
            "reward_scaling": 1000.0,
            "use_sentiment": False,
        })

        # Apply train split
        split_cfg = self.config.get("split", {})
        if split_cfg.get("train_end_date"):
            self.env_config["end_date"] = split_cfg["train_end_date"]
        if split_cfg.get("train_end_idx"):
            self.env_config["end_idx"] = split_cfg["train_end_idx"]

        self.policy_type = self.config.get("policy", {}).get("type", "mlp")
        self.features_dim = self.config.get("policy", {}).get("features_dim", 128)

        # Results directory
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(f"results/tune_{ts}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> dict:
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial: sample params, train, evaluate, return mean reward."""
        set_random_seed(self.seed)
        torch.manual_seed(self.seed)

        params = sample_ppo_params(trial)

        # Create train env
        train_env = DummyVecEnv([
            lambda: Monitor(create_env(self.data_path, self.env_config, self.seed))
        ])

        # Create eval env (same split for tuning — we compare relative performance)
        eval_env = create_env(self.data_path, self.env_config, self.seed + 1)

        # Get policy kwargs
        inner = train_env.envs[0]
        while hasattr(inner, "env"):
            inner = inner.env
        policy_kwargs = get_policy_kwargs(
            policy_type=self.policy_type,
            n_assets=inner.n_assets,
            n_features_per_asset=inner.n_features_per_asset,
            lookback_window=inner.lookback_window,
            observation_space=train_env.observation_space,
            features_dim=self.features_dim,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            seed=self.seed,
            verbose=0,
            policy_kwargs=policy_kwargs,
            **params,
        )

        eval_callback = TrialEvalCallback(
            eval_env, trial,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=self.eval_freq,
        )

        try:
            model.learn(total_timesteps=self.n_timesteps, callback=eval_callback)
        except optuna.TrialPruned:
            train_env.close()
            eval_env.close()
            raise

        # Final evaluation
        rewards = []
        for _ in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            ep_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_reward += reward
            rewards.append(ep_reward)

        train_env.close()
        eval_env.close()

        mean_reward = float(np.mean(rewards))
        logger.info(f"Trial {trial.number}: mean_reward={mean_reward:.4f}, params={params}")
        return mean_reward

    def tune(self) -> dict:
        """Run the full Optuna study and return best params."""
        sampler = TPESampler(seed=self.seed)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )

        logger.info(f"Starting Optuna study: {self.n_trials} trials, "
                     f"{self.n_timesteps} timesteps each")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)

        # Log results
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Save results
        results = {
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "study_name": self.study_name,
        }

        results_path = self.results_dir / "tuning_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        # Save best config as YAML (ready to use with train_ppo.py)
        best_config = dict(self.config)
        best_config.setdefault("training", {})
        best_config["training"].update(study.best_params)
        config_path = self.results_dir / "best_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(best_config, f, default_flow_style=False)
        logger.info(f"Best config saved to {config_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for PPO")
    parser.add_argument("--data-path", type=str, required=True,
                        help="Path to parquet data file")
    parser.add_argument("--config", type=str, default=None,
                        help="Base config YAML (env/policy settings)")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--n-timesteps", type=int, default=50000,
                        help="Training timesteps per trial")
    parser.add_argument("--eval-freq", type=int, default=5000,
                        help="Evaluation frequency for pruning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="ppo_portfolio")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g. sqlite:///study.db)")
    args = parser.parse_args()

    tuner = PPOTuner(
        data_path=args.data_path,
        config_path=args.config,
        n_trials=args.n_trials,
        n_timesteps=args.n_timesteps,
        eval_freq=args.eval_freq,
        seed=args.seed,
        study_name=args.study_name,
        storage=args.storage,
    )
    tuner.tune()


if __name__ == "__main__":
    main()
