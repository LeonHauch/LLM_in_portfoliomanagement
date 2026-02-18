# src/rl_agent/train_ppo.py
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch
import argparse
import logging
from datetime import datetime
import yaml
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from environment.portfolio_env import PortfolioEnv
from src.rl_agent.custom_policies import get_policy_kwargs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioTrainer:
    """PPO trainer with train/test split, policy selection, and multi-seed support."""

    def __init__(
        self,
        config_path: str = None,
        data_path: str = None,
        sentiment_path: str = None,
    ):
        self.config = self._load_config(config_path)
        self.data_path = data_path or self.config.get("data_path")
        self.sentiment_path = sentiment_path or self.config.get("sentiment_path")

        self._prepare_data()
        self.setup_directories()

        self.seed = self.config.get("seed", 42)
        set_random_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        logger.info(f"Trainer initialized with seed: {self.seed}")

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def _load_config(self, config_path: str) -> dict:
        default_config = {
            "seed": 42,
            "data_path": "data/preprocessed/data_ppo.parquet",
            "sentiment_path": None,
            "env": {
                "lookback_window": 60,
                "initial_capital": 100000.0,
                "transaction_cost": 0.001,
                "max_positions": None,
                "cash_weight": True,
                "normalize_observations": True,
                "reward_scaling": 1000.0,
                "use_sentiment": False,
                "risk_bonus_weight": 0.0,
            },
            # Train/test split
            "split": {
                "train_end_idx": None,
                "test_start_idx": None,
                "train_end_date": "2022-12-31",
                "test_start_date": "2023-01-01",
            },
            "policy": {
                "type": "mlp",       # mlp | cnn | lstm
                "features_dim": 128,
            },
            "training": {
                "total_timesteps": 1000000,
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": None,
                "n_envs": 4,
                "use_multiprocessing": False,
            },
            "evaluation": {
                "eval_freq": 10000,
                "n_eval_episodes": 10,
                "eval_deterministic": True,
            },
            "saving": {
                "save_freq": 50000,
                "save_path": "models/ppo_portfolio",
                "keep_best_only": True,
            },
            "logging": {
                "tensorboard": True,
                "wandb": False,
                "log_interval": 100,
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                if config_path.endswith((".yaml", ".yml")):
                    loaded = yaml.safe_load(f)
                else:
                    loaded = json.load(f)
            self._deep_update(default_config, loaded)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.info("Using default configuration")

        return default_config

    @staticmethod
    def _deep_update(base: dict, update: dict):
        for k, v in update.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                PortfolioTrainer._deep_update(base[k], v)
            else:
                base[k] = v

    # ------------------------------------------------------------------
    # Data preparation (sentiment merge)
    # ------------------------------------------------------------------
    def _prepare_data(self):
        if self.sentiment_path and os.path.exists(self.sentiment_path):
            logger.info(f"Loading sentiment from: {self.sentiment_path}")
            try:
                market = pd.read_parquet(self.data_path)
                sent = pd.read_csv(self.sentiment_path)
                sent["date"] = pd.to_datetime(sent["date"])

                # Pivot sentiment to wide format: Sentiment_{ticker}
                if "asset" in sent.columns and "sentiment" in sent.columns:
                    pivot = sent.pivot_table(
                        index="date", columns="asset", values="sentiment", aggfunc="mean"
                    )
                    pivot.columns = [f"Sentiment_{c}" for c in pivot.columns]
                    # Align with market index
                    if hasattr(market.index, "date"):
                        pivot.index = pd.to_datetime(pivot.index)
                        market = market.join(pivot, how="left")
                    else:
                        market = market.merge(
                            pivot, left_index=True, right_index=True, how="left"
                        )
                    # Fill missing sentiment with 0
                    for c in pivot.columns:
                        market[c] = market[c].fillna(0)

                    temp_dir = Path(self.data_path).parent / "temp"
                    temp_dir.mkdir(exist_ok=True)
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_path = temp_dir / f"merged_sentiment_{ts}.parquet"
                    market.to_parquet(temp_path)
                    self.data_path = str(temp_path)
                    logger.info(f"Merged sentiment data saved to {temp_path}")
                else:
                    logger.warning("Sentiment CSV missing 'asset'/'sentiment' columns")
            except Exception as e:
                logger.error(f"Failed to merge sentiment: {e}")
        elif self.sentiment_path:
            logger.warning(f"Sentiment path not found: {self.sentiment_path}")
        else:
            logger.info("Training without sentiment features")

    # ------------------------------------------------------------------
    # Directories
    # ------------------------------------------------------------------
    def setup_directories(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = Path(f"models/ppo_portfolio_{self.timestamp}")
        self.log_dir = Path(f"logs/ppo_portfolio_{self.timestamp}")
        self.results_dir = Path(f"results/ppo_portfolio_{self.timestamp}")
        for d in [self.model_dir, self.log_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Environment creation (with split support)
    # ------------------------------------------------------------------
    def _env_kwargs(self, split: str = "train") -> dict:
        """Build PortfolioEnv kwargs for a given split."""
        env_cfg = dict(self.config["env"])
        split_cfg = self.config.get("split", {})

        if split == "train":
            env_cfg["end_date"] = split_cfg.get("train_end_date")
            env_cfg["end_idx"] = split_cfg.get("train_end_idx")
        elif split == "test":
            env_cfg["start_date"] = split_cfg.get("test_start_date")
            env_cfg["start_idx"] = split_cfg.get("test_start_idx")

        return env_cfg

    def create_env(self, rank: int = 0, split: str = "train"):
        env_kwargs = self._env_kwargs(split)

        def _init():
            env = PortfolioEnv(data_path=self.data_path, **env_kwargs)
            env = Monitor(env, str(self.log_dir / f"monitor_{split}_{rank}.csv"))
            return env

        return _init

    def create_vec_env(self, split: str = "train"):
        n_envs = self.config["training"]["n_envs"]
        use_mp = self.config["training"]["use_multiprocessing"]
        fns = [self.create_env(i, split) for i in range(n_envs)]
        if use_mp and n_envs > 1:
            env = SubprocVecEnv(fns)
        else:
            env = DummyVecEnv(fns)
        logger.info(f"Created {split} vec-env with {n_envs} instances")
        return env

    # ------------------------------------------------------------------
    # Model creation (with policy selection)
    # ------------------------------------------------------------------
    def create_model(self, env):
        tc = self.config["training"]
        policy_cfg = self.config.get("policy", {})
        policy_type = policy_cfg.get("type", "mlp")

        # Get env metadata for structured policies
        sample_env = env.envs[0] if hasattr(env, "envs") else env
        # Unwrap Monitor if needed
        inner = sample_env
        while hasattr(inner, "env"):
            inner = inner.env
        n_assets = inner.n_assets
        n_features = inner.n_features_per_asset
        lookback = inner.lookback_window

        policy_kwargs = get_policy_kwargs(
            policy_type=policy_type,
            n_assets=n_assets,
            n_features_per_asset=n_features,
            lookback_window=lookback,
            observation_space=env.observation_space,
            features_dim=policy_cfg.get("features_dim", 128),
        )

        model_params = {
            "learning_rate": tc["learning_rate"],
            "n_steps": tc["n_steps"],
            "batch_size": tc["batch_size"],
            "n_epochs": tc["n_epochs"],
            "gamma": tc["gamma"],
            "gae_lambda": tc["gae_lambda"],
            "clip_range": tc["clip_range"],
            "ent_coef": tc["ent_coef"],
            "vf_coef": tc["vf_coef"],
            "max_grad_norm": tc["max_grad_norm"],
            "target_kl": tc.get("target_kl"),
            "tensorboard_log": str(self.log_dir)
            if self.config["logging"]["tensorboard"]
            else None,
            "seed": self.seed,
            "verbose": 1,
            "policy_kwargs": policy_kwargs,
        }

        model = PPO("MlpPolicy", env, **model_params)
        logger.info(f"PPO model created (policy={policy_type})")
        return model

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def create_callbacks(self, env):
        callbacks = []
        ec = self.config["evaluation"]
        if ec["eval_freq"] > 0:
            eval_env = DummyVecEnv([self.create_env(split="train")])
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(self.model_dir),
                    log_path=str(self.log_dir),
                    eval_freq=ec["eval_freq"],
                    n_eval_episodes=ec["n_eval_episodes"],
                    deterministic=ec["eval_deterministic"],
                    render=False,
                )
            )
        sc = self.config["saving"]
        if sc["save_freq"] > 0:
            callbacks.append(
                CheckpointCallback(
                    save_freq=sc["save_freq"],
                    save_path=str(self.model_dir),
                    name_prefix="ppo_portfolio",
                )
            )
        return CallbackList(callbacks) if callbacks else None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        logger.info("Starting PPO training...")
        self.save_config()
        env = self.create_vec_env("train")
        model = self.create_model(env)
        callbacks = self.create_callbacks(env)
        total = self.config["training"]["total_timesteps"]

        try:
            model.learn(
                total_timesteps=total,
                callback=callbacks,
                log_interval=self.config["logging"]["log_interval"],
                progress_bar=True,
            )
            path = self.model_dir / "final_model"
            model.save(path)
            logger.info(f"Training completed! Model saved to {path}")
            self.save_training_stats(model)
        except KeyboardInterrupt:
            path = self.model_dir / "interrupted_model"
            model.save(path)
            logger.info(f"Training interrupted. Model saved to {path}")
        finally:
            env.close()

        return model

    def save_config(self):
        p = self.results_dir / "config.yaml"
        with open(p, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)

    def save_training_stats(self, model):
        stats = {
            "total_timesteps": model.num_timesteps,
            "training_completed": True,
            "model_path": str(self.model_dir),
            "config": self.config,
            "timestamp": self.timestamp,
        }
        p = self.results_dir / "training_stats.json"
        with open(p, "w") as f:
            json.dump(stats, f, indent=2, default=str)


# ======================================================================
# Multi-seed training
# ======================================================================
def train_multi_seed(
    config_path: str,
    data_path: str,
    seeds: list,
    sentiment_path: str = None,
) -> list:
    """Train PPO across multiple seeds and return list of model paths."""
    model_paths = []
    for seed in seeds:
        logger.info(f"\n{'='*60}\nTraining with seed={seed}\n{'='*60}")
        trainer = PortfolioTrainer(
            config_path=config_path,
            data_path=data_path,
            sentiment_path=sentiment_path,
        )
        trainer.config["seed"] = seed
        trainer.seed = seed
        set_random_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        trainer.train()
        model_paths.append(str(trainer.model_dir / "final_model.zip"))
    return model_paths


# ======================================================================
# CLI
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Train PPO for Portfolio Optimization")
    parser.add_argument("--data-path", type=str, help="Path to parquet data file")
    parser.add_argument("--config", type=str, default=None, help="Config YAML/JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed training (e.g. 42,123,456)",
    )
    parser.add_argument("--sentiment-path", type=str, default=None)
    args = parser.parse_args()

    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
        paths = train_multi_seed(args.config, args.data_path, seeds, args.sentiment_path)
        logger.info(f"Multi-seed training done. Models: {paths}")
    else:
        trainer = PortfolioTrainer(
            config_path=args.config,
            data_path=args.data_path,
            sentiment_path=args.sentiment_path,
        )
        if args.seed != 42:
            trainer.config["seed"] = args.seed
            trainer.seed = args.seed
        if not os.path.exists(trainer.data_path):
            raise FileNotFoundError(f"Data file not found: {trainer.data_path}")
        trainer.train()


if __name__ == "__main__":
    main()
