# src/training/train_ppo.py
import os
import numpy as np
import pandas as pd

from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    StopTrainingOnRewardThreshold,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import torch
import argparse
import logging
from datetime import datetime
from pathlib import Path
import yaml
import json

# Import your custom environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from environment.portfolio_env import PortfolioEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioTrainer:
    """Main trainer class for PPO portfolio optimization."""
    
    def __init__(self, config_path: str = None, data_path: str = None, sentiment_path: str = None):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to configuration file
            data_path: Path to the parquet data file
            sentiment_path: Optional path to sentiment CSV file
        """
        self.config = self._load_config(config_path)
        self.data_path = data_path or self.config.get('data_path')
        self.sentiment_path = sentiment_path or self.config.get('sentiment_path')  # NEW
        
        # Prepare data (merge sentiment if provided)
        self._prepare_data()  # NEW
        
        # Setup directories
        self.setup_directories()
        
        # Set random seeds for reproducibility
        self.seed = self.config.get('seed', 42)
        set_random_seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        logger.info(f"Trainer initialized with seed: {self.seed}")
    
    def _prepare_data(self):
        """
        Load market data and optionally merge with sentiment.
        If sentiment is provided, creates a temporary merged file.
        """
        
        # Check if sentiment should be merged
        if self.sentiment_path and os.path.exists(self.sentiment_path):
            logger.info(f"📊 Loading sentiment from: {self.sentiment_path}")
            
            try:
                # Load market data
                market_data = pd.read_parquet(self.data_path)
                logger.info(f"   Market data shape: {market_data.shape}")
                
                # Load sentiment
                sentiment_data = pd.read_csv(self.sentiment_path)
                sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
                logger.info(f"   Sentiment data shape: {sentiment_data.shape}")
                
                # Ensure market data has datetime date column
                if 'date' in market_data.columns:
                    market_data['date'] = pd.to_datetime(market_data['date'])
                
                # Merge on date and asset
                merged_data = market_data.merge(
                    sentiment_data,
                    on=['date', 'asset'],
                    how='left'
                )
                
                # Fill missing sentiment with 0 (neutral for days without news)
                sentiment_cols = [c for c in sentiment_data.columns 
                                if c not in ['date', 'asset']]
                for col in sentiment_cols:
                    merged_data[col] = merged_data[col].fillna(0)
                    logger.info(f"   Filled {merged_data[col].isna().sum()} missing values in '{col}'")
                
                logger.info(f"✅ Merged data shape: {merged_data.shape}")
                logger.info(f"   Added sentiment columns: {sentiment_cols}")
                
                # Create temporary merged file
                temp_dir = Path(self.data_path).parent / 'temp'
                temp_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                temp_path = temp_dir / f"merged_with_sentiment_{timestamp}.parquet"
                merged_data.to_parquet(temp_path)
                
                # Update data_path to point to merged file
                self.original_data_path = self.data_path  # Keep reference to original
                self.data_path = str(temp_path)
                
                logger.info(f"💾 Temporary merged data saved to: {temp_path}")
                
            except Exception as e:
                logger.error(f"❌ Failed to merge sentiment data: {e}")
                logger.info("   Continuing with original data without sentiment")
                import traceback
                traceback.print_exc()
        
        elif self.sentiment_path:
            logger.warning(f"⚠️  Sentiment path provided but file not found: {self.sentiment_path}")
            logger.info("   Training without sentiment features")
        else:
            logger.info("📈 Training without sentiment features (none provided)")
           
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'seed': 42,
            'data_path': 'data/processed/portfolio_data.parquet',
            
            # Environment parameters
            'env': {
                'lookback_window': 60,
                'initial_capital': 100000.0,
                'transaction_cost': 0.001,
                'max_positions': None,
                'cash_weight': True,
                'normalize_observations': True,
                'reward_scaling': 1000.0
            },
            
            # Training parameters
            'training': {
                'total_timesteps': 1000000,
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'target_kl': None,
                'n_envs': 4,
                'use_multiprocessing': False
            },
            
            # Evaluation parameters
            'evaluation': {
                'eval_freq': 10000,
                'n_eval_episodes': 10,
                'eval_deterministic': True
            },
            
            # Saving parameters
            'saving': {
                'save_freq': 50000,
                'save_path': 'models/ppo_portfolio',
                'keep_best_only': True
            },
            
            # Logging
            'logging': {
                'tensorboard': True,
                'wandb': False,
                'log_interval': 100
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    loaded_config = yaml.safe_load(f)
                else:
                    loaded_config = json.load(f)
            
            # Merge with defaults
            self._deep_update(default_config, loaded_config)
            logger.info(f"Configuration loaded from {config_path}")
        else:
            logger.info("Using default configuration")
            
        return default_config
    
    def _deep_update(self, base_dict: dict, update_dict: dict):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def setup_directories(self):
        """Create necessary directories."""
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        self.model_dir = Path(f"models/ppo_portfolio_{self.timestamp}")
        self.log_dir = Path(f"logs/ppo_portfolio_{self.timestamp}")
        self.results_dir = Path(f"results/ppo_portfolio_{self.timestamp}")
        
        for directory in [self.model_dir, self.log_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Directories created: {self.model_dir}")
        
    def create_env(self, rank: int = 0):
        """Create a single environment instance."""
        def _init():
            env = PortfolioEnv(
                data_path=self.data_path,
                **self.config['env']
            )
            env = Monitor(env, str(self.log_dir / f"monitor_{rank}.csv"))
            return env
        return _init
    
    def create_vec_env(self):
        """Create vectorized environment."""
        n_envs = self.config['training']['n_envs']
        use_multiprocessing = self.config['training']['use_multiprocessing']
        
        if use_multiprocessing and n_envs > 1:
            env = SubprocVecEnv([self.create_env(i) for i in range(n_envs)])
        else:
            env = DummyVecEnv([self.create_env(i) for i in range(n_envs)])
            
        logger.info(f"Created vectorized environment with {n_envs} instances")
        return env
    
    def create_model(self, env):
        """Create PPO model."""
        training_config = self.config['training']
        
        # PPO parameters
        model_params = {
            'learning_rate': training_config['learning_rate'],
            'n_steps': training_config['n_steps'],
            'batch_size': training_config['batch_size'],
            'n_epochs': training_config['n_epochs'],
            'gamma': training_config['gamma'],
            'gae_lambda': training_config['gae_lambda'],
            'clip_range': training_config['clip_range'],
            'ent_coef': training_config['ent_coef'],
            'vf_coef': training_config['vf_coef'],
            'max_grad_norm': training_config['max_grad_norm'],
            'target_kl': training_config.get('target_kl'),
            'tensorboard_log': str(self.log_dir) if self.config['logging']['tensorboard'] else None,
            'seed': self.seed,
            'verbose': 1
        }
        
        # Create model
        model = PPO("MlpPolicy", env, **model_params)
        
        logger.info("PPO model created with parameters:")
        for key, value in model_params.items():
            logger.info(f"  {key}: {value}")
            
        return model
    
    def create_callbacks(self, env):
        """Create training callbacks."""
        callbacks = []
        
        # Evaluation callback
        eval_config = self.config['evaluation']
        if eval_config['eval_freq'] > 0:
            eval_env = DummyVecEnv([self.create_env()])
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir),
                log_path=str(self.log_dir),
                eval_freq=eval_config['eval_freq'],
                n_eval_episodes=eval_config['n_eval_episodes'],
                deterministic=eval_config['eval_deterministic'],
                render=False
            )
            callbacks.append(eval_callback)
            logger.info(f"Evaluation callback added (freq: {eval_config['eval_freq']})")
        
        # Checkpoint callback
        save_config = self.config['saving']
        if save_config['save_freq'] > 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=save_config['save_freq'],
                save_path=str(self.model_dir),
                name_prefix='ppo_portfolio'
            )
            callbacks.append(checkpoint_callback)
            logger.info(f"Checkpoint callback added (freq: {save_config['save_freq']})")
        
        return CallbackList(callbacks) if callbacks else None
    
    def save_config(self):
        """Save configuration to results directory."""
        config_path = self.results_dir / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting PPO training...")
        
        # Save configuration
        self.save_config()
        
        # Create environment
        env = self.create_vec_env()
        
        # Create model
        model = self.create_model(env)
        
        # Create callbacks
        callbacks = self.create_callbacks(env)
        
        # Start training
        total_timesteps = self.config['training']['total_timesteps']
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                log_interval=self.config['logging']['log_interval'],
                progress_bar=True
            )
            
            # Save final model
            final_model_path = self.model_dir / 'final_model'
            model.save(final_model_path)
            
            logger.info(f"Training completed! Final model saved to {final_model_path}")
            
            # Save training statistics
            self.save_training_stats(model)
            
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save current model
            interrupted_model_path = self.model_dir / 'interrupted_model'
            model.save(interrupted_model_path)
            logger.info(f"Model saved to {interrupted_model_path}")
        
        finally:
            env.close()
            logger.info("Environment closed")
            
        return model
    
    def save_training_stats(self, model):
        """Save training statistics."""
        stats = {
            'total_timesteps': model.num_timesteps,
            'training_completed': True,
            'model_path': str(self.model_dir),
            'config': self.config,
            'timestamp': self.timestamp
        }
        
        stats_path = self.results_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
            
        logger.info(f"Training statistics saved to {stats_path}")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Train PPO for Portfolio Optimization')
    parser.add_argument('--data-path', type=str, required=False,
                       help='Path to the parquet data file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--sentiment-path', type=str, default=None,
                       help='Optional: Path to sentiment CSV file (date, asset, sentiment)')
    
    args = parser.parse_args()
    
    
    trainer = PortfolioTrainer(
        config_path=args.config, 
        data_path=args.data_path,
        sentiment_path=args.sentiment_path
    )
    
    # Override seed if provided
    if args.seed != 42:
        trainer.config['seed'] = args.seed
        trainer.seed = args.seed
    
    # NOW verify data file exists (after trainer has determined the final data_path)
    if not os.path.exists(trainer.data_path):
        raise FileNotFoundError(f"Data file not found: {trainer.data_path}")
    
    # Start training
    model = trainer.train()
    
    logger.info("Training script completed!")

if __name__ == "__main__":
    main()