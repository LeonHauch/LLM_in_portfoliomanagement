# src/evaluation/eval_ppo_debug.py
import os
import time
import signal
import numpy as np
import torch 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional

# Import your custom environment
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from environment.portfolio_env import PortfolioEnv

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

class PortfolioEvaluatorDebug:
    """Evaluation class for trained PPO portfolio models with debugging."""
    
    def __init__(self, model_path: str, data_path: str, results_dir: str = None, 
                 device: str = 'auto', trading_days_per_year: int = 252, 
                 env_config: Dict = None, seed: int = 42, debug: bool = True):
        """Initialize the evaluator with debugging capabilities."""
        
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.trading_days_per_year = trading_days_per_year
        self.seed = seed
        self.debug = debug
        
        if self.debug:
            logger.info("🐛 DEBUG MODE ENABLED")
        
        # Environment configuration
        self.env_config = env_config or {
            'lookback_window': 60,
            'initial_capital': 100000.0,
            'transaction_cost': 0.001,
            'cash_weight': True,
            'normalize_observations': True,
            'reward_scaling': 1000.0
        }
        
        # Setup results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = Path(results_dir or f"results/evaluation_{timestamp}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing evaluator...")
        logger.info(f"  Model path: {self.model_path}")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Results dir: {self.results_dir}")
        
        # Load model with timeout
        self.model = self._load_model_with_timeout()
        
        # Test environment creation
        self.env_info = self._test_environment_creation()
        
        logger.info(f"✅ Evaluator initialized successfully")
        logger.info(f"  Device: {self.model.device}")
        logger.info(f"  Environment info: {self.env_info}")
    
    def _load_model_with_timeout(self, timeout_seconds: int = 30):
        """Load model with timeout to catch hanging."""
        logger.info("Loading model...")
        start_time = time.time()
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Handle device specification
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
        
        try:
            if self.debug:
                # Set timeout for model loading
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            model = PPO.load(self.model_path, device=device)
            
            if self.debug:
                signal.alarm(0)  # Cancel timeout
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.2f}s on device: {device}")
            
            return model
            
        except TimeoutError:
            logger.error(f"❌ Model loading timed out after {timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise
    
    def _test_environment_creation(self):
        """Test environment creation and get info."""
        logger.info("Testing environment creation...")
        start_time = time.time()
        
        try:
            # Test single environment creation
            env = PortfolioEnv(data_path=self.data_path, **self.env_config)
            
            logger.info("✅ Environment created successfully")
            logger.info(f"  Assets: {env.n_assets}")
            logger.info(f"  Data periods: {len(env.raw_data)}")
            logger.info(f"  Valid range: {env.valid_start_idx} to {env.valid_end_idx}")
            logger.info(f"  Observation space: {env.observation_space.shape}")
            logger.info(f"  Action space: {env.action_space.shape}")
            
            # Test reset
            obs, info = env.reset(seed=self.seed)
            logger.info(f"✅ Environment reset successful")
            logger.info(f"  Observation shape: {obs.shape}")
            logger.info(f"  Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
            
            env_info = {
                'n_assets': env.n_assets,
                'n_periods': len(env.raw_data),
                'valid_range': (env.valid_start_idx, env.valid_end_idx),
                'obs_shape': env.observation_space.shape,
                'action_shape': env.action_space.shape,
                'asset_symbols': env.asset_symbols[:5]  # First 5 for logging
            }
            
            env.close()
            
            creation_time = time.time() - start_time
            logger.info(f"✅ Environment test completed in {creation_time:.2f}s")
            
            return env_info
            
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _create_eval_env(self, seed_offset: int = 0, timeout_seconds: int = 10):
        """Create environment for evaluation with timeout."""
        if self.debug:
            logger.info(f"Creating eval environment (seed offset: {seed_offset})")
        
        try:
            if self.debug and timeout_seconds > 0:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
            
            env = PortfolioEnv(data_path=self.data_path, **self.env_config)
            
            # Create monitor
            monitor_path = str(self.results_dir / f"eval_monitor_{seed_offset}.csv")
            env = Monitor(env, monitor_path)
            
            # Seed the environment
            env.reset(seed=self.seed + seed_offset)
            
            if self.debug and timeout_seconds > 0:
                signal.alarm(0)  # Cancel timeout
            
            return env
            
        except TimeoutError:
            logger.error(f"❌ Environment creation timed out after {timeout_seconds}s")
            raise
        except Exception as e:
            logger.error(f"❌ Environment creation failed: {e}")
            raise
    
    def evaluate_episodes_debug(self, n_episodes: int = 10, deterministic: bool = True, 
                               max_steps_per_episode: int = 1000) -> Dict:
        """
        Evaluate with detailed debugging and progress tracking.
        """
        logger.info(f"Starting evaluation: {n_episodes} episodes, deterministic={deterministic}")
        
        # Use single environment to avoid parallel issues
        env = self._create_eval_env()
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        final_weights_list = []
        
        start_time = time.time()
        
        for episode in range(n_episodes):
            episode_start = time.time()
            logger.info(f"Episode {episode + 1}/{n_episodes} starting...")
            
            try:
                # Reset environment
                obs, info = env.reset(seed=self.seed + episode)
                
                episode_reward = 0
                episode_length = 0
                step_count = 0
                
                done = False
                while not done and step_count < max_steps_per_episode:
                    # Get action from model
                    obs_vec = obs.reshape(1, -1)  # Add batch dimension
                    action, _ = self.model.predict(obs_vec, deterministic=deterministic)
                    action = action[0]  # Remove batch dimension
                    
                    # Take step
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    
                    episode_reward += reward
                    episode_length += 1
                    step_count += 1
                    
                    # Progress feedback every 100 steps
                    if self.debug and step_count % 100 == 0:
                        portfolio_val = step_info.get('portfolio_value', 0)
                        logger.info(f"  Step {step_count}: reward={reward:.4f}, portfolio=${portfolio_val:,.0f}")
                
                # Store episode results
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                
                final_info = step_info if 'step_info' in locals() else {}
                portfolio_values.append(final_info.get('portfolio_value', self.env_config.get('initial_capital', 100000)))
                final_weights_list.append(final_info.get('weights', []))
                
                episode_time = time.time() - episode_start
                logger.info(f"✅ Episode {episode + 1} completed in {episode_time:.2f}s")
                logger.info(f"  Reward: {episode_reward:.4f}, Length: {episode_length}, Portfolio: ${portfolio_values[-1]:,.2f}")
                
            except Exception as e:
                logger.error(f"❌ Episode {episode + 1} failed: {e}")
                # Add default values to maintain list consistency
                episode_rewards.append(0.0)
                episode_lengths.append(0)
                portfolio_values.append(self.env_config.get('initial_capital', 100000))
                final_weights_list.append([])
        
        env.close()
        
        # Calculate metrics
        if not episode_rewards:
            logger.error("No episodes completed successfully")
            return {'error': 'No episodes completed'}
        
        total_time = time.time() - start_time
        
        metrics = {
            'n_episodes': len(episode_rewards),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values) if portfolio_values else 0,
            'std_portfolio_value': np.std(portfolio_values) if portfolio_values else 0,
            'success_rate': np.mean(np.array(portfolio_values) > self.env_config.get('initial_capital', 100000)),
            'total_evaluation_time': total_time,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'portfolio_values': portfolio_values,
            'final_weights': final_weights_list
        }
        
        logger.info(f"✅ Evaluation completed in {total_time:.2f}s:")
        logger.info(f"  Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        logger.info(f"  Mean portfolio: ${metrics['mean_portfolio_value']:,.2f}")
        logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
        
        return metrics
    
    def quick_backtest(self, max_steps: int = 100, deterministic: bool = True) -> Dict:
        """
        Run a quick backtest with limited steps for debugging.
        """
        logger.info(f"Starting quick backtest (max {max_steps} steps)...")
        start_time = time.time()
        
        env = self._create_eval_env()
        
        try:
            # Reset environment
            obs, info = env.reset(seed=self.seed)
            
            portfolio_values = [self.env_config.get('initial_capital', 100000)]
            weights_history = []
            returns_history = []
            
            step = 0
            done = False
            
            while not done and step < max_steps:
                # Get action
                obs_vec = obs.reshape(1, -1)
                action, _ = self.model.predict(obs_vec, deterministic=deterministic)
                action = action[0]
                
                # Take step
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                
                # Record data
                portfolio_value = step_info.get('portfolio_value', portfolio_values[-1])
                weights = step_info.get('weights', [])
                net_return = step_info.get('net_return', 0.0)
                
                portfolio_values.append(portfolio_value)
                weights_history.append(weights.copy() if weights else [])
                returns_history.append(net_return)
                
                step += 1
                
                if self.debug and step % 20 == 0:
                    logger.info(f"  Quick backtest step {step}: ${portfolio_value:,.2f}")
            
            env.close()
            
            # Quick metrics
            portfolio_values = np.array(portfolio_values)
            if len(portfolio_values) < 2:
                logger.error("Insufficient data for backtest")
                return {'error': 'Insufficient backtest data'}
            
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            returns = returns[np.isfinite(returns)]
            
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            
            backtest_results = {
                'total_steps': step,
                'initial_value': float(portfolio_values[0]),
                'final_value': float(portfolio_values[-1]),
                'total_return': float(total_return),
                'mean_return': float(np.mean(returns)) if len(returns) > 0 else 0,
                'volatility': float(np.std(returns)) if len(returns) > 0 else 0,
                'portfolio_values': portfolio_values.tolist(),
                'weights_history': weights_history,
                'returns_history': returns_history,
                'backtest_time': time.time() - start_time
            }
            
            logger.info(f"✅ Quick backtest completed in {backtest_results['backtest_time']:.2f}s:")
            logger.info(f"  Steps: {step}")
            logger.info(f"  Total return: {total_return:.2%}")
            logger.info(f"  Final value: ${portfolio_values[-1]:,.2f}")
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"❌ Quick backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def run_diagnostic(self):
        """Run comprehensive diagnostic checks."""
        logger.info("🔍 Running diagnostic checks...")
        
        results = {
            'model_info': {},
            'environment_info': self.env_info,
            'compatibility_test': {},
            'performance_test': {}
        }
        
        # Model diagnostics
        try:
            results['model_info'] = {
                'device': str(self.model.device),
                'policy': type(self.model.policy).__name__,
                'n_envs': getattr(self.model, 'n_envs', 1)
            }
            logger.info("✅ Model diagnostics completed")
        except Exception as e:
            logger.error(f"❌ Model diagnostics failed: {e}")
            results['model_info']['error'] = str(e)
        
        # Compatibility test
        try:
            env = self._create_eval_env()
            obs, _ = env.reset(seed=self.seed)
            
            # Test model prediction
            obs_vec = obs.reshape(1, -1)
            action, _ = self.model.predict(obs_vec, deterministic=True)
            action = action[0]
            
            # Test environment step
            obs2, reward, terminated, truncated, info = env.step(action)
            
            env.close()
            
            results['compatibility_test'] = {
                'status': 'success',
                'obs_shape': obs.shape,
                'action_shape': action.shape,
                'reward': float(reward),
                'info_keys': list(info.keys())
            }
            logger.info("✅ Compatibility test passed")
            
        except Exception as e:
            logger.error(f"❌ Compatibility test failed: {e}")
            results['compatibility_test'] = {'status': 'failed', 'error': str(e)}
        
        # Performance test
        try:
            start_time = time.time()
            metrics = self.evaluate_episodes_debug(n_episodes=2, deterministic=True)
            perf_time = time.time() - start_time
            
            results['performance_test'] = {
                'status': 'success',
                'time': perf_time,
                'mean_reward': metrics.get('mean_reward', 0)
            }
            logger.info(f"✅ Performance test completed in {perf_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Performance test failed: {e}")
            results['performance_test'] = {'status': 'failed', 'error': str(e)}
        
        # Save diagnostic results
        diag_path = self.results_dir / 'diagnostics.json'
        with open(diag_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"🔍 Diagnostic results saved to: {diag_path}")
        return results

def main():
    """Main function with debugging options."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO portfolio model (DEBUG VERSION)')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data file')
    parser.add_argument('--results-dir', type=str, default=None, help='Results directory')
    parser.add_argument('--n-episodes', type=int, default=5, help='Number of episodes (reduced for debugging)')
    parser.add_argument('--quick-test', action='store_true', help='Run quick test only')
    parser.add_argument('--diagnostic-only', action='store_true', help='Run diagnostics only')
    parser.add_argument('--max-steps', type=int, default=100, help='Max steps for quick test')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        return 1
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return 1
    
    try:
        # Create evaluator
        logger.info("🚀 Starting debug evaluation...")
        evaluator = PortfolioEvaluatorDebug(
            model_path=args.model_path,
            data_path=args.data_path,
            results_dir=args.results_dir,
            device=args.device,
            seed=args.seed,
            debug=True
        )
        
        if args.diagnostic_only:
            logger.info("Running diagnostics only...")
            results = evaluator.run_diagnostic()
            logger.info("✅ Diagnostics completed!")
            return 0
        
        if args.quick_test:
            logger.info("Running quick test...")
            results = evaluator.quick_backtest(max_steps=args.max_steps)
            if 'error' in results:
                logger.error(f"Quick test failed: {results['error']}")
                return 1
            logger.info("✅ Quick test completed!")
        else:
            logger.info("Running full evaluation...")
            results = evaluator.evaluate_episodes_debug(n_episodes=args.n_episodes)
            if 'error' in results:
                logger.error(f"Evaluation failed: {results['error']}")
                return 1
            logger.info("✅ Evaluation completed!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())