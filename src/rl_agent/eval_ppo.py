# src/evaluation/eval_ppo.py
import os
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioEvaluator:
    """Evaluation class for trained PPO portfolio models."""
    
    def __init__(self, model_path: str, data_path: str, results_dir: str = None, 
                 device: str = 'auto', trading_days_per_year: int = 252, 
                 env_config: Dict = None, seed: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained PPO model
            data_path: Path to the parquet data file
            results_dir: Directory to save evaluation results
            device: Device for model inference ('auto', 'cpu', 'cuda')
            trading_days_per_year: Number of trading days per year for annualization
            env_config: Environment configuration override
            seed: Random seed for reproducibility
        """
        self.model_path = model_path
        self.data_path = data_path
        self.device = device
        self.trading_days_per_year = trading_days_per_year
        self.seed = seed
        
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
        
        # Load model
        self.model = self._load_model()
        
        # Create single environment for sequential evaluation
        self.env = self._create_eval_env()
        
        logger.info(f"Evaluator initialized. Results will be saved to: {self.results_dir}")
        logger.info(f"Using device: {self.model.device}")
        logger.info(f"Trading days per year: {self.trading_days_per_year}")
        
    def _load_model(self):
        """Load the trained PPO model with proper device handling."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        # Handle device specification
        if self.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = self.device
            
        model = PPO.load(self.model_path, device=device)
        logger.info(f"Model loaded from: {self.model_path} on device: {device}")
        return model
    
    def _create_eval_env(self, seed_offset: int = 0):
        """Create environment for evaluation with proper seeding."""
        env = PortfolioEnv(
            data_path=self.data_path,
            **self.env_config
        )
        
        # Create monitor with seeding
        monitor_path = str(self.results_dir / f"eval_monitor_{seed_offset}.csv")
        env = Monitor(env, monitor_path)
        
        # Seed the environment
        env.reset(seed=self.seed + seed_offset)
        
        return env
    
    def _create_vec_env(self, n_envs: int = 1):
        """Create vectorized environment for parallel evaluation."""
        def make_env(rank: int):
            def _init():
                return self._create_eval_env(seed_offset=rank)
            return _init
        
        if n_envs == 1:
            return DummyVecEnv([make_env(0)])
        else:
            return DummyVecEnv([make_env(i) for i in range(n_envs)])
    
    def evaluate_episodes(self, n_episodes: int = 100, deterministic: bool = True, 
                         n_envs: int = 1) -> Dict:
        """
        Evaluate the model over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            n_envs: Number of parallel environments for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Starting evaluation over {n_episodes} episodes with {n_envs} parallel envs...")
        
        # Create vectorized environment for parallel evaluation
        if n_envs > 1:
            vec_env = self._create_vec_env(n_envs)
        else:
            vec_env = DummyVecEnv([lambda: self._create_eval_env()])
        
        episode_rewards = []
        episode_lengths = []
        portfolio_values = []
        final_weights_list = []
        all_actions = []
        
        episodes_completed = 0
        current_episode_rewards = [0] * n_envs
        current_episode_lengths = [0] * n_envs
        current_episode_actions = [[] for _ in range(n_envs)]
        
        obs = vec_env.reset()
        
        while episodes_completed < n_episodes:
            actions, _ = self.model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(actions)
            
            # Update episode tracking
            for env_idx in range(n_envs):
                current_episode_rewards[env_idx] += rewards[env_idx]
                current_episode_lengths[env_idx] += 1
                current_episode_actions[env_idx].append(actions[env_idx].copy())
                
                if dones[env_idx]:
                    episodes_completed += 1
                    
                    # Store episode results
                    episode_rewards.append(current_episode_rewards[env_idx])
                    episode_lengths.append(current_episode_lengths[env_idx])
                    
                    # Safely extract info with defaults
                    info = infos[env_idx] if env_idx < len(infos) else {}
                    portfolio_values.append(info.get('portfolio_value', self.env_config.get('initial_capital', 100000)))
                    final_weights_list.append(info.get('weights', []))
                    all_actions.append(current_episode_actions[env_idx].copy())
                    
                    # Reset tracking for this env
                    current_episode_rewards[env_idx] = 0
                    current_episode_lengths[env_idx] = 0
                    current_episode_actions[env_idx] = []
                    
                    if episodes_completed % 10 == 0:
                        logger.info(f"Completed {episodes_completed}/{n_episodes} episodes")
                    
                    if episodes_completed >= n_episodes:
                        break
        
        vec_env.close()
        
        # Calculate metrics with safe operations
        if not episode_rewards:
            logger.warning("No episodes completed successfully")
            return {'error': 'No episodes completed'}
        
        metrics = {
            'n_episodes': len(episode_rewards),
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_portfolio_value': np.mean(portfolio_values) if portfolio_values else 0,
            'std_portfolio_value': np.std(portfolio_values) if portfolio_values else 0,
            'min_portfolio_value': np.min(portfolio_values) if portfolio_values else 0,
            'max_portfolio_value': np.max(portfolio_values) if portfolio_values else 0,
            'success_rate': (np.mean(np.array(portfolio_values) > self.env_config.get('initial_capital', 100000)) 
                           if portfolio_values else 0),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'portfolio_values': portfolio_values,
            'final_weights': final_weights_list,
            'all_actions': all_actions
        }
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Episodes completed: {metrics['n_episodes']}")
        logger.info(f"  Mean reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}")
        logger.info(f"  Mean portfolio value: ${metrics['mean_portfolio_value']:,.2f}")
        logger.info(f"  Success rate: {metrics['success_rate']:.2%}")
        
        return metrics
    
    def backtest(self, start_date: str = None, end_date: str = None, 
                deterministic: bool = True) -> Dict:
        """
        Run a full backtest on historical data.
        
        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)  
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest...")
        
        # Use single environment for backtest
        env = self._create_eval_env()
        
        # Reset environment for full episode
        obs, info = env.reset(seed=self.seed)
        
        # Track backtest data
        portfolio_values = [self.env_config.get('initial_capital', 100000)]
        weights_history = []
        actions_history = []
        returns_history = []
        
        step = 0
        done = False
        
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=deterministic)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Safely extract info with defaults
            portfolio_value = info.get('portfolio_value', portfolio_values[-1])
            weights = info.get('weights', [])
            net_return = info.get('net_return', 0.0)
            
            # Record data
            portfolio_values.append(portfolio_value)
            weights_history.append(weights.copy() if weights else [])
            actions_history.append(action.copy())
            returns_history.append(net_return)
            
            step += 1
            
            if step % 100 == 0:
                logger.info(f"Backtest step {step}, Portfolio value: ${portfolio_value:,.2f}")
        
        env.close()
        
        # Calculate performance metrics with proper error handling
        portfolio_values = np.array(portfolio_values)
        
        if len(portfolio_values) < 2:
            logger.error("Insufficient data for backtest analysis")
            return {'error': 'Insufficient backtest data'}
        
        # Calculate returns
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[np.isfinite(returns)]  # Remove inf/nan values
        
        if len(returns) == 0:
            logger.error("No valid returns calculated")
            return {'error': 'No valid returns'}
        
        # Performance metrics with proper annualization
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Use actual number of steps and trading days per year
        if len(returns) > 0:
            periods_per_year = self.trading_days_per_year / len(returns) * len(returns)
            annualized_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(periods_per_year / len(returns))
        else:
            annualized_return = 0
            volatility = 0
        
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns)
        
        # Win rate
        positive_returns = np.sum(returns > 0) if len(returns) > 0 else 0
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0
        
        backtest_results = {
            'total_steps': step,
            'initial_value': float(portfolio_values[0]),
            'final_value': float(portfolio_values[-1]),
            'total_return': float(total_return),
            'annualized_return': float(annualized_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'trading_days_used': len(returns),
            'annualization_factor': periods_per_year / len(returns) if len(returns) > 0 else 0,
            'portfolio_values': portfolio_values.tolist(),
            'weights_history': weights_history,
            'actions_history': actions_history,
            'returns_history': returns_history,
            'returns': returns.tolist(),
            'drawdowns': drawdowns.tolist()
        }
        
        logger.info("Backtest completed:")
        logger.info(f"  Total steps: {step}")
        logger.info(f"  Trading periods used: {len(returns)}")
        logger.info(f"  Annualization factor: {backtest_results['annualization_factor']:.4f}")
        logger.info(f"  Total return: {total_return:.2%}")
        logger.info(f"  Annualized return: {annualized_return:.2%}")
        logger.info(f"  Volatility: {volatility:.2%}")
        logger.info(f"  Sharpe ratio: {sharpe_ratio:.4f}")
        logger.info(f"  Max drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win rate: {win_rate:.2%}")
        
        return backtest_results
    
    def create_visualizations(self, backtest_results: Dict, save_plots: bool = True):
        """
        Create visualization plots for the backtest results.
        
        Args:
            backtest_results: Results from backtest() method
            save_plots: Whether to save plots to disk
        """
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Portfolio Value Over Time
        portfolio_values = backtest_results['portfolio_values']
        axes[0, 0].plot(portfolio_values, linewidth=2, color='blue', alpha=0.8)
        axes[0, 0].axhline(y=backtest_results['initial_value'], color='gray', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Drawdown
        drawdowns = backtest_results['drawdowns']
        axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, 0, 
                               color='red', alpha=0.6, label='Drawdown')
        axes[0, 1].set_title('Drawdown Over Time', fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Drawdown (%)')
        axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns Distribution
        returns = backtest_results['returns']
        axes[1, 0].hist(returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].axvline(x=np.mean(returns), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(returns):.4f}')
        axes[1, 0].set_title('Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Daily Returns')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Portfolio Weights Over Time (sample)
        weights_history = backtest_results['weights_history']
        if weights_history and len(weights_history) > 0:
            # Sample every 10th observation to avoid overcrowding
            sample_indices = range(0, len(weights_history), max(1, len(weights_history) // 100))
            sampled_weights = [weights_history[i] for i in sample_indices]
            
            if sampled_weights and len(sampled_weights[0]) > 0:
                weights_array = np.array(sampled_weights)
                
                # Plot top 5 assets by average weight
                avg_weights = np.mean(weights_array, axis=0)
                top_assets = np.argsort(avg_weights)[-5:]
                
                for i, asset_idx in enumerate(top_assets):
                    axes[1, 1].plot(sample_indices, weights_array[:, asset_idx], 
                                   label=f'Asset {asset_idx}', alpha=0.8)
                
                axes[1, 1].set_title('Top 5 Asset Weights Over Time', fontweight='bold')
                axes[1, 1].set_xlabel('Time Steps')
                axes[1, 1].set_ylabel('Weight')
                axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.results_dir / 'portfolio_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plots saved to: {plot_path}")
        
        plt.show()
        
        # Create additional detailed weight analysis
        self._create_weight_heatmap(backtest_results, save_plots)
    
    def _create_weight_heatmap(self, backtest_results: Dict, save_plots: bool = True):
        """Create heatmap of portfolio weights over time."""
        weights_history = backtest_results['weights_history']
        
        if not weights_history or len(weights_history) == 0:
            logger.warning("No weights history available for heatmap")
            return
        
        # Sample weights for visualization (every 20th observation)
        sample_freq = max(1, len(weights_history) // 50)
        sampled_weights = weights_history[::sample_freq]
        
        if len(sampled_weights) == 0:
            return
            
        weights_df = pd.DataFrame(sampled_weights)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(weights_df.T, cmap='RdYlBu_r', cbar_kws={'label': 'Weight'})
        plt.title('Portfolio Weights Heatmap Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps (Sampled)')
        plt.ylabel('Asset Index')
        
        if save_plots:
            heatmap_path = self.results_dir / 'weights_heatmap.png'
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to: {heatmap_path}")
        
        plt.show()
    
    def compare_with_benchmark(self, backtest_results: Dict, benchmark_return: float = 0.07):
        """
        Compare portfolio performance with a benchmark.
        
        Args:
            backtest_results: Results from backtest()
            benchmark_return: Annual benchmark return (default: 7%)
        """
        portfolio_return = backtest_results['annualized_return']
        portfolio_sharpe = backtest_results['sharpe_ratio']
        
        # Simple benchmark metrics (assuming same volatility for simplicity)
        benchmark_sharpe = benchmark_return / backtest_results['volatility'] if backtest_results['volatility'] > 0 else 0
        
        comparison = {
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'excess_return': portfolio_return - benchmark_return,
            'portfolio_sharpe': portfolio_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'information_ratio': (portfolio_return - benchmark_return) / backtest_results['volatility'] if backtest_results['volatility'] > 0 else 0
        }
        
        logger.info("Benchmark Comparison:")
        logger.info(f"  Portfolio Return: {portfolio_return:.2%}")
        logger.info(f"  Benchmark Return: {benchmark_return:.2%}")
        logger.info(f"  Excess Return: {comparison['excess_return']:.2%}")
        logger.info(f"  Portfolio Sharpe: {portfolio_sharpe:.4f}")
        logger.info(f"  Information Ratio: {comparison['information_ratio']:.4f}")
        
        return comparison
    
    def save_results(self, metrics: Dict, backtest_results: Dict, comparison: Dict = None):
        """Save all evaluation results to files."""
        # Save metrics
        metrics_path = self.results_dir / 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, np.ndarray):
                    json_metrics[key] = value.tolist()
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    json_metrics[key] = [arr.tolist() if isinstance(arr, np.ndarray) else arr for arr in value]
                else:
                    json_metrics[key] = value
            json.dump(json_metrics, f, indent=2, default=str)
        
        # Save backtest results
        backtest_path = self.results_dir / 'backtest_results.json'
        with open(backtest_path, 'w') as f:
            json.dump(backtest_results, f, indent=2, default=str)
        
        # Save comparison if provided
        if comparison:
            comparison_path = self.results_dir / 'benchmark_comparison.json'
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report(metrics, backtest_results, comparison)
        
        logger.info(f"Results saved to: {self.results_dir}")
    
    def _create_summary_report(self, metrics: Dict, backtest_results: Dict, comparison: Dict = None):
        """Create a human-readable summary report."""
        report_path = self.results_dir / 'summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("PPO Portfolio Optimization - Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Model: {self.model_path}\n")
            f.write(f"Data: {self.data_path}\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Episode metrics
            f.write("Episode Evaluation Metrics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of Episodes: {metrics['n_episodes']}\n")
            f.write(f"Mean Reward: {metrics['mean_reward']:.4f} ± {metrics['std_reward']:.4f}\n")
            f.write(f"Mean Portfolio Value: ${metrics['mean_portfolio_value']:,.2f}\n")
            f.write(f"Success Rate: {metrics['success_rate']:.2%}\n\n")
            
            # Backtest metrics
            f.write("Backtest Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Initial Value: ${backtest_results['initial_value']:,.2f}\n")
            f.write(f"Final Value: ${backtest_results['final_value']:,.2f}\n")
            f.write(f"Total Return: {backtest_results['total_return']:.2%}\n")
            f.write(f"Annualized Return: {backtest_results['annualized_return']:.2%}\n")
            f.write(f"Volatility: {backtest_results['volatility']:.2%}\n")
            f.write(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}\n")
            f.write(f"Maximum Drawdown: {backtest_results['max_drawdown']:.2%}\n")
            f.write(f"Win Rate: {backtest_results['win_rate']:.2%}\n\n")
            
            if comparison:
                f.write("Benchmark Comparison:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Portfolio Return: {comparison['portfolio_return']:.2%}\n")
                f.write(f"Benchmark Return: {comparison['benchmark_return']:.2%}\n")
                f.write(f"Excess Return: {comparison['excess_return']:.2%}\n")
                f.write(f"Information Ratio: {comparison['information_ratio']:.4f}\n")
        
        logger.info(f"Summary report saved to: {report_path}")

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO portfolio model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the trained PPO model')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to the parquet data file')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Directory to save results')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of episodes for evaluation')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments for evaluation')
    parser.add_argument('--benchmark-return', type=float, default=0.07,
                       help='Annual benchmark return for comparison')
    parser.add_argument('--trading-days-per-year', type=int, default=252,
                       help='Number of trading days per year for annualization')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for model inference')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating plots')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
    
    # Create evaluator
    evaluator = PortfolioEvaluator(
        model_path=args.model_path,
        data_path=args.data_path,
        results_dir=args.results_dir,
        device=args.device,
        trading_days_per_year=args.trading_days_per_year,
        seed=args.seed
    )
    
    # Run evaluation with parallel environments
    metrics = evaluator.evaluate_episodes(
        n_episodes=args.n_episodes,
        n_envs=args.n_envs
    )
    
    if 'error' in metrics:
        logger.error(f"Evaluation failed: {metrics['error']}")
        return
    
    # Run backtest
    backtest_results = evaluator.backtest()
    
    if 'error' in backtest_results:
        logger.error(f"Backtest failed: {backtest_results['error']}")
        return
    
    # Compare with benchmark
    comparison = evaluator.compare_with_benchmark(
        backtest_results, 
        benchmark_return=args.benchmark_return
    )
    
    # Create visualizations
    if not args.no_plots:
        evaluator.create_visualizations(backtest_results)
    
    # Save all results
    evaluator.save_results(metrics, backtest_results, comparison)
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main()