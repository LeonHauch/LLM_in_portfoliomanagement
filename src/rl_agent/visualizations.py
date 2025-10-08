# src/rl_agent/visualizations.py
"""
Visualization utilities for portfolio evaluation results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PortfolioVisualizer:
    """Create visualizations for portfolio backtest results."""
    
    def __init__(self, results_dir: Path, env_config: Dict, env_info: Dict):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory to save plots
            env_config: Environment configuration dict
            env_info: Environment information dict
        """
        self.results_dir = Path(results_dir)
        self.env_config = env_config
        self.env_info = env_info
        
        # Set style
        sns.set_style("whitegrid")
    
    def create_all_visualizations(self, backtest_results: Dict, save_plots: bool = True):
        """
        Create all standard visualizations from backtest results.
        
        Args:
            backtest_results: Dictionary containing backtest data
            save_plots: Whether to save plots to disk
        """
        logger.info("Creating visualizations...")
        
        try:
            # Create main performance plots
            self.create_performance_plots(backtest_results, save_plots)
            
            # Create weights visualization if available
            weights_history = backtest_results.get('weights_history', [])
            if weights_history and len(weights_history) > 0:
                self.create_weights_plot(weights_history, save_plots)
            
            logger.info("✅ Visualizations created successfully")
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
            import traceback
            traceback.print_exc()
    
    def create_performance_plots(self, backtest_results: Dict, save_plots: bool = True):
        """Create main performance visualization plots."""
        portfolio_values = np.array(backtest_results.get('portfolio_values', []))
        returns_history = backtest_results.get('returns_history', [])
        
        if len(portfolio_values) == 0:
            logger.warning("No portfolio values to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1. Portfolio Value Over Time
        axes[0].plot(portfolio_values, linewidth=2, color='#2E86AB')
        initial_capital = self.env_config.get('initial_capital', 100000)
        axes[0].axhline(y=initial_capital, color='red', linestyle='--', 
                       alpha=0.5, label='Initial Capital')
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time Steps')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'${x:,.0f}')
        )
        
        # 2. Returns Distribution
        if len(returns_history) > 0:
            returns_array = np.array(returns_history)
            returns_array = returns_array[np.isfinite(returns_array)]
            
            axes[1].hist(returns_array, bins=50, color='#A23B72', 
                        alpha=0.7, edgecolor='black')
            axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
            axes[1].axvline(x=np.mean(returns_array), color='green', 
                           linestyle='--', alpha=0.5, 
                           label=f'Mean: {np.mean(returns_array):.4f}')
            axes[1].set_title('Returns Distribution', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Returns')
            axes[1].set_ylabel('Frequency')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        # 3. Cumulative Returns
        if len(portfolio_values) > 1:
            cumulative_returns = ((portfolio_values - portfolio_values[0]) / 
                                 portfolio_values[0] * 100)
            axes[2].plot(cumulative_returns, linewidth=2, color='#F18F01')
            axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[2].set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Time Steps')
            axes[2].set_ylabel('Cumulative Return (%)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        if save_plots:
            plot_path = self.results_dir / 'backtest_performance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Performance plot saved to: {plot_path}")
        
        plt.close()
    
    def create_weights_plot(self, weights_history: List, save_plots: bool = True):
        """Create portfolio weights stacked area chart."""
        try:
            weights_array = np.array(weights_history)
            
            # Validate shape
            if weights_array.ndim != 2 or weights_array.shape[0] == 0:
                logger.warning("Invalid weights history shape for visualization")
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create stacked area chart
            time_steps = range(len(weights_array))
            
            # Get asset names
            asset_names = self.env_info.get('asset_symbols', [])
            if self.env_config.get('cash_weight', False):
                asset_names = list(asset_names) + ['Cash']
            
            # Ensure we have the right number of labels
            n_weights = weights_array.shape[1]
            if len(asset_names) < n_weights:
                asset_names = (list(asset_names) + 
                             [f'Asset_{i}' for i in range(len(asset_names), n_weights)])
            
            # Plot stacked area
            ax.stackplot(time_steps, weights_array.T, 
                        labels=asset_names[:n_weights], 
                        alpha=0.7)
            
            ax.set_title('Portfolio Weights Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Weight')
            ax.set_ylim([0, 1])
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                plot_path = self.results_dir / 'portfolio_weights.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"📊 Weights plot saved to: {plot_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"❌ Weights visualization failed: {e}")


def create_visualizations(backtest_results: Dict, results_dir: Path, 
                         env_config: Dict, env_info: Dict, 
                         save_plots: bool = True):
    """
    Convenience function to create all visualizations.
    
    Args:
        backtest_results: Dictionary containing backtest data
        results_dir: Directory to save plots
        env_config: Environment configuration
        env_info: Environment information
        save_plots: Whether to save plots to disk
    """
    visualizer = PortfolioVisualizer(results_dir, env_config, env_info)
    visualizer.create_all_visualizations(backtest_results, save_plots)
    
def compare_with_benchmark(backtest_results: Dict, results_dir: Path,
                          trading_days_per_year: int = 252,
                          benchmark_return: float = 0.07) -> Dict:
    """
    Compare portfolio performance with a benchmark.
    
    Args:
        backtest_results: Dictionary containing backtest data
        results_dir: Directory to save comparison results
        trading_days_per_year: Number of trading days per year
        benchmark_return: Annual benchmark return (default 7% for market average)
        
    Returns:
        Dictionary with comparison metrics
    """
    import json
    
    logger.info(f"Comparing with benchmark (annual return: {benchmark_return:.2%})...")
    
    try:
        # Extract portfolio data
        portfolio_values = np.array(backtest_results.get('portfolio_values', []))
        total_steps = backtest_results.get('total_steps', len(portfolio_values) - 1)
        
        if len(portfolio_values) < 2:
            logger.warning("Insufficient data for benchmark comparison")
            return {'error': 'Insufficient data'}
        
        # Calculate portfolio metrics
        initial_value = float(portfolio_values[0])
        final_value = float(portfolio_values[-1])
        portfolio_return = (final_value - initial_value) / initial_value
        
        # Annualize returns (assuming daily steps)
        years = total_steps / trading_days_per_year
        
        if years > 0:
            annualized_portfolio_return = (1 + portfolio_return) ** (1/years) - 1
        else:
            annualized_portfolio_return = portfolio_return
        
        # Calculate benchmark final value
        benchmark_final_value = initial_value * (1 + benchmark_return) ** years
        benchmark_total_return = (benchmark_final_value - initial_value) / initial_value
        
        # Outperformance
        outperformance = annualized_portfolio_return - benchmark_return
        outperformance_pct = (outperformance / benchmark_return * 100) if benchmark_return != 0 else 0
        
        # Calculate Sharpe ratio if available
        returns_history = backtest_results.get('returns_history', [])
        if len(returns_history) > 1:
            returns_array = np.array(returns_history)
            returns_array = returns_array[np.isfinite(returns_array)]
            
            portfolio_sharpe = backtest_results.get('sharpe_ratio', 
                (np.mean(returns_array) / np.std(returns_array) * np.sqrt(trading_days_per_year)) 
                if np.std(returns_array) > 0 else 0)
        else:
            portfolio_sharpe = 0
        
        comparison = {
            'portfolio': {
                'total_return': portfolio_return,
                'annualized_return': annualized_portfolio_return,
                'final_value': final_value,
                'sharpe_ratio': portfolio_sharpe
            },
            'benchmark': {
                'annual_return': benchmark_return,
                'total_return': benchmark_total_return,
                'final_value': benchmark_final_value
            },
            'comparison': {
                'outperformance': outperformance,
                'outperformance_pct': outperformance_pct,
                'value_difference': final_value - benchmark_final_value,
                'years': years
            }
        }
        
        # Log results
        logger.info(f"📊 Benchmark Comparison Results:")
        logger.info(f"  Portfolio annualized return: {annualized_portfolio_return:.2%}")
        logger.info(f"  Benchmark annualized return: {benchmark_return:.2%}")
        logger.info(f"  Outperformance: {outperformance:.2%} ({outperformance_pct:+.1f}%)")
        logger.info(f"  Portfolio final value: ${final_value:,.2f}")
        logger.info(f"  Benchmark final value: ${benchmark_final_value:,.2f}")
        logger.info(f"  Value difference: ${final_value - benchmark_final_value:+,.2f}")
        
        # Save comparison to file
        results_dir = Path(results_dir)
        comparison_path = results_dir / 'benchmark_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"💾 Comparison saved to: {comparison_path}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"❌ Benchmark comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}