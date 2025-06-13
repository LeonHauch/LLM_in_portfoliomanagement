# src/environment/portfolio_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PortfolioEnv(gym.Env):
    """
    Portfolio management environment for PPO training.
    
    This environment simulates portfolio allocation decisions across multiple assets
    based on historical price and volume data.
    """
    
    def __init__(
        self,
        data_path: str,
        lookback_window: int = 60,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        max_positions: Optional[int] = None,
        cash_weight: bool = True,
        normalize_observations: bool = True,
        reward_scaling: float = 1000.0
    ):
        """
        Initialize the portfolio environment.
        
        Args:
            data_path: Path to the parquet file with preprocessed data
            lookback_window: Number of historical periods to include in observation
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as percentage of trade value
            max_positions: Maximum number of positions (None for no limit)
            cash_weight: Whether to include cash as an asset option
            normalize_observations: Whether to normalize observation features
            reward_scaling: Scaling factor for rewards
        """
        super().__init__()
        
        self.data_path = data_path
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_positions = max_positions
        self.cash_weight = cash_weight
        self.normalize_observations = normalize_observations
        self.reward_scaling = reward_scaling
        
        # Load and prepare data
        self._load_data()
        self._setup_spaces()
        
        # Initialize state
        self.reset()
        
    def _load_data(self):
        """Load and process the parquet data."""
        logger.info(f"Loading data from {self.data_path}")
        
        # Load the parquet file
        self.raw_data = pd.read_parquet(self.data_path)
        
        # Extract asset names from column names
        adj_close_cols = [col for col in self.raw_data.columns if col.startswith('Adj Close_')]
        volume_ratio_cols = [col for col in self.raw_data.columns if col.startswith('Volume_Ratio_')]
        
        # Extract asset symbols
        self.asset_symbols = [col.replace('Adj Close_', '') for col in adj_close_cols]
        self.n_assets = len(self.asset_symbols)
        
        logger.info(f"Found {self.n_assets} assets: {self.asset_symbols[:5]}...")
        
        # Organize data by asset
        self.price_data = self.raw_data[adj_close_cols].copy()
        self.volume_data = self.raw_data[volume_ratio_cols].copy()
        
        # Rename columns to just asset symbols
        self.price_data.columns = self.asset_symbols
        self.volume_data.columns = self.asset_symbols
        
        # Calculate returns
        self.returns_data = self.price_data.pct_change().fillna(0)
        
        # Calculate additional features
        self._calculate_features()
        
        # Set valid date range (excluding lookback period)
        self.valid_start_idx = self.lookback_window
        self.valid_end_idx = len(self.raw_data) - 1
        
        logger.info(f"Data loaded: {len(self.raw_data)} periods, valid range: {self.valid_start_idx} to {self.valid_end_idx}")
        
    def _calculate_features(self):
        """Calculate additional technical features."""
        # Rolling volatility (20-period)
        self.volatility_data = self.returns_data.rolling(window=20).std().fillna(0)
        
        # Rolling correlation with market (using first asset as proxy)
        market_returns = self.returns_data.iloc[:, 0]
        self.correlation_data = pd.DataFrame(index=self.returns_data.index, columns=self.asset_symbols)
        
        for asset in self.asset_symbols:
            self.correlation_data[asset] = self.returns_data[asset].rolling(window=20).corr(market_returns).fillna(0)
        
        # Simple moving averages ratio (price / SMA20)
        sma_20 = self.price_data.rolling(window=20).mean()
        self.sma_ratio_data = (self.price_data / sma_20).fillna(1.0)
        
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: [lookback_window, n_features_per_asset]
        # Features per asset: [returns, volume_ratio, volatility, correlation, sma_ratio]
        self.n_features_per_asset = 5
        
        if self.normalize_observations:
            obs_low = -np.inf
            obs_high = np.inf
        else:
            obs_low = -10.0  # Reasonable bounds for financial data
            obs_high = 10.0
            
        # Add current portfolio weights to observation
        portfolio_features = self.n_assets + (1 if self.cash_weight else 0)
        total_obs_features = self.lookback_window * self.n_assets * self.n_features_per_asset + portfolio_features
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(total_obs_features,),
            dtype=np.float32
        )
        
        # Action space: portfolio weights (sum to 1)
        # If cash_weight=True, include cash as an option
        action_dim = self.n_assets + (1 if self.cash_weight else 0)
        
        # Use Box space with post-processing to ensure weights sum to 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
        
        logger.info(f"Observation space: {self.observation_space.shape}")
        logger.info(f"Action space: {self.action_space.shape}")
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Get lookback window of features
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        # Collect features for all assets over lookback window
        features = []
        
        for i in range(start_idx, end_idx):
            for asset in self.asset_symbols:
                asset_features = [
                    self.returns_data.iloc[i][asset],
                    self.volume_data.iloc[i][asset],
                    self.volatility_data.iloc[i][asset],
                    self.correlation_data.iloc[i][asset],
                    self.sma_ratio_data.iloc[i][asset]
                ]
                features.extend(asset_features)
        
        # Add current portfolio weights
        features.extend(self.current_weights)
        
        obs = np.array(features, dtype=np.float32)
        
        # Normalize if requested
        if self.normalize_observations:
            obs = np.clip(obs, -10, 10)  # Clip extreme values
            
        return obs
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process raw action to valid portfolio weights."""
        # Apply softmax to ensure weights sum to 1 and are non-negative
        action = np.clip(action, -10, 10)  # Prevent overflow
        exp_action = np.exp(action)
        weights = exp_action / np.sum(exp_action)
        
        # Handle max positions constraint
        if self.max_positions is not None and self.max_positions < len(weights):
            # Keep only top max_positions weights, set others to 0
            top_indices = np.argsort(weights)[-self.max_positions:]
            new_weights = np.zeros_like(weights)
            new_weights[top_indices] = weights[top_indices]
            new_weights = new_weights / np.sum(new_weights)  # Renormalize
            weights = new_weights
            
        return weights
    
    def _calculate_reward(self, new_weights: np.ndarray) -> float:
        """Calculate reward for the current step."""
        # Get current period returns
        current_returns = self.returns_data.iloc[self.current_step][self.asset_symbols].values
        
        # Calculate portfolio return
        if self.cash_weight:
            # Last weight is cash (0% return)
            asset_weights = new_weights[:-1]
            portfolio_return = np.dot(asset_weights, current_returns)
        else:
            portfolio_return = np.dot(new_weights, current_returns)
        
        # Calculate transaction costs
        weight_changes = np.abs(new_weights - self.current_weights)
        transaction_costs = np.sum(weight_changes) * self.transaction_cost
        
        # Net return after transaction costs
        net_return = portfolio_return - transaction_costs
        
        # Update portfolio value
        self.portfolio_value *= (1 + net_return)
        
        # Base reward is the net return
        reward = net_return
        
        # Add risk-adjusted component (Sharpe-like)
        if len(self.returns_history) > 20:
            recent_returns = np.array(self.returns_history[-20:])
            if np.std(recent_returns) > 0:
                risk_adjusted_reward = np.mean(recent_returns) / np.std(recent_returns)
                reward += 0.1 * risk_adjusted_reward  # Small risk adjustment bonus
        
        # Scale reward
        reward *= self.reward_scaling
        
        # Track returns
        self.returns_history.append(net_return)
        
        return reward
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Process action to get valid portfolio weights  
        new_weights = self._process_action(action)
        
        # Calculate reward
        reward = self._calculate_reward(new_weights)
        
        # Update current weights
        self.current_weights = new_weights.copy()
        
        # Move to next time step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= self.valid_end_idx
        truncated = False
        
        # Get new observation
        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Info dictionary
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'step': self.current_step,
            'net_return': self.returns_history[-1] if self.returns_history else 0.0
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)
        
        # Reset to random starting point in valid range
        self.current_step = np.random.randint(self.valid_start_idx, 
                                            max(self.valid_start_idx + 1, self.valid_end_idx - 100))
        
        # Initialize portfolio
        self.portfolio_value = self.initial_capital
        
        # Initialize weights (equal weight or cash-heavy)
        action_dim = self.n_assets + (1 if self.cash_weight else 0)
        if self.cash_weight:
            # Start with more cash
            self.current_weights = np.ones(action_dim) * 0.1
            self.current_weights[-1] = 0.9 - np.sum(self.current_weights[:-1])  # Remaining to cash
        else:
            self.current_weights = np.ones(self.n_assets) / self.n_assets
        
        # Reset tracking variables
        self.returns_history = []
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'portfolio_value': self.portfolio_value,
            'weights': self.current_weights.copy(),
            'step': self.current_step
        }
        
        return obs, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (optional)."""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Current Weights: {self.current_weights}")
            print(f"Returns History Length: {len(self.returns_history)}")
            if self.returns_history:
                print(f"Last Return: {self.returns_history[-1]:.4f}")
            print("-" * 50)