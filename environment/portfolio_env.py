# environment/portfolio_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """
    Portfolio management environment for PPO training.

    Simulates portfolio allocation decisions across multiple assets
    based on historical price and volume data. Supports temporal
    train/test splitting, optional sentiment features, and structured
    observations for CNN/LSTM policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        lookback_window: int = 60,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        max_positions: Optional[int] = None,
        cash_weight: bool = True,
        normalize_observations: bool = True,
        reward_scaling: float = 1000.0,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_sentiment: bool = False,
        risk_bonus_weight: float = 0.0,
    ):
        """
        Args:
            data_path: Path to the parquet file with preprocessed data.
            lookback_window: Number of historical periods in observation.
            initial_capital: Starting portfolio value.
            transaction_cost: Cost as fraction of traded value.
            max_positions: Maximum simultaneous positions (None = no limit).
            cash_weight: Whether to include cash as an allocable asset.
            normalize_observations: Clip observations to [-10, 10].
            reward_scaling: Multiplicative factor applied to reward.
            start_idx: First usable row index (inclusive). Overridden by start_date.
            end_idx: Last usable row index (inclusive). Overridden by end_date.
            start_date: Earliest date string (inclusive, ISO format) for this split.
            end_date: Latest date string (inclusive, ISO format) for this split.
            use_sentiment: If True, include Sentiment_* columns in observations.
            risk_bonus_weight: Weight for rolling-Sharpe reward bonus (0 = pure return).
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
        self.use_sentiment = use_sentiment
        self.risk_bonus_weight = risk_bonus_weight

        # Train/test boundaries (resolved after data load)
        self._start_idx_param = start_idx
        self._end_idx_param = end_idx
        self._start_date_param = start_date
        self._end_date_param = end_date

        # Load and prepare data
        self._load_data()
        self._resolve_date_range()
        self._setup_spaces()

        # Initialize state
        self.reset()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _load_data(self):
        """Load parquet and derive features."""
        logger.info(f"Loading data from {self.data_path}")
        self.raw_data = pd.read_parquet(self.data_path)

        # Extract asset columns
        adj_close_cols = sorted(
            [c for c in self.raw_data.columns if c.startswith("Adj Close_")]
        )
        volume_ratio_cols = sorted(
            [c for c in self.raw_data.columns if c.startswith("Volume_Ratio_")]
        )

        self.asset_symbols = [c.replace("Adj Close_", "") for c in adj_close_cols]
        self.n_assets = len(self.asset_symbols)
        logger.info(f"Found {self.n_assets} assets: {self.asset_symbols[:5]}...")

        # Price & volume dataframes with simple column names
        self.price_data = self.raw_data[adj_close_cols].copy()
        self.price_data.columns = self.asset_symbols

        self.volume_data = pd.DataFrame(index=self.raw_data.index)
        for sym in self.asset_symbols:
            col = f"Volume_Ratio_{sym}"
            if col in self.raw_data.columns:
                self.volume_data[sym] = self.raw_data[col]
            else:
                self.volume_data[sym] = 0.0

        # Log returns: r_t = ln(P_t / P_{t-1})
        self.returns_data = np.log(self.price_data / self.price_data.shift(1)).fillna(0)

        # Technical features
        self._calculate_features()

        # Sentiment features (optional)
        self.has_sentiment = False
        if self.use_sentiment:
            sent_cols = sorted(
                [c for c in self.raw_data.columns if c.startswith("Sentiment_")]
            )
            if sent_cols:
                self.sentiment_data = self.raw_data[sent_cols].copy()
                self.sentiment_data.columns = [
                    c.replace("Sentiment_", "") for c in sent_cols
                ]
                # Align to asset list, fill missing with 0
                self.sentiment_data = self.sentiment_data.reindex(
                    columns=self.asset_symbols, fill_value=0.0
                )
                self.has_sentiment = True
                logger.info(f"Sentiment features loaded for {len(sent_cols)} assets")
            else:
                logger.warning("use_sentiment=True but no Sentiment_* columns found")

        logger.info(f"Data loaded: {len(self.raw_data)} periods")

    def _calculate_features(self):
        """Derive rolling volatility, correlation, and SMA ratio."""
        self.volatility_data = self.returns_data.rolling(window=20).std().fillna(0)

        # Equal-weight market proxy
        market_returns = self.returns_data.mean(axis=1)
        self.correlation_data = pd.DataFrame(
            index=self.returns_data.index, columns=self.asset_symbols, dtype=float
        )
        for asset in self.asset_symbols:
            self.correlation_data[asset] = (
                self.returns_data[asset]
                .rolling(window=20)
                .corr(market_returns)
                .fillna(0)
            )

        sma_20 = self.price_data.rolling(window=20).mean()
        self.sma_ratio_data = (self.price_data / sma_20).fillna(1.0)

    # ------------------------------------------------------------------
    # Train / test split resolution
    # ------------------------------------------------------------------
    def _resolve_date_range(self):
        """Convert date/idx parameters into valid_start_idx and valid_end_idx."""
        data_len = len(self.raw_data)

        # Start from date if provided
        if self._start_date_param is not None and hasattr(self.raw_data.index, "get_loc"):
            try:
                idx = self.raw_data.index.get_indexer(
                    [pd.Timestamp(self._start_date_param)], method="bfill"
                )[0]
                self._start_idx_param = max(idx, self.lookback_window)
            except Exception:
                logger.warning(f"Could not resolve start_date={self._start_date_param}")
        elif self._start_date_param is not None:
            # Index might be integer-based; try matching a date column
            logger.warning("start_date provided but index is not datetime; ignoring")

        if self._end_date_param is not None and hasattr(self.raw_data.index, "get_loc"):
            try:
                idx = self.raw_data.index.get_indexer(
                    [pd.Timestamp(self._end_date_param)], method="ffill"
                )[0]
                self._end_idx_param = min(idx, data_len - 1)
            except Exception:
                logger.warning(f"Could not resolve end_date={self._end_date_param}")

        # Fall back to defaults
        self.valid_start_idx = (
            max(self._start_idx_param, self.lookback_window)
            if self._start_idx_param is not None
            else self.lookback_window
        )
        self.valid_end_idx = (
            min(self._end_idx_param, data_len - 1)
            if self._end_idx_param is not None
            else data_len - 1
        )

        if self.valid_end_idx <= self.valid_start_idx:
            raise ValueError(
                f"Invalid data range: start={self.valid_start_idx}, "
                f"end={self.valid_end_idx}"
            )

        logger.info(
            f"Date range resolved: idx [{self.valid_start_idx}, {self.valid_end_idx}] "
            f"({self.valid_end_idx - self.valid_start_idx} tradable periods)"
        )

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Base features per asset: returns, volume, volatility, correlation, sma_ratio
        self.n_features_per_asset = 5
        if self.has_sentiment:
            self.n_features_per_asset += 1  # sentiment score

        portfolio_features = self.n_assets + (1 if self.cash_weight else 0)
        total_obs_dim = (
            self.lookback_window * self.n_assets * self.n_features_per_asset
            + portfolio_features
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
        )

        action_dim = self.n_assets + (1 if self.cash_weight else 0)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        logger.info(
            f"Obs space: {self.observation_space.shape}, "
            f"Action space: {self.action_space.shape}"
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def _get_observation(self) -> np.ndarray:
        """Build flat observation vector."""
        start = self.current_step - self.lookback_window
        end = self.current_step

        features: List[float] = []
        for i in range(start, end):
            for asset in self.asset_symbols:
                asset_feats = [
                    self.returns_data.iloc[i][asset],
                    self.volume_data.iloc[i][asset],
                    self.volatility_data.iloc[i][asset],
                    float(self.correlation_data.iloc[i][asset]),
                    self.sma_ratio_data.iloc[i][asset],
                ]
                if self.has_sentiment:
                    asset_feats.append(self.sentiment_data.iloc[i][asset])
                features.extend(asset_feats)

        features.extend(self.current_weights)
        obs = np.array(features, dtype=np.float32)

        if self.normalize_observations:
            obs = np.clip(obs, -10, 10)

        return obs

    # ------------------------------------------------------------------
    # Action processing
    # ------------------------------------------------------------------
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Convert raw network output to portfolio weights via softmax."""
        action = np.asarray(action, dtype=np.float64).squeeze()
        if action.ndim == 0:
            action = action.reshape(1)

        # Numerically-stable softmax
        action = np.clip(action, -50, 50)
        exp_a = np.exp(action - np.max(action))
        weights = exp_a / exp_a.sum()

        if self.max_positions is not None and self.max_positions < len(weights):
            top = np.argsort(weights)[-self.max_positions :]
            mask = np.zeros_like(weights)
            mask[top] = weights[top]
            weights = mask / mask.sum()

        return weights

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _calculate_reward(self, new_weights: np.ndarray) -> float:
        """Net log-return minus transaction costs, with optional risk bonus."""
        current_returns = self.returns_data.iloc[self.current_step][
            self.asset_symbols
        ].values.astype(np.float64)

        if self.cash_weight:
            asset_weights = new_weights[:-1]
        else:
            asset_weights = new_weights

        portfolio_return = float(np.dot(asset_weights, current_returns))

        weight_changes = np.abs(new_weights - self.current_weights)
        tc = float(np.sum(weight_changes)) * self.transaction_cost
        net_return = portfolio_return - tc

        self.portfolio_value *= 1 + net_return

        reward = net_return

        # Optional risk-adjusted bonus (off by default: risk_bonus_weight=0)
        if self.risk_bonus_weight > 0 and len(self.returns_history) > 20:
            recent = np.array(self.returns_history[-20:])
            std = np.std(recent)
            if std > 0:
                reward += self.risk_bonus_weight * (np.mean(recent) / std)

        reward *= self.reward_scaling
        self.returns_history.append(net_return)
        return reward

    # ------------------------------------------------------------------
    # Step / Reset
    # ------------------------------------------------------------------
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action)
        if action.ndim > 1 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 1:
            raise ValueError(
                f"Unexpected action ndim {action.ndim}, shape: {action.shape}"
            )

        new_weights = self._process_action(action)
        reward = self._calculate_reward(new_weights)
        self.current_weights = new_weights.copy()
        self.current_step += 1

        terminated = self.current_step >= self.valid_end_idx
        truncated = False

        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights.copy(),
            "step": self.current_step,
            "net_return": self.returns_history[-1] if self.returns_history else 0.0,
        }
        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)

        # Random start within valid range, leaving room for at least 100 steps
        hi = max(self.valid_start_idx + 1, self.valid_end_idx - 100)
        self.current_step = np.random.randint(self.valid_start_idx, hi)

        self.portfolio_value = self.initial_capital
        action_dim = self.n_assets + (1 if self.cash_weight else 0)
        self.current_weights = np.ones(action_dim) / action_dim
        self.returns_history: List[float] = []

        obs = self._get_observation()
        info = {
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights.copy(),
            "step": self.current_step,
        }
        return obs, info

    def render(self, mode: str = "human"):
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Weights: {self.current_weights}")
            if self.returns_history:
                print(f"Last Return: {self.returns_history[-1]:.6f}")
            print("-" * 50)
