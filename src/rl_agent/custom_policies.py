# src/rl_agent/custom_policies.py
"""
Custom feature extractors for Stable-Baselines3 PPO.

The default MlpPolicy flattens the observation and ignores its temporal
structure. These extractors reshape the flat vector back into a
(lookback × assets × features) tensor and apply architectures with
appropriate inductive biases.
"""
import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict


class PortfolioCNNExtractor(BaseFeaturesExtractor):
    """
    1-D CNN over the time axis of a (lookback, n_assets, n_features) tensor.

    Architecture:
        Input: (batch, lookback, n_assets * n_features)   [treat assets*features as channels]
        → Conv1d(time_axis) → ReLU → Conv1d → ReLU → AdaptiveAvgPool → Flatten
        → Linear → ReLU
        + portfolio weights branch concatenated
        → Linear → output embedding

    This gives the network an inductive bias for temporal patterns
    while keeping the model small enough for PPO on CPU.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_assets: int,
        n_features_per_asset: int,
        lookback_window: int,
        features_dim: int = 128,
    ):
        # features_dim is the final output dimensionality
        super().__init__(observation_space, features_dim)

        self.n_assets = n_assets
        self.n_features = n_features_per_asset
        self.lookback = lookback_window

        # Number of portfolio-weight features appended at the end
        obs_total = observation_space.shape[0]
        self.n_portfolio_features = obs_total - (lookback_window * n_assets * n_features_per_asset)

        in_channels = n_assets * n_features_per_asset

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → (batch, 64, 1)
            nn.Flatten(),             # → (batch, 64)
        )

        # Combine CNN embedding + portfolio weights
        combined_dim = 64 + self.n_portfolio_features
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]

        # Split: market data vs portfolio weights
        market_flat = observations[:, : self.lookback * self.n_assets * self.n_features]
        port_weights = observations[:, self.lookback * self.n_assets * self.n_features :]

        # Reshape to (batch, lookback, n_assets * n_features)
        market = market_flat.view(batch, self.lookback, self.n_assets * self.n_features)

        # Conv1d expects (batch, channels, length) → transpose
        market = market.permute(0, 2, 1)  # (batch, n_assets*n_features, lookback)

        cnn_out = self.cnn(market)  # (batch, 64)
        combined = torch.cat([cnn_out, port_weights], dim=1)
        return self.fc(combined)


class PortfolioLSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM over the time axis, processing each timestep's (n_assets * n_features)
    vector sequentially.

    Architecture:
        Input: (batch, lookback, n_assets * n_features)
        → LSTM → take last hidden state
        + portfolio weights branch
        → Linear → output embedding
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_assets: int,
        n_features_per_asset: int,
        lookback_window: int,
        features_dim: int = 128,
        lstm_hidden: int = 64,
    ):
        super().__init__(observation_space, features_dim)

        self.n_assets = n_assets
        self.n_features = n_features_per_asset
        self.lookback = lookback_window

        obs_total = observation_space.shape[0]
        self.n_portfolio_features = obs_total - (lookback_window * n_assets * n_features_per_asset)

        input_size = n_assets * n_features_per_asset
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        combined_dim = lstm_hidden + self.n_portfolio_features
        self.fc = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]

        market_flat = observations[:, : self.lookback * self.n_assets * self.n_features]
        port_weights = observations[:, self.lookback * self.n_assets * self.n_features :]

        market = market_flat.view(batch, self.lookback, self.n_assets * self.n_features)

        _, (h_n, _) = self.lstm(market)  # h_n: (1, batch, hidden)
        lstm_out = h_n.squeeze(0)  # (batch, hidden)

        combined = torch.cat([lstm_out, port_weights], dim=1)
        return self.fc(combined)


def get_policy_kwargs(
    policy_type: str,
    n_assets: int,
    n_features_per_asset: int,
    lookback_window: int,
    observation_space: spaces.Box,
    features_dim: int = 128,
) -> Dict:
    """
    Build the policy_kwargs dict for SB3's PPO constructor.

    Args:
        policy_type: One of "mlp", "cnn", "lstm".
        n_assets: Number of tradeable assets.
        n_features_per_asset: Features per asset per timestep.
        lookback_window: Temporal lookback.
        observation_space: The env's observation space.
        features_dim: Embedding dimension for actor/critic heads.

    Returns:
        Dict suitable for PPO(..., policy_kwargs=...).
    """
    if policy_type == "mlp":
        return {"net_arch": [dict(pi=[256, 128], vf=[256, 128])]}

    if policy_type == "cnn":
        return {
            "features_extractor_class": PortfolioCNNExtractor,
            "features_extractor_kwargs": {
                "n_assets": n_assets,
                "n_features_per_asset": n_features_per_asset,
                "lookback_window": lookback_window,
                "features_dim": features_dim,
            },
            "net_arch": [dict(pi=[128], vf=[128])],
        }

    if policy_type == "lstm":
        return {
            "features_extractor_class": PortfolioLSTMExtractor,
            "features_extractor_kwargs": {
                "n_assets": n_assets,
                "n_features_per_asset": n_features_per_asset,
                "lookback_window": lookback_window,
                "features_dim": features_dim,
            },
            "net_arch": [dict(pi=[128], vf=[128])],
        }

    raise ValueError(f"Unknown policy_type: {policy_type!r}. Use 'mlp', 'cnn', or 'lstm'.")
