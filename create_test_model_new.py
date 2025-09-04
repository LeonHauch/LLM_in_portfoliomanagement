# create_test_model.py
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Add src to path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Path to save the test model
MODEL_PATH = "models/sample_model.zip"

# --- Minimal portfolio environment (no changes needed) ---
class MinimalPortfolioEnv(gym.Env):
    def __init__(self, n_assets=5, lookback=10):
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback
        self.current_step = 0
        self.max_steps = 100

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(n_assets,), dtype=np.float32)
        obs_dim = n_assets * lookback + n_assets
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        np.random.seed(42)
        self.returns_data = np.random.normal(0.001, 0.02, (1000, n_assets))
        self.current_weights = np.ones(n_assets) / n_assets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_observation(), {}

    def step(self, action):
        action = np.abs(action)
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            action = np.ones(self.n_assets) / self.n_assets

        returns = self.returns_data[self.current_step]
        portfolio_return = np.dot(action, returns)
        turnover = np.sum(np.abs(action - self.current_weights))
        transaction_cost = 0.001 * turnover
        reward = portfolio_return - transaction_cost
        
        self.current_weights = action.copy()
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "portfolio_value": 100000 * (1 + reward),
            "weights": self.current_weights.tolist(),
            "net_return": reward
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step
        recent_returns = self.returns_data[start_idx:end_idx]
        if len(recent_returns) < self.lookback:
            padding = np.zeros((self.lookback - len(recent_returns), self.n_assets))
            recent_returns = np.vstack([padding, recent_returns])
        obs = np.concatenate([recent_returns.flatten(), self.current_weights])
        return obs.astype(np.float32)

# --- Create test model ---
def create_test_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        print(f"✅ Test model already exists at {model_path}, skipping creation.")
        return True

    print("Creating test model...")
    
    # 💥 FIX: Set a specific number of threads for PyTorch
    import torch
    torch.set_num_threads(1) 
    print(f"PyTorch using {torch.get_num_threads()} threads.")
    
    device = "cpu"
    print(f"Using device: {device}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    env = make_vec_env(lambda: MinimalPortfolioEnv(), n_envs=1)
    
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=64, batch_size=32, n_epochs=4, verbose=1, device=device)
    
    try:
        print("Starting training loop...")
        model.learn(total_timesteps=1000)
        print("Training finished!")
        model.save(model_path)
        print(f"✅ Test model saved to: {model_path}")

        loaded_model = PPO.load(model_path, device=device)
        obs = env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        print(f"✅ Model can make predictions: {action}")
        
    except Exception as e:
        print(f"❌ An error occurred during training or testing: {e}")
        return False
        
    finally:
        env.close()
    
    return True

if __name__ == "__main__":
    success = create_test_model()
    if success:
        print("\n🎉 Test model ready!")
    else:
        print("\n❌ Failed to create test model.")