import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from src.rl_agent import eval_ppo

@pytest.fixture
def dummy_paths(tmp_path: Path):
    """Fixture to provide a temporary model path and the real data path."""
    model_path = tmp_path / "models.sample_model.zip"
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'sample', 'data_ppo_sample.parquet'
    )
    model_path.write_text("")
    return str(model_path), str(data_path)

def test_evaluate_and_backtest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dummy_paths: tuple[str, str]):
    model_path, data_path = dummy_paths
    
    # 1. Load REAL data to understand structure
    print(f"Loading real data from: {data_path}")
    try:
        real_data = pd.read_parquet(data_path)
        print(f"Data shape: {real_data.shape}")
        print(f"Data columns: {real_data.columns.tolist()}")
        print(f"Data index: {real_data.index}")
    except Exception as e:
        print(f"Could not load real data: {e}")
        pytest.skip(f"Could not load test data: {e}")
    
    # 2. Introspect REAL environment to get correct structure
    try:
        from environment.portfolio_env import PortfolioEnv
        real_env = PortfolioEnv(data_path=data_path)
        
        # Get observation and action space info
        obs_result = real_env.reset(seed=42)
        if isinstance(obs_result, tuple):  # New Gymnasium API
            obs, info = obs_result
        else:  # Old Gym API
            obs = obs_result
            
        obs_shape = obs.shape
        action_space_size = real_env.action_space.shape[0]
        
        # Extract environment parameters
        asset_symbols = getattr(real_env, 'asset_symbols', [])
        cash_weight = getattr(real_env, 'cash_weight', False)
        initial_capital = getattr(real_env, 'initial_capital', 100000.0)
        
        real_env.close()
        
        print(f"Environment introspection:")
        print(f"  - Observation shape: {obs_shape}")
        print(f"  - Action space size: {action_space_size}")
        print(f"  - Asset symbols: {asset_symbols}")
        print(f"  - Cash weight: {cash_weight}")
        print(f"  - Initial capital: {initial_capital}")
        
    except Exception as e:
        print(f"Could not introspect real environment: {e}")
        # Use fallback values
        obs_shape = (10,)
        action_space_size = 5
        asset_symbols = ['ASSET1', 'ASSET2', 'ASSET3', 'ASSET4']
        cash_weight = True
        initial_capital = 100000.0
    
    # 3. Create Dummy Model that matches expected action size
    class DummyModel:
        device = "cpu"
        
        def predict(self, obs, deterministic=True):
            # Return action array with correct size and batch dimension (1 env)
            action = np.full((1, action_space_size), 1.0 / action_space_size)
            print(f"DummyModel predict: obs_shape={obs.shape if hasattr(obs, 'shape') else 'unknown'}, action_shape={action.shape}")
            return action, None
    
    # Mock PPO.load to return our dummy
    monkeypatch.setattr("src.rl_agent.eval_ppo.PPO", 
                       type("PPO", (), {"load": staticmethod(lambda *args, **kwargs: DummyModel())}))
    
    # 4. Create Smart Dummy Environment using real data structure
    class SmartDummyEvalEnv(gym.Env):
        def __init__(self, data_path=None, render_mode=None, **kwargs):
            super().__init__()
            self.steps = 0
            self.data_path = data_path
            self.max_steps = 5  # Short episodes for testing
            
            # Load real data for realistic structure
            if data_path and os.path.exists(data_path):
                self.real_data = pd.read_parquet(data_path)
                self.n_data_points = len(self.real_data)
                print(f"SmartDummyEnv: Loaded {self.n_data_points} data points")
            else:
                self.n_data_points = 100
            
            # Store expected parameters
            self.action_space_size = action_space_size
            self.obs_shape = obs_shape
            self.asset_symbols = asset_symbols
            self.cash_weight = cash_weight
            self.initial_capital = initial_capital
            
            # Define action and observation spaces
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_size,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            obs = np.random.random(self.obs_shape).astype(np.float32)
            info = {"reset": True}
            return obs, info
        
        def step(self, action):
            self.steps += 1
            terminated = self.steps >= self.max_steps
            truncated = False
            
            print(f"SmartDummyEnv.step: step={self.steps}, action_shape={action.shape if hasattr(action, 'shape') else 'scalar'}")
            
            # Simulate realistic portfolio evolution
            portfolio_value = self.initial_capital * (1 + 0.001 * self.steps)
            
            # Convert action to weights list
            if hasattr(action, 'tolist'):
                weights = action.tolist()
            elif hasattr(action, '__iter__'):
                weights = list(action)
            else:
                weights = [float(action)]  # Handle scalar case
            
            # Ensure correct weights length
            expected_weights_len = len(self.asset_symbols) + (1 if self.cash_weight else 0)
            if len(weights) != expected_weights_len:
                print(f"WARNING: Adjusting weights from {len(weights)} to {expected_weights_len}")
                weights = weights[:expected_weights_len] + [0] * max(0, expected_weights_len - len(weights))
            
            info = {
                "portfolio_value": portfolio_value,
                "weights": weights,
                "step": self.steps
            }
            
            next_obs = np.random.random(self.obs_shape).astype(np.float32)
            reward = float(1.0 + 0.1 * np.random.randn())  # Small random reward
            
            return next_obs, reward, terminated, truncated, info
        
        def close(self):
            pass
    
    # 5. Backtest version with different return format
    class SmartDummyBacktestEnv(gym.Env):
        def __init__(self, data_path=None, render_mode=None, **kwargs):
            super().__init__()
            self.steps = 0
            self.data_path = data_path
            self.max_steps = 10  # Longer for backtest
        
            # Load real data for realistic structure
            if data_path and os.path.exists(data_path):
                self.real_data = pd.read_parquet(data_path)
                self.n_data_points = len(self.real_data)
                print(f"SmartDummyEnv: Loaded {self.n_data_points} data points")
            else:
                self.n_data_points = 100
            
            # Store expected parameters
            self.action_space_size = action_space_size
            self.obs_shape = obs_shape
            self.asset_symbols = asset_symbols
            self.cash_weight = cash_weight
            self.initial_capital = initial_capital
            
            # Define action and observation spaces
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_space_size,), dtype=np.float32)
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.steps = 0
            obs = np.random.random(self.obs_shape).astype(np.float32)
            info = {"reset": True}
            return obs, info
        
        def step(self, action):
            self.steps += 1
            terminated = self.steps >= self.max_steps
            truncated = False  # Separate terminated/truncated
            
            # More realistic portfolio progression for backtest
            portfolio_value = self.initial_capital * (1 + 0.01 * self.steps)
            
            if hasattr(action, 'tolist'):
                weights = action.tolist()
            elif hasattr(action, '__iter__'):
                weights = list(action)
            else:
                weights = [float(action)]
            
            # Ensure correct weights length
            expected_weights_len = len(self.asset_symbols) + (1 if self.cash_weight else 0)
            if len(weights) != expected_weights_len:
                weights = weights[:expected_weights_len] + [0] * max(0, expected_weights_len - len(weights))
            
            info = {
                "portfolio_value": portfolio_value,
                "weights": weights,
                "net_return": 0.01 + 0.001 * np.random.randn(),  # Realistic daily return
                "step": self.steps
            }
            
            next_obs = np.random.random(self.obs_shape).astype(np.float32)
            reward = info["net_return"] * 100  # Scale reward
            
            return next_obs, reward, terminated, truncated, info
        
        def close(self):
            pass
    
    # 6. Patch the PortfolioEnv where it's imported in eval_ppo
    monkeypatch.setattr("src.rl_agent.eval_ppo.PortfolioEnv", SmartDummyEvalEnv)
    
    # 7. Test evaluation
    evaluator = eval_ppo.PortfolioEvaluator(
        model_path=model_path, 
        data_path=data_path, 
        results_dir=str(tmp_path / "eval_results")
    )
    
    # Run evaluation with small number for testing
    metrics = evaluator.evaluate_episodes(n_episodes=2, n_envs=1, deterministic=True)
    
    # Verify evaluation results
    assert "mean_reward" in metrics, f"Missing mean_reward in metrics: {metrics.keys()}"
    assert metrics["n_episodes"] == 2, f"Expected 2 episodes, got {metrics['n_episodes']}"
    assert "portfolio_values" in metrics, "Missing portfolio_values in metrics"
    assert len(metrics["episode_rewards"]) == 2, f"Expected 2 episode rewards, got {len(metrics.get('episode_rewards', []))}"
    
    print(f"Evaluation metrics: {metrics}")
    
    # 8. Switch to backtest environment and test backtest
    monkeypatch.setattr("src.rl_agent.eval_ppo.PortfolioEnv", SmartDummyBacktestEnv)
    
    # Create new evaluator instance for backtest (to get new env)
    evaluator_bt = eval_ppo.PortfolioEvaluator(
        model_path=model_path, 
        data_path=data_path, 
        results_dir=str(tmp_path / "backtest_results")
    )
    
    backtest_results = evaluator_bt.backtest(deterministic=True)
    
    # Verify backtest results
    assert "total_return" in backtest_results, f"Missing total_return in backtest: {backtest_results.keys()}"
    assert "final_value" in backtest_results, "Missing final_value in backtest"
    assert "initial_value" in backtest_results, "Missing initial_value in backtest" 
    assert backtest_results["final_value"] > backtest_results["initial_value"], \
        f"Expected profit: final={backtest_results['final_value']} <= initial={backtest_results['initial_value']}"
    
    print(f"Backtest results: {backtest_results}")
    
    # Test key metrics exist
    required_backtest_keys = [
        'total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
        'max_drawdown', 'win_rate', 'portfolio_values'
    ]
    for key in required_backtest_keys:
        assert key in backtest_results, f"Missing required backtest key: {key}"


def test_model_loading_fails_without_file(dummy_paths: tuple[str, str]):
    """Test that evaluator raises FileNotFoundError when model doesn't exist."""
    model_path, data_path = dummy_paths
    os.remove(model_path)
    
    with pytest.raises(FileNotFoundError, match="Model not found"):
        eval_ppo.PortfolioEvaluator(model_path, data_path)