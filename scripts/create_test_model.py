# create_test_model_fixed.py
import os
import sys
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import your actual environment
try:
    from environment.portfolio_env import PortfolioEnv
    print("✅ Successfully imported PortfolioEnv")
except ImportError as e:
    print(f"❌ Failed to import PortfolioEnv: {e}")
    print("Make sure the environment directory structure is correct")
    sys.exit(1)

def create_minimal_sample_data(output_path="data/sample/data_ppo_sample.parquet"):
    """
    Create minimal sample data that matches the expected format for PortfolioEnv.
    """
    print("Creating minimal sample data...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create 200 trading days of data (enough for lookback + training)
    n_periods = 200
    n_assets = 10  # Small number of assets for testing
    
    # Asset symbols
    asset_symbols = [f"ASSET_{i:02d}" for i in range(n_assets)]
    
    # Generate realistic price data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    # Starting prices
    initial_prices = np.random.uniform(50, 200, n_assets)
    
    # Generate price series with random walk + small trend
    prices = np.zeros((n_periods, n_assets))
    prices[0] = initial_prices
    
    for i in range(1, n_periods):
        # Daily returns: small positive drift + noise
        daily_returns = np.random.normal(0.0005, 0.02, n_assets)  # ~0.05% daily return, 2% volatility
        prices[i] = prices[i-1] * (1 + daily_returns)
    
    # Create DataFrame with required column format
    data = {}
    
    # Add Adjusted Close prices
    for i, asset in enumerate(asset_symbols):
        data[f'Adj Close_{asset}'] = prices[:, i]
    
    # Add Volume ratios (normalized volume)
    for asset in asset_symbols:
        data[f'Volume_Ratio_{asset}'] = np.random.uniform(0.5, 2.0, n_periods)  # Volume ratio
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    
    # Save to parquet
    df.to_parquet(output_path)
    print(f"✅ Sample data created: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Assets: {n_assets}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Columns: {list(df.columns[:5])}...")
    
    return output_path, df

def load_and_inspect_existing_data(data_path="data/sample/data_ppo_sample.parquet"):
    """
    Load and inspect your existing sample data to understand its structure.
    """
    print(f"Loading existing sample data: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Sample data not found at: {data_path}")
    
    # Load the existing data
    df = pd.read_parquet(data_path)
    
    print(f"✅ Sample data loaded successfully")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Columns ({len(df.columns)}): {list(df.columns[:10])}...")
    
    # Analyze column structure
    adj_close_cols = [col for col in df.columns if col.startswith('Adj Close_')]
    volume_cols = [col for col in df.columns if col.startswith('Volume_Ratio_')]
    
    print(f"   Adj Close columns: {len(adj_close_cols)}")
    print(f"   Volume Ratio columns: {len(volume_cols)}")
    
    if adj_close_cols:
        asset_symbols = [col.replace('Adj Close_', '') for col in adj_close_cols]
        print(f"   Asset symbols: {asset_symbols[:5]}{'...' if len(asset_symbols) > 5 else ''}")
        print(f"   Number of assets: {len(asset_symbols)}")
    
    # Check for any missing data
    missing_data = df.isnull().sum().sum()
    print(f"   Missing data points: {missing_data}")
    
    return data_path, df

def test_environment_with_data(data_path):
    """
    Test that the environment works with the created data.
    """
    print(f"\nTesting environment with data: {data_path}")
    
    try:
        # Create environment with same config as your actual training
        env = PortfolioEnv(
            data_path=data_path,
            lookback_window=60,  # Same as your config
            initial_capital=100000.0,
            transaction_cost=0.001,
            cash_weight=True,
            normalize_observations=True,
            reward_scaling=1000.0
        )
        
        print(f"✅ Environment created successfully")
        print(f"   Assets: {env.n_assets}")
        print(f"   Asset symbols: {env.asset_symbols[:5]}...")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
        
        # Test step
        action = np.ones(env.action_space.shape[0]) / env.action_space.shape[0]  # Equal weights
        obs2, reward, terminated, truncated, info2 = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward:.4f}")
        print(f"   Portfolio value: ${info2['portfolio_value']:,.2f}")
        
        env.close()
        return env.observation_space.shape, env.action_space.shape
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        raise

def create_test_model(model_path="models/sample_model.zip", data_path="data/sample/data_ppo_sample.parquet"):
    """
    Create a test model using the actual PortfolioEnv.
    """
    print(f"\nCreating test model: {model_path}")
    
    if os.path.exists(model_path):
        print(f"✅ Test model already exists at {model_path}, skipping creation.")
        return True
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Set PyTorch threads
    torch.set_num_threads(1)
    print(f"PyTorch using {torch.get_num_threads()} threads.")
    
    try:
        # Create environment for training
        def make_env():
            env = PortfolioEnv(
                data_path=data_path,
                lookback_window=60,
                initial_capital=100000.0,
                transaction_cost=0.001,
                cash_weight=True,
                normalize_observations=True,
                reward_scaling=1000.0
            )
            # Add monitor for logging
            return Monitor(env)
        
        # Create vectorized environment
        vec_env = DummyVecEnv([make_env])
        
        print("✅ Vectorized environment created")
        
        # Create PPO model with simple configuration
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            learning_rate=3e-4,
            n_steps=128,  # Small for quick training
            batch_size=32,
            n_epochs=3,
            verbose=1,
            device="cpu"
        )
        
        print("✅ PPO model created")
        
        # Short training run
        print("Starting brief training...")
        model.learn(total_timesteps=1000)  # Very short for testing
        print("✅ Training completed")
        
        # Save model
        model.save(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Test loading and prediction
        print("Testing model loading and prediction...")
        loaded_model = PPO.load(model_path, device="cpu")
        
        # Get test observation
        obs = vec_env.reset()
        action, _ = loaded_model.predict(obs, deterministic=True)
        print(f"✅ Model prediction successful")
        print(f"   Action shape: {action.shape}")
        print(f"   Action sum: {action.sum():.4f}")
        
        vec_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_env_compatibility(model_path, data_path):
    """
    Verify that the model and environment are compatible.
    """
    print(f"\nVerifying compatibility between model and environment...")
    
    try:
        # Load model
        model = PPO.load(model_path, device="cpu")
        print("✅ Model loaded")
        
        # Create environment
        env = PortfolioEnv(
            data_path=data_path,
            lookback_window=60,
            initial_capital=100000.0,
            transaction_cost=0.001,
            cash_weight=True,
            normalize_observations=True,
            reward_scaling=1000.0
        )
        print("✅ Environment created")
        
        # Test compatibility
        obs, info = env.reset(seed=42)
        print(f"   Environment obs shape: {obs.shape}")
        
        # Model expects vectorized input
        obs_vec = obs.reshape(1, -1)  # Add batch dimension
        action, _ = model.predict(obs_vec, deterministic=True)
        print(f"   Model action shape: {action.shape}")
        
        # Test step
        obs2, reward, terminated, truncated, info2 = env.step(action[0])  # Remove batch dimension
        print(f"✅ Full environment step successful")
        print(f"   Reward: {reward:.4f}")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to create test model using existing sample data.
    """
    print("🚀 Creating test model with your existing sample data...")
    
    # Paths
    data_path = "data/sample/data_ppo_sample.parquet"
    model_path = "models/sample_model.zip"
    
    try:
        # Step 1: Load and inspect existing sample data
        print("\n" + "="*60)
        print("STEP 1: Loading existing sample data")
        print("="*60)
        data_path, df = load_and_inspect_existing_data(data_path)
        
        # Step 2: Test environment with your data
        print("\n" + "="*60)
        print("STEP 2: Testing environment with your data")
        print("="*60)
        obs_shape, action_shape = test_environment_with_data(data_path)
        
        # Step 3: Create compatible model
        print("\n" + "="*60)
        print("STEP 3: Creating test model")
        print("="*60)
        success = create_test_model(model_path, data_path)
        
        if not success:
            print("❌ Failed to create test model")
            return False
        
        # Step 4: Verify compatibility
        print("\n" + "="*60)
        print("STEP 4: Verifying compatibility")
        print("="*60)
        compatible = verify_model_env_compatibility(model_path, data_path)
        
        if compatible:
            print("\n" + "🎉"*20)
            print("SUCCESS! Test model created with your existing data!")
            print("🎉"*20)
            print(f"\nUsing your existing data:")
            print(f"  Data:  {data_path}")
            print(f"  Shape: {df.shape}")
            print(f"  New compatible model: {model_path}")
            print(f"\nTo test eval_ppo.py:")
            print(f"  python src/rl_agent/eval_ppo.py --model-path {model_path} --data-path {data_path}")
            return True
        else:
            print("❌ Model and environment are not compatible")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ Your sample data file was not found: {e}")
        print("Make sure the file exists at: data/sample/data_ppo_sample.parquet")
        return False
    except Exception as e:
        print(f"❌ Script failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)