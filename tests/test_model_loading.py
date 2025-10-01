# test_model_loading.py
import os
import sys
import time
import torch
import zipfile
from pathlib import Path

def test_model_file(model_path):
    """Test if the model file is valid and readable."""
    print(f"Testing model file: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file doesn't exist: {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"✅ Model file exists, size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    # Test if it's a valid zip file (SB3 models are zipped)
    try:
        with zipfile.ZipFile(model_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            print(f"✅ Valid zip file with {len(file_list)} files:")
            for f in file_list:
                print(f"   - {f}")
        return True
    except zipfile.BadZipFile:
        print(f"❌ Model file is corrupted or not a valid zip file")
        return False

def test_torch_setup():
    """Test PyTorch setup."""
    print("\nTesting PyTorch setup...")
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ CUDA device: {torch.cuda.get_device_name()}")
    
    # Test device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✅ Selected device: {device}")
    
    # Test simple tensor creation
    try:
        x = torch.randn(10, 10).to(device)
        print(f"✅ Tensor creation successful on {device}")
        return device
    except Exception as e:
        print(f"❌ Tensor creation failed: {e}")
        return 'cpu'

def test_minimal_sb3_loading():
    """Test minimal SB3 loading without our model."""
    print("\nTesting minimal Stable Baselines3 loading...")
    
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        import gymnasium as gym
        
        print("✅ SB3 imports successful")
        
        # Create dummy environment
        env = make_vec_env('CartPole-v1', n_envs=1)
        print("✅ Dummy environment created")
        
        # Create and save a tiny model
        print("Creating tiny test model...")
        model = PPO('MlpPolicy', env, verbose=0)
        
        # Test saving
        test_model_path = "test_tiny_model.zip"
        model.save(test_model_path)
        print(f"✅ Tiny model saved to {test_model_path}")
        
        # Test loading
        print("Testing tiny model loading...")
        start_time = time.time()
        loaded_model = PPO.load(test_model_path, device='cpu')
        load_time = time.time() - start_time
        print(f"✅ Tiny model loaded in {load_time:.2f} seconds")
        
        # Cleanup
        env.close()
        if os.path.exists(test_model_path):
            os.remove(test_model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ SB3 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_our_model_loading(model_path, timeout_seconds=10):
    """Test loading our specific model with timeout."""
    print(f"\nTesting our model loading with {timeout_seconds}s timeout...")
    
    try:
        from stable_baselines3 import PPO
        
        # Force CPU to avoid CUDA issues
        device = 'cpu'
        print(f"Using device: {device}")
        
        # Set torch threads to 1 to avoid hanging
        torch.set_num_threads(1)
        print(f"Set PyTorch threads to: {torch.get_num_threads()}")
        
        print("Starting model load...")
        start_time = time.time()
        
        # Try loading with timeout using a simple approach
        import threading
        import queue
        
        def load_model_worker(q, model_path, device):
            try:
                model = PPO.load(model_path, device=device)
                q.put(('success', model))
            except Exception as e:
                q.put(('error', str(e)))
        
        q = queue.Queue()
        thread = threading.Thread(target=load_model_worker, args=(q, model_path, device))
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout
        try:
            result_type, result = q.get(timeout=timeout_seconds)
            load_time = time.time() - start_time
            
            if result_type == 'success':
                print(f"✅ Model loaded successfully in {load_time:.2f} seconds")
                print(f"   Model type: {type(result)}")
                print(f"   Model device: {result.device}")
                return True, result
            else:
                print(f"❌ Model loading failed: {result}")
                return False, None
                
        except queue.Empty:
            print(f"❌ Model loading timed out after {timeout_seconds} seconds")
            print("This suggests a hanging issue in Stable Baselines3 or PyTorch")
            return False, None
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_environment_loading(data_path):
    """Test environment loading separately."""
    print(f"\nTesting environment loading...")
    
    try:
        # Add path for imports
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        
        from environment.portfolio_env import PortfolioEnv
        
        print("✅ Environment import successful")
        
        start_time = time.time()
        env = PortfolioEnv(data_path=data_path)
        load_time = time.time() - start_time
        
        print(f"✅ Environment created in {load_time:.2f} seconds")
        print(f"   Assets: {env.n_assets}")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successful")
        
        env.close()
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔍 MODEL LOADING DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Get paths
    model_path = input("Enter model path (or press Enter for 'models/sample_model.zip'): ").strip()
    if not model_path:
        model_path = "models/sample_model.zip"
    
    data_path = input("Enter data path (or press Enter for 'data/sample/data_ppo_sample.parquet'): ").strip()
    if not data_path:
        data_path = "data/sample/data_ppo_sample.parquet"
    
    print(f"\nTesting with:")
    print(f"  Model: {model_path}")
    print(f"  Data:  {data_path}")
    print()
    
    # Run tests
    tests = [
        ("Model File Check", lambda: test_model_file(model_path)),
        ("PyTorch Setup", lambda: test_torch_setup()),
        ("Environment Loading", lambda: test_environment_loading(data_path)),
        ("SB3 Basic Test", lambda: test_minimal_sb3_loading()),
        ("Our Model Loading", lambda: test_our_model_loading(model_path, timeout_seconds=15))
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if isinstance(result, tuple):
                success = result[0]
            else:
                success = result
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*50}")
    
    for test_name, result in results.items():
        if isinstance(result, tuple):
            status = "✅ PASSED" if result[0] else "❌ FAILED"
        else:
            status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    # Recommendations
    print(f"\n🔧 RECOMMENDATIONS:")
    if not results.get("Model File Check", False):
        print("- Recreate the model using create_test_model_fixed.py")
    if not results.get("SB3 Basic Test", False):
        print("- Reinstall stable-baselines3: pip install stable-baselines3")
    if not results.get("Our Model Loading", (False, None))[0]:
        print("- Model loading is hanging - try forcing CPU device")
        print("- Try: torch.set_num_threads(1) before loading")
        print("- Check if model was saved properly")

if __name__ == "__main__":
    main()