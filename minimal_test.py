# minimal_test.py - Test imports one by one to find the segfault source

import sys
import os

print("🔍 MINIMAL SEGFAULT TEST")
print("========================")

# Test 1: Basic Python
print("✅ Basic Python works")

# Test 2: Standard library
try:
    import json
    import time
    import pathlib
    print("✅ Standard library imports work")
except Exception as e:
    print(f"❌ Standard library failed: {e}")
    sys.exit(1)

# Test 3: NumPy (common segfault source)
try:
    print("Testing NumPy import...")
    import numpy as np
    print(f"✅ NumPy {np.__version__} imported successfully")
    
    # Test basic operation
    x = np.array([1, 2, 3])
    y = x * 2
    print(f"✅ NumPy operations work: {y}")
    
except Exception as e:
    print(f"❌ NumPy failed: {e}")
    print("💡 Try: pip uninstall numpy && pip install numpy==1.24.3")
    sys.exit(1)

# Test 4: PyTorch (major segfault source on macOS)
try:
    print("Testing PyTorch import...")
    import torch
    print(f"✅ PyTorch {torch.__version__} imported successfully")
    
    # Test basic operations
    x = torch.randn(3, 3)
    y = x * 2
    print(f"✅ PyTorch operations work")
    print(f"   Device available: {torch.device('cpu')}")
    
except Exception as e:
    print(f"❌ PyTorch failed: {e}")
    print("💡 Try: pip uninstall torch && pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

# Test 5: Gymnasium
try:
    print("Testing Gymnasium import...")
    import gymnasium as gym
    print("✅ Gymnasium imported successfully")
    
    # Test environment creation
    env = gym.make('CartPole-v1')
    env.close()
    print("✅ Gymnasium environment creation works")
    
except Exception as e:
    print(f"❌ Gymnasium failed: {e}")
    print("💡 Try: pip install gymnasium[classic_control]")

# Test 6: Pandas
try:
    print("Testing Pandas import...")
    import pandas as pd
    print(f"✅ Pandas {pd.__version__} imported successfully")
except Exception as e:
    print(f"❌ Pandas failed: {e}")

# Test 7: Stable Baselines3 (final boss)
try:
    print("Testing Stable Baselines3 import...")
    from stable_baselines3 import PPO
    print("✅ Stable Baselines3 imported successfully")
    
    # Test basic functionality
    print("Testing SB3 basic functionality...")
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, verbose=0)
    print("✅ SB3 model creation works")
    env.close()
    
except Exception as e:
    print(f"❌ Stable Baselines3 failed: {e}")
    print("💡 Try: pip uninstall stable-baselines3 && pip install stable-baselines3")

print("\n🎉 All imports successful! The segfault is likely in your specific model loading.")
print("\n💡 Next steps:")
print("1. Try setting environment variables:")
print("   export OMP_NUM_THREADS=1")
print("   export MKL_NUM_THREADS=1") 
print("2. Run your evaluation with these set")
print("3. If still failing, recreate your model with CPU-only PyTorch")