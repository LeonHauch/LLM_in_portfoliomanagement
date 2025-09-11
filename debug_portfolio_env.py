# debug_portfolio_env.py
import os
import numpy as np
import traceback
from stable_baselines3.common.vec_env import DummyVecEnv

# adjust path if needed
from environment.portfolio_env import PortfolioEnv

DATA_PATH = os.path.join("data", "sample", "data_ppo_sample.parquet")

def run_minimal_checks(data_path=DATA_PATH):
    print("Creating env instance...")
    env = PortfolioEnv(data_path=data_path)
    print("Observation space:", env.observation_space, " Action space:", env.action_space)
    print("Calling reset() ...")
    try:
        reset_res = env.reset()
        print("reset() returned type:", type(reset_res))
        if isinstance(reset_res, tuple):
            obs, info = reset_res
            print("  obs.shape:", getattr(obs, "shape", None), "dtype:", getattr(obs, "dtype", None))
            print("  info keys:", list(info.keys()))
        else:
            obs = reset_res
            print("  obs.shape:", getattr(obs, "shape", None), "dtype:", getattr(obs, "dtype", None))
    except Exception as e:
        print("Exception during reset():")
        traceback.print_exc()
        env.close()
        return

    # create a safe action (zeros) with correct shape
    action_shape = env.action_space.shape
    action = np.zeros(action_shape, dtype=np.float32)
    print("Stepping with action shape:", action.shape)
    try:
        step_res = env.step(action)
        print("step() returned types:", [type(x) for x in step_res])
        obs2 = step_res[0]
        print("  obs2.shape:", getattr(obs2, "shape", None), "dtype:", getattr(obs2, "dtype", None))
        print("  reward:", step_res[1], "terminated:", step_res[2], "truncated:", step_res[3])
    except Exception as e:
        print("Exception during step():")
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    run_minimal_checks()
