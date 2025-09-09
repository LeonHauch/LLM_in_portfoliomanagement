import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from src.rl_agent import eval_ppo

@pytest.fixture
def dummy_paths(tmp_path: eval_ppo.Path):
    """Fixture to provide a temporary model path and the real data path."""
    model_path = tmp_path / "models.sample_model.zip"
    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'sample', 'data_ppo_sample.parquet'
    )
    model_path.write_text("")
    return str(model_path), str(data_path)

def test_evaluate_and_backtest(monkeypatch: pytest.MonkeyPatch, tmp_path: eval_ppo.Path, dummy_paths: tuple[str, str]):
    model_path, data_path = dummy_paths
    
    # 1. ECHTE Daten laden um Struktur zu verstehen
    print(f"Loading real data from: {data_path}")
    real_data = pd.read_parquet(data_path)
    print(f"Data shape: {real_data.shape}")
    print(f"Data columns: {real_data.columns.tolist()}")
    print(f"Data column types: {type(real_data.columns)}")
    
    # Für MultiIndex Spalten
    if isinstance(real_data.columns, pd.MultiIndex):
        print(f"MultiIndex levels: {real_data.columns.levels}")
        print(f"MultiIndex names: {real_data.columns.names}")
    
    # 2. ECHTE Umgebung kurz erstellen um Parameter zu extrahieren
    try:
        # Verwende echte Daten für Umgebungs-Initialisierung
        real_env = eval_ppo.PortfolioEnv(data_path=data_path)
        
        # Observation space herausfinden
        obs_result = real_env.reset()
        if isinstance(obs_result, tuple):
            obs = obs_result[0]  # Gymnasium format
        else:
            obs = obs_result
        
        obs_shape = obs.shape
        action_space_size = real_env.action_space.shape[0]
        
        # Zusätzliche Umgebungs-Parameter extrahieren
        asset_symbols = getattr(real_env, 'asset_symbols', [])
        cash_weight = getattr(real_env, 'cash_weight', False)
        
        real_env.close()
        
        print(f"Environment parameters:")
        print(f"  - Observation shape: {obs_shape}")
        print(f"  - Action space size: {action_space_size}")
        print(f"  - Asset symbols: {asset_symbols}")
        print(f"  - Cash weight: {cash_weight}")
        
    except Exception as e:
        print(f"Could not introspect real environment: {e}")
        raise  # Fail fast - wir brauchen diese Info
    
    # 3. Dummy Model mit KORREKTER Action-Größe basierend auf echten Daten
    class DummyModel:
        device = "cpu"
        def predict(self, obs, deterministic=True):
            # Gleichmäßige Verteilung über alle Assets (inkl. Cash falls vorhanden)
            equal_weight = 1.0 / action_space_size
            action = np.full(action_space_size, equal_weight)
            print(f"DummyModel returning action shape: {action.shape}, values: {action}")
            return action, None
    
    monkeypatch.setattr("src.rl_agent.eval_ppo.PPO", 
                       type("PPO", (), {"load": staticmethod(lambda *a, **k: DummyModel())}))
    
    # 4. Intelligente Dummy Environment - nutzt ECHTE Datenstruktur
    class IntelligentDummyEnv:
        def __init__(self, data_path=None):
            self.steps = 0
            self.data_path = data_path
            
            # ECHTE Daten laden und Struktur übernehmen
            if data_path and os.path.exists(data_path):
                self.real_data = pd.read_parquet(data_path)
                print(f"DummyEnv loaded real data with shape: {self.real_data.shape}")
                
                # Asset-Informationen aus echten Daten extrahieren
                if isinstance(self.real_data.columns, pd.MultiIndex):
                    # Für MultiIndex - Asset-Namen aus Level extrahieren
                    self.asset_symbols = [col for col in self.real_data.columns.get_level_values(0).unique() 
                                        if col not in ['Date', 'timestamp']]
                else:
                    # Für normale Spalten
                    self.asset_symbols = [col for col in self.real_data.columns 
                                        if col not in ['Date', 'timestamp']]
                
                self.n_assets = len(self.asset_symbols)
                print(f"DummyEnv detected {self.n_assets} assets: {self.asset_symbols}")
            else:
                raise FileNotFoundError(f"Data file not found: {data_path}")
        
        def reset(self, seed=None):
            self.steps = 0
            # Korrekte Observation Shape basierend auf echter Umgebung
            return np.random.random(obs_shape)
        
        def step(self, action):
            self.steps += 1
            done = self.steps >= 2
            
            print(f"DummyEnv.step received action shape: {action.shape}, type: {type(action)}")
            
            # Realistische Portfolio-Werte
            portfolio_value = 100000 * (1 + 0.001 * self.steps)
            
            # Action in Weights konvertieren (wie echte Umgebung)
            if isinstance(action, np.ndarray):
                weights = action.tolist()
            else:
                weights = [float(action)] if np.isscalar(action) else list(action)
            
            # Ensure weights match expected asset count
            if len(weights) != self.n_assets + (1 if cash_weight else 0):
                print(f"WARNING: Weight length mismatch. Expected: {self.n_assets + (1 if cash_weight else 0)}, got: {len(weights)}")
            
            info = {
                "portfolio_value": portfolio_value,
                "weights": weights
            }
            return np.random.random(obs_shape), 1.0, done, info
        
        def close(self):
            pass
    
    # 5. Backtest Dummy Environment
    class BacktestDummyEnv(IntelligentDummyEnv):
        def reset(self, seed=None):
            self.steps = 0
            return np.random.random(obs_shape), {}
        
        def step(self, action):
            self.steps += 1
            done = self.steps >= 3
            
            portfolio_value = 100000 * (1 + 0.01 * self.steps)
            
            if isinstance(action, np.ndarray):
                weights = action.tolist()
            else:
                weights = [float(action)] if np.isscalar(action) else list(action)
            
            info = {
                "portfolio_value": portfolio_value,
                "weights": weights,
                "net_return": 0.01
            }
            return np.random.random(obs_shape), 1.0, False, False, info
    
    # 6. Patching mit intelligenten Dummies
    monkeypatch.setattr("src.rl_agent.eval_ppo.PortfolioEnv", IntelligentDummyEnv)
    
    # 7. Test evaluate_episodes
    evaluator = eval_ppo.PortfolioEvaluator(model_path, data_path, results_dir=tmp_path)
    metrics = evaluator.evaluate_episodes(n_episodes=2, n_envs=1)
    
    assert "mean_reward" in metrics
    assert metrics["n_episodes"] == 2
    print(f"Evaluation metrics: {metrics}")
    
    # 8. Test backtest
    monkeypatch.setattr("src.rl_agent.eval_ppo.PortfolioEnv", BacktestDummyEnv)
    
    backtest_results = evaluator.backtest()
    assert "total_return" in backtest_results
    assert backtest_results["final_value"] > backtest_results["initial_value"]
    print(f"Backtest results: {backtest_results}")

def test_model_loading_fails_without_file(dummy_paths: tuple[str, str]):
    model_path, data_path = dummy_paths
    os.remove(model_path)
    with pytest.raises(FileNotFoundError):
        eval_ppo.PortfolioEvaluator(model_path, data_path)