# LLM-Sentiment Enhanced RL for Portfolio Management (CAC 40)

## Project Overview

This repository implements a **reinforcement learning (RL)** portfolio management framework for the French equity market (CAC 40), enhanced with optional sentiment signals extracted from French financial news via large language models.

The core algorithm is **Proximal Policy Optimization (PPO)** with pluggable policy architectures (MLP, CNN, LSTM). Sentiment integration is toggled via configuration for clean ablation studies.

**Key features:**

* End-to-end pipeline: data ingestion, preprocessing, RL training, out-of-sample evaluation
* Temporal train/test split (default: train 2018-2022, test 2023-2024) to prevent look-ahead bias
* Classical baselines (equal-weight, buy-and-hold, inverse-volatility) for rigorous comparison
* CNN and LSTM policy architectures that exploit the temporal structure of observations
* French-language sentiment via CamemBERT (with FinBERT fallback for comparison)
* Multi-seed training for statistical significance
* CI via GitHub Actions

---

## Repository Structure

```text
LLM_in_portfoliomanagement/
│
├── data/
│   ├── raw/                     # Per-ticker CSVs from Yahoo Finance
│   ├── preprocessed/            # Cleaned parquet files (prices, features)
│   ├── sample/                  # Minimal 2-asset sample for quick tests
│   └── download_prices.py       # Download CAC 40 price data
│
├── environment/
│   └── portfolio_env.py         # Custom Gymnasium environment
│
├── src/
│   ├── rl_agent/
│   │   ├── train_ppo.py         # PPO training (split, policy select, multi-seed)
│   │   ├── eval_ppo.py          # Full evaluation with metrics & visualization
│   │   ├── baselines.py         # Classical portfolio baselines
│   │   ├── custom_policies.py   # CNN / LSTM feature extractors for SB3
│   │   └── visualizations.py    # Plotting utilities
│   └── sentiment/
│       └── extract_sentiment.py # CamemBERT / FinBERT sentiment pipeline
│
├── notebooks/
│   ├── 01_prepare_data.ipynb    # Raw data → preprocessed parquet
│   ├── convert_data_returns.ipynb
│   └── create_sample.ipynb      # Create minimal sample dataset
│
├── tests/                       # pytest suite
│   ├── test_portfolio_env.py
│   ├── test_train_ppo_integration.py
│   ├── test_eval_ppo.py
│   └── test_model_loading.py
│
├── models/                      # Saved model checkpoints (timestamped)
├── logs/                        # TensorBoard logs & monitor CSVs
├── results/                     # Evaluation outputs
│
├── environment.yml              # Conda environment specification
├── config_minimal.yaml          # Minimal config for quick testing
├── .github/workflows/conda-tests.yml
├── LICENSE                      # MIT
└── README.md
```

---

## Dependencies & Environment Setup

```bash
# 1. Clone
git clone https://github.com/LeonHauch/LLM_in_portfoliomanagement.git
cd LLM_in_portfoliomanagement

# 2. Create Conda environment
conda env create --file environment.yml
conda activate fr_sent_ml

# 3. Verify
python -c "import pandas, gymnasium, stable_baselines3, torch; print('OK')"
```

---

## Data Acquisition

### Price Data

* **Script:** `data/download_prices.py`
* **Assets:** 27 CAC 40 constituents (e.g. `AIR.PA`, `BNP.PA`, `MC.PA`, ...)
* **Period:** 2018-01-01 to 2024-12-31
* **Source:** Yahoo Finance via `yfinance`

```bash
python data/download_prices.py
```

### Data Preprocessing

* **Notebook:** `notebooks/01_prepare_data.ipynb` and `notebooks/convert_data_returns.ipynb`
* Features per asset: Adj Close, Volume Ratio, Log Return, RSI, MACD, Bollinger, Rolling Volatility
* Output: `data/preprocessed/data_ppo.parquet` (1,744 rows x 216 columns)

---

## The Environment

`environment/portfolio_env.py` — a Gymnasium environment for daily portfolio allocation.

**Observations:** Flat vector of `(lookback_window x n_assets x n_features) + portfolio_weights`. Default: 60-day lookback, 5 features per asset (log returns, volume ratio, rolling volatility, correlation, SMA ratio), plus optional sentiment.

**Actions:** Continuous weights passed through softmax to sum to 1. Supports a cash allocation.

**Reward:** Net log-return after transaction costs (0.1%), with an optional configurable risk-adjustment bonus (off by default).

**Train/test split:** Set via `start_date`/`end_date` or `start_idx`/`end_idx` parameters. Default split: train on 2018-2022, test on 2023-2024.

---

## Training the PPO Agent

```bash
# Basic training (uses default config)
python src/rl_agent/train_ppo.py \
    --data-path data/preprocessed/data_ppo.parquet \
    --config config_minimal.yaml

# Multi-seed training
python src/rl_agent/train_ppo.py \
    --data-path data/preprocessed/data_ppo.parquet \
    --config config_minimal.yaml \
    --seeds 42,123,456

# Training with sentiment
python src/rl_agent/train_ppo.py \
    --data-path data/preprocessed/data_ppo.parquet \
    --sentiment-path data/preprocessed/sentiment.csv \
    --config config_minimal.yaml
```

### Policy Selection

Set in the config YAML:

```yaml
policy:
  type: cnn      # mlp | cnn | lstm
  features_dim: 128
```

* **mlp:** Default fully-connected (baseline)
* **cnn:** 1D convolution over the time axis — captures temporal patterns
* **lstm:** Recurrent processing of the lookback window

All three use the same PPO optimization; only the feature extractor changes.

---

## Evaluation

```bash
python src/rl_agent/eval_ppo.py \
    --model-path models/<run>/best_model.zip \
    --data-path data/preprocessed/data_ppo.parquet \
    --n-episodes 10
```

**Metrics:** Annualized return, volatility, Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio.

### Classical Baselines

```python
from src.rl_agent.baselines import run_all_baselines

results = run_all_baselines(
    data_path="data/preprocessed/data_ppo.parquet",
    start_idx=1200,  # test period start
    end_idx=1743,     # test period end
)
for name, metrics in results.items():
    print(f"{name}: Sharpe={metrics['sharpe_ratio']:.3f}")
```

Baselines included: equal-weight (daily & monthly rebalance), buy-and-hold, inverse-volatility.

---

## Sentiment Integration

### 1. Score headlines

```bash
# From real news data
python src/sentiment/extract_sentiment.py score \
    --input data/raw/news/headlines.csv \
    --output data/preprocessed/sentiment.csv \
    --model camembert

# Or generate dummy sentiment for testing
python src/sentiment/extract_sentiment.py dummy \
    --data-path data/preprocessed/data_ppo.parquet \
    --output data/preprocessed/sentiment_dummy.csv
```

**Models available:**
* `camembert` — CamemBERT fine-tuned on French text (tblard/tf-allocine)
* `finbert` — ProsusAI/finbert for English financial text

### 2. Train with sentiment

```yaml
# In config:
env:
  use_sentiment: true
sentiment_path: data/preprocessed/sentiment.csv
```

The trainer merges sentiment into `Sentiment_{ticker}` columns; the environment reads them automatically.

---

## Configuration

Example config (see `config_minimal.yaml` for a quick-test version):

```yaml
seed: 42
data_path: data/preprocessed/data_ppo.parquet

split:
  train_end_date: "2022-12-31"
  test_start_date: "2023-01-01"

policy:
  type: cnn
  features_dim: 128

env:
  lookback_window: 60
  transaction_cost: 0.001
  cash_weight: true
  use_sentiment: false
  risk_bonus_weight: 0.0

training:
  total_timesteps: 1000000
  n_envs: 4
  learning_rate: 0.0003
```

---

## Reproducibility & CI

* **GitHub Actions** in `.github/workflows/conda-tests.yml` — runs pytest on push/PR
* **Multi-seed training** via `--seeds` flag for statistical robustness
* **Conda environment** pinned in `environment.yml`

---

## Project Roadmap

* [x] Baseline PPO on CAC 40 returns
* [x] Train/test temporal split & classical baselines
* [x] CNN/LSTM policy architectures
* [x] Sentiment extraction pipeline (CamemBERT)
* [ ] Acquire French financial news corpus (Les Echos, Reuters France)
* [ ] Full sentiment-enhanced training & ablation study
* [ ] Cross-market comparison (US vs France)
* [ ] Analysis of translation vs direct French sentiment

---

## License

MIT — see `LICENSE`.
