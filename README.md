# French RL Sentiment Portfolio

## Project Overview

This repository implements a **reinforcement learning (RL)** based portfolio management framework for the French equity market (CAC 40), enhanced with sentiment signals extracted from French financial news via large language models (LLMs).  The primary algorithm is **Proximal Policy Optimization (PPO)**.  Sentiment integration is optional and can be toggled for ablation.

**Key contributions:**

* End-to-end pipeline from data ingestion (prices + news) to RL training & evaluation
* Reproducible environment using Conda, GitHub, and best practices
* Clear separation of raw vs processed data, code modules, and notebooks
* Statistical evaluation of model performance (Sharpe ratio, drawdown, Welch’s t-test)

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Dependencies & Environment Setup](#dependencies--environment-setup)
3. [Data Acquisition](#data-acquisition)
4. [Data Preprocessing](#data-preprocessing)
5. [Training the PPO Agent](#training-the-ppo-agent)
6. [Evaluation](#evaluation)
7. [Extending with Sentiment](#extending-with-sentiment)
8. [Reproducibility & CI](#reproducibility--ci)
9. [Project Roadmap](#project-roadmap)
10. [Contributing](#contributing)
11. [License](#license)

---

## Repository Structure

```text
fr_sent_ml/
│
├── data/
│   ├── raw/                 # Raw CSVs & Parquet files (prices, news archives)
│   ├── processed/           # Cleaned, merged datasets (e.g., returns, sentiment)
│   └── preprocessed/        # Final data for modeling (parquet, CSV)
│
├── notebooks/              # Jupyter notebooks for EDA & preprocessing
│   └── 01_prepare_data.ipynb
│
├── src/                    # Source code modules
│   ├── data/               # Scripts to download & preprocess data
│   │   └── download_prices.py
│   ├── sentiment/          # Sentiment extraction wrappers (GPT-4 prompts)
│   ├── rl_agent/           # Environment + PPO training & evaluation
│   │   ├── portfolio_env.py
│   │   ├── train_ppo.py
│   │   └── eval_ppo.py
│   ├── utils/              # Utility functions (logging, seeding, config)
│   └── config.yaml         # Hyperparameters and paths
│
├── models/                 # Saved model checkpoints & HF Hub links
├── .github/                # CI workflows (lint, tests)
├── environment.yml         # Conda environment specification
├── requirements.txt        # pip fallback for specific packages
├── .gitignore
├── README.md               # ← You are here
└── LICENSE
```

---

## Dependencies & Environment Setup

All code should be run inside the **Conda** environment `fr_sent_ml` for reproducibility.

1. **Clone the repo**:

   ```bash
   git clone git@github.com:<YourUser>/french-rl-sentiment-portfolio.git
   cd french-rl-sentiment-portfolio
   ```

2. **Create and activate the Conda environment**:

   ```bash
   conda env create --file environment.yml
   conda activate fr_sent_ml
   ```

3. **Install additional pip packages**:

   ```bash
   pip install stable-baselines3 openai yfinance
   ```

4. **Set API keys**: create a `.env` file in the project root:

   ```ini
   OPENAI_API_KEY=sk-...
   HUGGINGFACE_TOKEN=hf-...
   ```

5. **Verify installation**:

   ```bash
   python -c "import pandas, gym, stable_baselines3, openai; print('✅ OK')"
   ```

---

## Data Acquisition

### 1. Price Data

* **Script**: `src/data/download_prices.py`
* **Assets**: CAC 40 constituents (e.g., `AIR.PA`, `BNP.PA`, …)
* **Period**: 2018-01-01 to 2024-12-31
* **Output**: MultiIndex CSVs in `data/raw/`

Run:

```bash
python src/data/download_prices.py
```

Raw data saved as `<ticker>.csv`.

### 2. News Data (Future Extension)

* Planned via RSS/API scraping of Les Échos, Reuters France, etc.
* Stored in `data/raw/news/` as JSON/CSV.

---

## Data Preprocessing

### Price Data → Returns

* **Notebook**: `notebooks/01_prepare_data.ipynb`
* **Steps**:

  1. Load `data/raw/*.csv` with `pd.read_csv(header=[0,1], index_col=0)`
  2. Concatenate into one DataFrame `price_data`
  3. Compute log-returns: `r_t = log(P_t/P_{t-1})`
  4. Drop missing
  5. Save to `data/preprocessed/prices.parquet`

```python
price_data.to_parquet('data/preprocessed/prices.parquet')
```

### Sentiment Data → Signals

*(Optional for baseline)*

* Extract daily sentiment scores using GPT-4 API
* Stored in `data/processed/sentiment.parquet`

---

## Training the PPO Agent

### 1. Environment

* **File**: `src/rl_agent/portfolio_env.py`
* **Observations**: price returns + portfolio weights
* **Actions**: continuous allocations (∑w\_i=1)
* **Reward**: portfolio return minus transaction cost (0.4%)

### 2. Training Script

* **File**: `src/rl_agent/train_ppo.py`
* **Command**:

  ```bash
  python src/rl_agent/train_ppo.py --config src/config.yaml
  ```
* **Output**: Saved model in `models/ppo_base.zip` (and `ppo_sentiment.zip` if sentiment used)

---

## Evaluation

* **File**: `src/rl_agent/eval_ppo.py`
* **Metrics**: Annualized return, volatility, Sharpe, Max drawdown
* **Statistical Test**: Welch’s t-test on Sharpe distributions across seeds

```bash
python src/rl_agent/eval_ppo.py --model models/ppo_base.zip --output results/base_metrics.json
```

---

## Extending with Sentiment

1. Run sentiment extraction:

   ```bash
   python src/sentiment/extract_sentiment.py --input data/raw/news --output data/processed/sentiment.parquet
   ```
2. Retrain PPO with sentiment flag:

   ```bash
   python src/rl_agent/train_ppo.py --config src/config_sent.yaml
   ```

Compare `ppo_base` vs `ppo_sentiment` metrics.

---

## Reproducibility & CI

* **GitHub Actions** in `.github/workflows/ci.yml` to lint, test, and validate environment.
* **Seed control** in `src/utils/seed.py` ensures deterministic runs per seed.
* Use **Docker** (optional) for environment encapsulation.

---

## Project Roadmap

* [x] Baseline PPO on CAC 40 returns
* [ ] Integrate French financial news sentiment
* [ ] Cross-market comparison (US vs France)
* [ ] Analysis of translation vs direct sentiment on French text

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Write tests and update docs
4. Open a Pull Request

---

## License

This project is licensed under the MIT License — see `LICENSE` for details.
