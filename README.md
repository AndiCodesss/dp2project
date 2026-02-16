# Crypto Index: ML-Driven Portfolio Construction and Backtesting

A Spark-based pipeline that downloads cryptocurrency market data, applies four machine learning models from Spark MLlib, constructs diversified portfolios, and evaluates them through a rigorous 12-month backtest.

## Project Structure

```
0842_Team1/
├── 01_data_preparation.ipynb   # Data download and Parquet conversion
├── 02_ml_models.ipynb          # ML models and portfolio construction
├── 03_backtesting.ipynb        # Monthly rebalancing backtest
├── report.pdf                  # Project report
├── HTML/                       # Notebook HTML exports
├── crypto_data_parquet/        # Generated: partitioned Parquet dataset
├── crypto_data_4h/             # Generated: raw CSV files (can be deleted)
└── output/                     # Generated: results, charts, CSV exports
```

## Notebooks

### 01 — Data Preparation

Downloads 4-hour OHLCV candle data for ~600 USDT trading pairs from the Binance public data archive (Oct 2024 – Dec 2025) and converts it into year/month-partitioned Parquet files.

**Pipeline steps:**
1. Discover historical trading pairs via the Binance S3 archive
2. Download monthly ZIP archives in parallel (10 workers)
3. Parse raw CSVs — convert timestamps, add headers
4. Write Snappy-compressed Parquet partitioned by `year/month` using Spark

**Key outputs:**
- `crypto_data_parquet/` — ~1.1M rows across ~500 symbols, ~40–60 MB on disk
- Oct–Dec 2024 serves as warmup data (90-day lookback for indicators); Jan–Dec 2025 is the evaluation period

**Runtime:** ~10 minutes (network-dependent)

---

### 02 — ML Models

Applies four Spark MLlib models to the prepared data and constructs an investable portfolio.

| Model | Purpose | Key Output |
|-------|---------|------------|
| **Pearson Correlation** | Identify redundant asset pairs | 150 x 150 correlation matrix, tournament-filtered assets |
| **PCA** | Decompose returns into market factors | Scree plot, explained variance (PC1 captures ~55%) |
| **K-Means Clustering** | Group assets by risk/return profile | 5 clusters, silhouette score ~0.47 |
| **Gradient Boosted Trees** | Predict next-period returns | Feature importances, test RMSE |

**Portfolio construction:** Correlation tournament (drop one asset from pairs with |rho| > 0.85) followed by top-20 selection by Sharpe ratio, weighted by inverse volatility.

**Key outputs (in `output/`):**

| File | Content |
|------|---------|
| `smart_index.csv` | Final portfolio: 20 symbols with weights |
| `kmeans_clusters.csv` | Cluster assignments for all 500 assets |
| `pca_components.csv` | Explained variance per principal component |
| `gbt_feature_importance.csv` | Feature importances from the GBT model |
| `correlation_matrix.png` | Correlation heatmap |
| `kmeans_clusters_viz.png` | Cluster scatter plot (volatility vs. Sharpe) |
| `pca_scree_plot.png` | Variance explained per component |
| `gbt_feature_importance.png` | Feature importance bar chart |

---

### 03 — Backtesting

Simulates five portfolio strategies with monthly rebalancing over Jan–Dec 2025 using a rolling 90-day lookback window.

| Strategy | Method | Assets |
|----------|--------|--------|
| **A** | BTC buy-and-hold benchmark | 1 |
| **B** | Correlation tournament + inverse-volatility weights | 20 |
| **C** | K-Means cluster-balanced round-robin selection | 20 |
| **D** | PCA residual variance (most market-independent assets) | 20 |
| **E** | GBT predicted-return ranking | 20 |

**Design principles:**
- No lookahead bias — assets selected using only past data
- Strict tradability — an asset must have an exact 4h candle at rebalance time
- No transaction costs assumed (documented limitation)

**Key outputs (in `output/`):**

| File | Content |
|------|---------|
| `backtest_results.csv` | Monthly returns per strategy |
| `backtest_performance.csv` | Annualized return, volatility, Sharpe, max drawdown, win rate |
| `backtest_equity.csv` | Month-by-month portfolio values |
| `backtest_comparison.png` | Equity curves, returns heatmap, drawdown chart, summary table |

## Requirements

- **Python 3.8+**
- **Apache Spark 3.x** with PySpark
- **Java 8 or 11** (required by Spark)

Python packages:

```
pyspark
findspark
numpy
matplotlib
seaborn
requests
```

## Quick Start

Run the notebooks in order:

```
1. 01_data_preparation.ipynb   — downloads data, writes Parquet (~10 min)
2. 02_ml_models.ipynb          — trains models, builds portfolio
3. 03_backtesting.ipynb        — runs 12-month backtest, generates charts
```

Each notebook initializes its own Spark session. No shared state is required between sessions — only the `crypto_data_parquet/` directory produced by notebook 01 is needed by notebooks 02 and 03.

## Configuration

Key parameters (editable in each notebook's configuration cell):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAIR_MODE` | `historical-usdt` | Universe selection mode |
| `LOOKBACK_DAYS` | 90 | Rolling window for model fitting |
| `TOP_N_ASSETS` | 20 | Portfolio size |
| `CORRELATION_THRESHOLD` | 0.85 | Redundancy filter cutoff |
| `N_CLUSTERS` | 5 | K-Means cluster count |
| `MAX_WIDE_MATRIX_SYMBOLS` | 150 | Cap for Spark codegen stability |
| `INITIAL_CAPITAL` | $10,000 | Backtest starting value |

## Data

- **Source:** [Binance Public Data](https://data.binance.vision/) (4-hour spot klines)
- **Period:** October 2024 – December 2025 (15 months)
- **Universe:** ~500 USDT pairs (historical, excluding leveraged tokens)
- **Volume:** ~1.1 million rows, ~40–60 MB compressed Parquet

## License

- **Code:** [MIT License](https://opensource.org/licenses/MIT)
- **Report and figures:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
