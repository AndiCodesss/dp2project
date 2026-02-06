# Crypto Smart Index Engine

A scalable Big Data analytics engine built with **Apache Spark** to construct a **"Smart Beta" cryptocurrency index**.

Unlike standard market-cap weighted indices (which are dominated by highly correlated assets like BTC/ETH), this engine mathematically identifies a basket of assets that are **uncorrelated** and **risk-efficient**.

## 🧠 The Strategy

The core objective is to maximize the **Sharpe Ratio** (Risk-Adjusted Return) of the portfolio through a linear data pipeline:

1.  **Ingestion & Alignment**: Ingests ragged time-series data for 100+ pairs (millions of rows) into a unified Master Matrix.
2.  **Performance Scoring**: Calculates `Log Returns`, `Volatility` (Risk), and `Sharpe Ratio` for every asset using distributed Window Functions.
3.  **Correlation "Tournament"**:
    - Computes a massive Pearson Correlation Matrix (O(N²) complexity).
    - Identifies pairs with correlation > **0.85**.
    - **Prunes** the redundant asset with the lower Sharpe Ratio.
4.  **Portfolio Weighting**: Constructs the final portfolio using **Inverse Volatility Weighting** to prioritize stability.

## 🛠️ Technology Stack

- **Apache Spark (PySpark)**: Distributed processing for ETL and ML tasks.
- **Parquet**: Columnar storage for efficient I/O (millisecond precision timestamps).
- **Environment**: Linux / WSL (Windows Subsystem for Linux) recommended for Spark.

## 🚀 Usage

### 1. Setup Environment (WSL/Linux Recommended)

Spark runs best on POSIX systems. For Windows users, **WSL is highly recommended**.

```bash
# In your WSL terminal:
pip install pyspark numpy pandas matplotlib seaborn pyarrow findspark
```

### 2. Download & Prepare Data

Downloads Binance 4h-kline data, adds readable headers, and converts to optimized Parquet.

```bash
python download_crypto_data.py
```

_Note: This generates CSVs with readable `datetime` columns AND partitions parquet files._

### 3. Run Pipeline

Run the full pipeline including ingestion, metrics, and all 3 ML models:

```bash
python spark_crypto_index.py
```

## 📊 Output

All results are saved to the `output/` directory:

| Category    | File                         | Description                                            |
| ----------- | ---------------------------- | ------------------------------------------------------ |
| **Core**    | `smart_index.csv`            | Final index weights (Correlation + Inverse Volatility) |
| **K-Means** | `kmeans_clusters.csv`        | Cluster assignments for all assets                     |
|             | `kmeans_portfolio.csv`       | Top asset from each cluster                            |
|             | `kmeans_clusters_viz.png`    | Scatter plot of clusters (Risk vs Return)              |
| **PCA**     | `pca_components.csv`         | Principal components & variance explained              |
|             | `pca_scree_plot.png`         | Variance explained visualization                       |
| **GBT**     | `gbt_feature_importance.csv` | Feature importance from return prediction model        |
|             | `gbt_feature_importance.png` | Feature importance visualization                       |

## 🤖 Machine Learning Models

This project demonstrates **three distinct machine learning models** using Apache Spark MLlib:

### 1. K-Means Clustering

Groups cryptocurrencies into 5 clusters based on risk/return profiles (Sharpe ratio, volatility, returns).

- **Output**: Cluster assignments + top performer from each cluster
- **Evaluation**: Silhouette score for clustering quality
- **Visualization**: Risk vs Return scatter plot colored by cluster

### 2. PCA (Principal Component Analysis)

Identifies independent market factors by reducing dimensionality of the returns correlation matrix.

- **Output**: 10 principal components with explained variance ratios
- **Analysis**: Shows how much variance each component captures
- **Visualization**: Scree plot showing cumulative variance explained

### 3. Gradient Boosted Trees (GBT)

Predicts future returns using technical features (lagged returns, moving averages, volatility).

- **Features**: 5 engineered features from OHLCV data
- **Evaluation**: RMSE on 20% test set
- **Output**: Feature importance rankings
- **Visualization**: Horizontal bar chart of feature importance

### Model Comparison

The pipeline generates outputs for comparing:

1. **Baseline**: Correlation-filtered + Sharpe-weighted index
2. **K-Means**: Cluster-representative index
3. **PCA**: Factor-based index (components available)
4. **GBT**: Prediction-weighted index (future work)
