# Crypto Smart Index Engine

A scalable Big Data analytics engine built with **Apache Spark** to construct a **"Smart Beta" cryptocurrency index**.

Unlike standard market-cap weighted indices (which are dominated by highly correlated assets like BTC/ETH), this engine mathematically identifies a basket of assets that are **uncorrelated** and **risk-efficient**.

## 🧠 The Strategy
The core objective is to maximize the **Sharpe Ratio** (Risk-Adjusted Return) of the portfolio through a linear data pipeline:

1.  **Ingestion & Alignment**: Ingests ragged time-series data for 100+ pairs (millions of rows) into a unified Master Matrix.
2.  **Performance Scoring**: Calculates `Log Returns`, `Volatility` (Risk), and `Sharpe Ratio` for every asset using distributed Window Functions.
3.  **Correlation "Tournament"**:
    *   Computes a massive Pearson Correlation Matrix (O(N²) complexity).
    *   Identifies pairs with correlation > **0.85**.
    *   **Prunes** the redundant asset with the lower Sharpe Ratio.
4.  **Portfolio Weighting**: Constructs the final portfolio using **Inverse Volatility Weighting** to prioritize stability.

## 🛠️ Technology Stack
*   **Apache Spark (PySpark)**: Distributed processing for ETL and Matrix operations.
*   **Parquet**: Columnar storage for efficient I/O.
*   **Pandas/PyArrow**: Specific localized verification tasks.

## 🚀 Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
Downloads Binance 4h-kline data and converts it to partitioned Parquet for high-performance reading.
```bash
python download_crypto_data.py
```

### 3. Run Pipeline
**Option A: Production (Spark Cluster/Linux)**
Scalable implementation. Use this for the full dataset on a university server or cluster.
```bash
spark-submit spark_crypto_index.py
# OR
python spark_crypto_index.py
```

**Option B: Local Verification (Windows/Mac)**
Pandas-based proxy for quick local results on machines without Hadoop binaries.
```bash
python pandas_crypto_index.py
```

## 📊 Output
- `output/smart_index.csv`: Final portfolio weights and metrics.
- `output/correlation_matrix.png`: Visual heatmap of the market structure before filtering.
