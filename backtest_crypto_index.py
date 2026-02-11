"""
Crypto Index Backtester - Monthly Rebalancing Engine
=====================================================
Simulates 5 portfolio strategies with monthly rebalancing.

Strategies:
A. BTC buy-and-hold benchmark
B. Correlation-based tournament filtering (top 20 assets, inverse-vol weights)
C. K-Means clustering (cluster-balanced top 20 assets)
D. PCA residual variance filtering (top 20 most market-independent assets)
E. Gradient-Boosted Trees (GBT) return prediction

Features:
- Monthly rebalancing using an exact rolling 90-day lookback window
- Pure Spark implementation (no pandas in hot path)
- Performance metrics: total return, Sharpe, max drawdown, win rate
- Visualization: equity curves, monthly returns, drawdowns

Usage:
    python backtest_crypto_index.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data source
DATA_SOURCE = "crypto_data_parquet"
OUTPUT_FOLDER = "output"

# Backtest parameters
INITIAL_CAPITAL = 10000.0
LOOKBACK_DAYS = 90
BACKTEST_YEAR = 2025
BACKTEST_START_MONTH = 1
BACKTEST_END_MONTH = 12

# Strategy parameters
CORRELATION_THRESHOLD = 0.85
TOP_N_ASSETS = 20  # Standardized portfolio size for ML strategies A/B/C/E
N_CLUSTERS = 5
N_PCA_COMPONENTS = 10
TOP_PCA_ASSETS = TOP_N_ASSETS
TOP_GBT_ASSETS = TOP_N_ASSETS
# IMPORTANT: Keep full universe for ingestion/selection, but cap only the
# wide-matrix operators (pivot + VectorAssembler) in strategies B/D.
# Without this, Spark/Janino can fail with:
# "InternalCompilerException: Code grows beyond 64 KB"
# when using very large symbol universes (e.g., historical 500+ pairs).
MAX_WIDE_MATRIX_SYMBOLS = 150

# Constants
CANDLES_PER_DAY = 6  # 4-hour candles
ANNUALIZATION_FACTOR = 365 * CANDLES_PER_DAY
# Keep output order fixed as A -> B -> C -> D -> E.
STRATEGY_DISPLAY_ORDER = ['A', 'B', 'C', 'D', 'E']

# =============================================================================
# SPARK SESSION
# =============================================================================

import findspark
try:
    findspark.init()
except:
    pass

def create_spark_session():
    """Initialize Spark session with optimized settings for backtesting."""
    return (SparkSession.builder
            .appName("CryptoIndexBacktester")
            .config("spark.driver.memory", "6g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "20")
            .config("spark.sql.codegen.hugeMethodLimit", "8000")
            .getOrCreate())

def next_month_start(dt: datetime) -> datetime:
    """Return first day of the next month for a given datetime."""
    if dt.month == 12:
        return datetime(dt.year + 1, 1, 1)
    return datetime(dt.year, dt.month + 1, 1)

def filter_by_time_window(df, start_ts: datetime, end_ts: datetime):
    """Filter a dataframe to [start_ts, end_ts) based on timestamp column."""
    return df.filter(
        (F.col("timestamp") >= F.lit(start_ts)) &
        (F.col("timestamp") < F.lit(end_ts))
    )

def select_top_candidate_symbols(metrics_pdf: pd.DataFrame, max_symbols: int) -> List[str]:
    """
    Select highest-Sharpe symbols with valid volatility for wide-matrix steps.
    """
    if metrics_pdf.empty:
        return []

    clean = metrics_pdf.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["symbol", "sharpe", "annual_vol"])
    clean = clean[clean["annual_vol"] > 0]
    clean = clean.sort_values("sharpe", ascending=False)
    return clean["symbol"].head(max_symbols).tolist()

def build_inverse_vol_top_sharpe_portfolio(metrics_pdf: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Fallback portfolio constructor:
    - top Sharpe assets
    - inverse-volatility weights
    """
    if metrics_pdf.empty:
        return pd.DataFrame(columns=["symbol", "weight", "sharpe", "annual_vol"])

    clean = metrics_pdf.copy().replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=["symbol", "sharpe", "annual_vol"])
    clean = clean[clean["annual_vol"] > 0]
    clean = clean.sort_values("sharpe", ascending=False).head(top_n)
    if clean.empty:
        return pd.DataFrame(columns=["symbol", "weight", "sharpe", "annual_vol"])

    clean["inv_vol"] = 1.0 / clean["annual_vol"]
    inv_sum = clean["inv_vol"].sum()
    if inv_sum <= 0:
        return pd.DataFrame(columns=["symbol", "weight", "sharpe", "annual_vol"])

    clean["weight"] = clean["inv_vol"] / inv_sum
    return clean[["symbol", "weight", "sharpe", "annual_vol"]]

# =============================================================================
# STAGE 1: DATA INGESTION
# =============================================================================

def stage1_ingest_data(spark):
    """
    Load all available 4-hour data and create month_id column for reporting.

    Month mapping:
    - Oct 2024: month_id = -2 (warmup only, not traded)
    - Nov 2024: month_id = -1 (warmup only, not traded)
    - Dec 2024: month_id = 0 (lookback only, not traded)
    - Jan 2025: month_id = 1 (first backtest month)
    - ...
    - Dec 2025: month_id = 12 (last backtest month)

    Returns:
        long_df: DataFrame with (timestamp, symbol, close, month_id)
        symbols: List of all unique symbols
    """
    print("\n" + "="*60)
    print("STAGE 1: DATA INGESTION")
    print("="*60)

    if not os.path.exists(DATA_SOURCE):
        raise FileNotFoundError(f"Data source not found: {DATA_SOURCE}")

    # Read all partitioned data
    df = spark.read.parquet(DATA_SOURCE)

    # Create month_id: Oct 2024 = -2, Nov 2024 = -1, Dec 2024 = 0, Jan 2025 = 1, ..., Dec 2025 = 12
    df = df.withColumn(
        "month_id",
        F.when(F.col("year") == 2024, F.col("month") - 12)  # Oct=10-12=-2, Nov=11-12=-1, Dec=12-12=0
         .otherwise(F.col("month"))  # Jan=1, Feb=2, ..., Dec=12
    )

    # Select necessary columns
    long_df = df.select(
        F.col("timestamp"),
        F.col("symbol"),
        F.col("close").cast("double"),
        F.col("month_id")
    ).filter(F.col("close").isNotNull())

    # Cache for reuse
    long_df = long_df.cache()

    # Statistics
    count = long_df.count()
    symbols = sorted([r.symbol for r in long_df.select("symbol").distinct().collect()])
    month_counts = long_df.groupBy("month_id").count().orderBy("month_id").collect()

    print(f"\nLoaded {count:,} rows across {len(symbols)} assets")
    print(f"Data spans {len(month_counts)} months (month_id -2 to 12)")
    print("\nMonth distribution:")
    for row in month_counts:
        month_map = {
            -2: "Oct 2024", -1: "Nov 2024", 0: "Dec 2024",
            1: "Jan 2025", 2: "Feb 2025", 3: "Mar 2025", 4: "Apr 2025",
            5: "May 2025", 6: "Jun 2025", 7: "Jul 2025", 8: "Aug 2025",
            9: "Sep 2025", 10: "Oct 2025", 11: "Nov 2025", 12: "Dec 2025"
        }
        month_name = month_map.get(row.month_id, f"Month {row.month_id}")
        print(f"  Month {row.month_id:2d} ({month_name:10s}): {row['count']:,} rows")

    return long_df, symbols

# =============================================================================
# STAGE 2: MONTHLY WINDOWING LOGIC
# =============================================================================

def stage2_create_rebalance_windows():
    """
    Create mapping of:
    (backtest_month, lookback_start_ts, lookback_end_ts, holding_start_ts, holding_end_ts)

    Backtest window:
    - Rebalance monthly from Jan 2025 to Dec 2025
    - Use exact 90-day lookback ending at each month start
    - Hold through the full rebalance month

    Returns:
        List of (month, lookback_start, lookback_end, holding_start, holding_end) tuples
    """
    print("\n" + "="*60)
    print("STAGE 2: MONTHLY WINDOWING")
    print("="*60)

    windows = []
    for month in range(BACKTEST_START_MONTH, BACKTEST_END_MONTH + 1):
        holding_start = datetime(BACKTEST_YEAR, month, 1)
        holding_end = next_month_start(holding_start)
        lookback_end = holding_start
        lookback_start = holding_start - timedelta(days=LOOKBACK_DAYS)
        windows.append((month, lookback_start, lookback_end, holding_start, holding_end))

    print(f"\nCreated {len(windows)} rebalance windows ({LOOKBACK_DAYS}-day lookback):")
    print("  Month  | Lookback Range             | Holding Range")
    print("  " + "-" * 75)
    for month, lb_start, lb_end, hold_start, hold_end in windows[:3]:
        print(
            f"  {month:>2d}     | {lb_start:%Y-%m-%d} -> {(lb_end - timedelta(days=1)):%Y-%m-%d} | "
            f"{hold_start:%Y-%m-%d} -> {(hold_end - timedelta(days=1)):%Y-%m-%d}"
        )
    if len(windows) > 3:
        print("   ...    |             ...            |      ...")
        month, lb_start, lb_end, hold_start, hold_end = windows[-1]
        print(
            f"  {month:>2d}     | {lb_start:%Y-%m-%d} -> {(lb_end - timedelta(days=1)):%Y-%m-%d} | "
            f"{hold_start:%Y-%m-%d} -> {(hold_end - timedelta(days=1)):%Y-%m-%d}"
        )

    return windows

# =============================================================================
# HELPER: CALCULATE METRICS FOR LOOKBACK PERIOD
# =============================================================================

def calculate_metrics(long_df, lookback_start, lookback_end):
    """
    Calculate returns, volatility, and Sharpe ratio for assets in a lookback window.

    Args:
        long_df: Full dataset
        lookback_start: Start of lookback window (inclusive)
        lookback_end: End of lookback window (exclusive)

    Returns:
        DataFrame with columns: (symbol, annual_ret, annual_vol, sharpe, count)
    """
    lookback_data = filter_by_time_window(long_df, lookback_start, lookback_end)

    # Calculate log returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = lookback_data.withColumn("prev_close", F.lag("close").over(window))
    df = df.withColumn("log_ret", F.log(F.col("close") / F.col("prev_close")))
    df = df.filter(F.col("log_ret").isNotNull())

    # Aggregate metrics per symbol
    metrics = df.groupBy("symbol").agg(
        F.count("log_ret").alias("count"),
        F.sum("log_ret").alias("sum_log_ret"),
        F.stddev("log_ret").alias("std_log_ret")
    )

    # Annualize
    ann_factor = ANNUALIZATION_FACTOR
    metrics = metrics.withColumn(
        "annual_ret",
        F.col("sum_log_ret") * (ann_factor / F.col("count"))
    )
    metrics = metrics.withColumn(
        "annual_vol",
        F.col("std_log_ret") * np.sqrt(ann_factor)
    )
    metrics = metrics.withColumn(
        "sharpe",
        F.col("annual_ret") / F.col("annual_vol")
    )

    # Filter minimum data points
    metrics = metrics.filter(F.col("count") > 30)

    return metrics

# =============================================================================
# HELPER: CALCULATE CORRELATION MATRIX
# =============================================================================

def calculate_correlation_matrix(long_df, lookback_start, lookback_end, symbols):
    """
    Calculate Pearson correlation matrix on returns for a lookback window.

    Returns:
        corr_mat: NumPy correlation matrix
        valid_cols: List of symbols (column order)
    """
    if len(symbols) < 2:
        return np.empty((0, 0)), []

    lookback_data = filter_by_time_window(long_df, lookback_start, lookback_end)

    # Calculate returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = lookback_data.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()).select("timestamp", "symbol", "ret")

    # Pivot returns matrix
    wide_ret = df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("ret"))
    wide_ret = wide_ret.fillna(0)  # Forward fill (0 return = constant price)

    # VectorAssembler
    valid_cols = [c for c in symbols if c in wide_ret.columns]
    if len(valid_cols) < 2:
        return np.empty((0, 0)), []
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")

    # Correlation
    corr_mat = Correlation.corr(vec_df, "features", "pearson").head()[0].toArray()

    return corr_mat, valid_cols

# =============================================================================
# STAGE 3A: PORTFOLIO A - CORRELATION-BASED STRATEGY
# =============================================================================

def stage3a_portfolio_correlation(long_df, symbols, lookback_start, lookback_end):
    """
    Correlation-based tournament filtering strategy.

    Algorithm:
    1. Calculate metrics (returns, volatility, Sharpe) over a 90-day window
    2. Build correlation matrix on returns
    3. Apply tournament filter: for each highly correlated pair (|ρ| > 0.85),
       drop the asset with lower Sharpe ratio
    4. Select top 20 survivors by Sharpe
    5. Apply inverse-volatility weights

    Returns:
        Pandas DataFrame with (symbol, weight, sharpe, annual_vol)
    """
    # Calculate metrics over lookback window
    metrics = calculate_metrics(long_df, lookback_start, lookback_end)
    metrics_pdf = metrics.toPandas()
    if metrics_pdf.empty:
        return pd.DataFrame(columns=["symbol", "weight", "sharpe", "annual_vol"])
    pdf = metrics_pdf.set_index("symbol")

    # Cap wide-matrix correlation universe for Spark codegen stability
    corr_candidates = select_top_candidate_symbols(metrics_pdf, MAX_WIDE_MATRIX_SYMBOLS)
    corr_candidates = [s for s in corr_candidates if s in symbols]
    corr_mat, valid_symbols = calculate_correlation_matrix(
        long_df, lookback_start, lookback_end, corr_candidates
    )
    if len(valid_symbols) == 0:
        return build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_N_ASSETS)

    # Tournament filter
    idx_to_sym = {i: s for i, s in enumerate(valid_symbols)}
    sym_to_idx = {s: i for i, s in enumerate(valid_symbols)}

    keep = set(valid_symbols)
    dropped = set()

    rows, cols = corr_mat.shape
    for i in range(rows):
        for j in range(i+1, cols):
            if abs(corr_mat[i, j]) > CORRELATION_THRESHOLD:
                s1 = idx_to_sym[i]
                s2 = idx_to_sym[j]

                if s1 in dropped or s2 in dropped:
                    continue

                # Compare Sharpe
                if s1 not in pdf.index or s2 not in pdf.index:
                    continue

                sh1 = pdf.loc[s1, "sharpe"]
                sh2 = pdf.loc[s2, "sharpe"]

                if sh1 < sh2:
                    dropped.add(s1)
                    if s1 in keep: keep.remove(s1)
                else:
                    dropped.add(s2)
                    if s2 in keep: keep.remove(s2)

    # Select top N by Sharpe (only from assets in pdf index)
    available_symbols = [s for s in keep if s in pdf.index]
    final_df = pdf.loc[available_symbols].copy()
    final_df = final_df.sort_values("sharpe", ascending=False).head(TOP_N_ASSETS)
    final_df = final_df.replace([np.inf, -np.inf], np.nan)
    final_df = final_df.dropna(subset=["annual_vol", "sharpe"])
    final_df = final_df[final_df["annual_vol"] > 0]
    if final_df.empty:
        return build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_N_ASSETS)

    # Inverse volatility weights
    final_df["inv_vol"] = 1.0 / final_df["annual_vol"]
    inv_sum = final_df["inv_vol"].sum()
    if inv_sum <= 0:
        return build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_N_ASSETS)
    final_df["weight"] = final_df["inv_vol"] / inv_sum

    # Return as Pandas DataFrame
    result_df = final_df.reset_index()[["symbol", "weight", "sharpe", "annual_vol"]]

    return result_df

# =============================================================================
# STAGE 3B: PORTFOLIO B - K-MEANS CLUSTERING STRATEGY
# =============================================================================

def stage3b_portfolio_kmeans(long_df, lookback_start, lookback_end):
    """
    K-Means clustering strategy.

    Algorithm:
    1. Calculate risk/return metrics over a 90-day window
    2. Standardize features: [annual_ret, annual_vol, sharpe]
    3. Train K-Means with k=5
    4. Rank assets by Sharpe within each cluster
    5. Select a cluster-balanced top 20 using round-robin across clusters
    6. Apply inverse-volatility weights

    Returns:
        DataFrame with (symbol, weight, cluster, sharpe, annual_vol)
    """
    # Calculate metrics over lookback window
    metrics = calculate_metrics(long_df, lookback_start, lookback_end)

    # Prepare features
    features = ["annual_ret", "annual_vol", "sharpe"]
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    df_features = assembler.transform(metrics)

    # Standardize
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df_features)
    df_scaled = scaler_model.transform(df_features)

    # Train K-Means
    kmeans = KMeans(k=N_CLUSTERS, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df_scaled)

    # Predict clusters
    clustered = model.transform(df_scaled)

    # Rank assets within each cluster by Sharpe
    window = Window.partitionBy("cluster").orderBy(F.desc("sharpe"))
    ranked = clustered.withColumn("rank", F.row_number().over(window)) \
                     .select("symbol", "cluster", "sharpe", "annual_vol", "rank")

    ranked_pdf = ranked.toPandas()
    if ranked_pdf.empty:
        return pd.DataFrame(columns=["symbol", "weight", "cluster", "sharpe", "annual_vol"])

    # Cluster-balanced selection: round-robin by cluster rank
    ranked_pdf = ranked_pdf.sort_values(["cluster", "rank"], ascending=[True, True])
    target_n = min(TOP_N_ASSETS, len(ranked_pdf))

    by_cluster = {
        c: g.reset_index(drop=True)
        for c, g in ranked_pdf.groupby("cluster", sort=True)
    }
    cluster_ids = sorted(by_cluster.keys())
    pos = {c: 0 for c in cluster_ids}

    selected_rows = []
    while len(selected_rows) < target_n:
        added = False
        for c in cluster_ids:
            idx = pos[c]
            group = by_cluster[c]
            if idx < len(group):
                selected_rows.append(group.iloc[idx])
                pos[c] += 1
                added = True
                if len(selected_rows) >= target_n:
                    break
        if not added:
            break

    selected_pdf = pd.DataFrame(selected_rows)
    selected_pdf = selected_pdf[np.isfinite(selected_pdf["annual_vol"])]
    selected_pdf = selected_pdf[selected_pdf["annual_vol"] > 0].copy()
    if selected_pdf.empty:
        return pd.DataFrame(columns=["symbol", "weight", "cluster", "sharpe", "annual_vol"])

    selected_pdf["inv_vol"] = 1.0 / selected_pdf["annual_vol"]
    inv_vol_sum = selected_pdf["inv_vol"].sum()
    if inv_vol_sum <= 0:
        return pd.DataFrame(columns=["symbol", "weight", "cluster", "sharpe", "annual_vol"])
    selected_pdf["weight"] = selected_pdf["inv_vol"] / inv_vol_sum

    return selected_pdf[["symbol", "weight", "cluster", "sharpe", "annual_vol"]]

# =============================================================================
# STAGE 3C: PORTFOLIO C - PCA RESIDUAL STRATEGY
# =============================================================================

def stage3c_portfolio_pca_residual(long_df, symbols, lookback_start, lookback_end):
    """
    PCA Residual Variance strategy - select LOW market correlation assets.

    Algorithm:
    1. Build returns matrix over a 90-day window (timestamp × symbol pivot)
       from a capped candidate set (max 150) for Spark stability
    2. Apply PCA with k=10 components
    3. Calculate variance explained by PC1 for each asset
    4. Residual variance = total variance - PC1 explained variance
    5. Select top 20 assets by HIGHEST residual variance (most independent)
    6. Apply inverse-volatility weights

    Rationale:
    Assets with HIGH residual variance don't move with the market (PC1).
    These provide true diversification and defensive protection in downturns.
    When market crashes, low-correlation assets are less affected.

    Returns:
        DataFrame with (symbol, weight, residual_var, pc1_loading, sharpe, annual_vol)
    """
    metrics = calculate_metrics(long_df, lookback_start, lookback_end)
    metrics_pdf = metrics.toPandas()
    if metrics_pdf.empty:
        return pd.DataFrame(columns=["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"])

    # Cap PCA universe for Spark codegen stability on large symbol sets.
    pca_candidates = select_top_candidate_symbols(metrics_pdf, MAX_WIDE_MATRIX_SYMBOLS)
    pca_candidates = [s for s in pca_candidates if s in symbols]
    if len(pca_candidates) < 2:
        fallback = build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_PCA_ASSETS)
        fallback = fallback.assign(residual_var=np.nan, pc1_loading=np.nan)
        return fallback[["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"]]

    lookback_data = filter_by_time_window(long_df, lookback_start, lookback_end)

    # Calculate returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = lookback_data.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()) \
           .filter(F.col("symbol").isin(pca_candidates)) \
           .select("timestamp", "symbol", "ret")

    # Pivot returns
    wide_ret = df.groupBy("timestamp").pivot("symbol", pca_candidates).agg(F.first("ret"))
    wide_ret = wide_ret.fillna(0)

    # VectorAssembler
    valid_cols = [c for c in pca_candidates if c in wide_ret.columns]
    if len(valid_cols) < 2:
        fallback = build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_PCA_ASSETS)
        fallback = fallback.assign(residual_var=np.nan, pc1_loading=np.nan)
        return fallback[["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"]]
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")

    # Apply PCA
    pca = PCA(k=min(N_PCA_COMPONENTS, len(valid_cols)), inputCol="features", outputCol="pca_features")
    try:
        model = pca.fit(vec_df)
    except Exception:
        fallback = build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_PCA_ASSETS)
        fallback = fallback.assign(residual_var=np.nan, pc1_loading=np.nan)
        return fallback[["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"]]

    # Get PC1 loadings and explained variance
    # pc.shape = (n_features, n_components) - transposed compared to some implementations
    pc_matrix = model.pc.toArray()
    pc1_loadings = pc_matrix[:, 0]  # First principal component loadings

    # Calculate variance explained by PC1 for each asset
    # PC1 loading^2 = proportion of asset's variance explained by PC1
    explained_variance_ratio = pc1_loadings ** 2

    # Residual variance = variance NOT explained by PC1
    # Higher residual = more independent from market factor
    residual_variance = 1.0 - explained_variance_ratio

    # Create DataFrame with residual variance
    residual_df = pd.DataFrame({
        'symbol': valid_cols,
        'pc1_loading': pc1_loadings,
        'explained_var': explained_variance_ratio,
        'residual_var': residual_variance
    })

    # Select top N by HIGHEST residual variance (most market-independent)
    residual_df = residual_df.sort_values('residual_var', ascending=False).head(TOP_PCA_ASSETS)

    # Merge with metrics
    result_df = residual_df.merge(metrics_pdf[['symbol', 'sharpe', 'annual_vol']],
                                   on='symbol', how='left')
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.dropna(subset=['annual_vol', 'sharpe'])
    result_df = result_df[result_df['annual_vol'] > 0]
    if result_df.empty:
        fallback = build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_PCA_ASSETS)
        fallback = fallback.assign(residual_var=np.nan, pc1_loading=np.nan)
        return fallback[["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"]]

    # Apply inverse-volatility weights
    result_df['inv_vol'] = 1.0 / result_df['annual_vol']
    inv_sum = result_df['inv_vol'].sum()
    if inv_sum <= 0:
        fallback = build_inverse_vol_top_sharpe_portfolio(metrics_pdf, TOP_PCA_ASSETS)
        fallback = fallback.assign(residual_var=np.nan, pc1_loading=np.nan)
        return fallback[["symbol", "weight", "residual_var", "pc1_loading", "sharpe", "annual_vol"]]
    result_df['weight'] = result_df['inv_vol'] / inv_sum

    return result_df[['symbol', 'weight', 'residual_var', 'pc1_loading', 'sharpe', 'annual_vol']]

# =============================================================================
# STAGE 3D: PORTFOLIO D - BTC BUY & HOLD BENCHMARK
# =============================================================================

def stage3d_portfolio_btc():
    """
    Simple buy-and-hold BTC benchmark.

    Algorithm:
    1. Allocate 100% to BTCUSDT
    2. Hold for entire backtest period (no rebalancing)

    Returns:
        DataFrame with (symbol, weight)
    """
    return pd.DataFrame({
        'symbol': ['BTCUSDT'],
        'weight': [1.0]
    })

# =============================================================================
# STAGE 3E: PORTFOLIO E - GBT PREDICTION STRATEGY
# =============================================================================

def engineer_features_for_gbt(long_df, lookback_start, lookback_end):
    """Create technical features from price data for GBT prediction."""
    lookback_data = filter_by_time_window(long_df, lookback_start, lookback_end)

    window = Window.partitionBy("symbol").orderBy("timestamp")

    # Lagged returns
    df = lookback_data.withColumn("prev_close", F.lag("close").over(window))
    df = df.withColumn("log_ret", F.log(F.col("close") / F.col("prev_close")))
    df = df.withColumn("ret_lag1", F.lag("log_ret", 1).over(window))
    df = df.withColumn("ret_lag2", F.lag("log_ret", 2).over(window))
    df = df.withColumn("ret_lag3", F.lag("log_ret", 3).over(window))

    # Moving averages (7 and 30 periods)
    ma7 = F.avg("close").over(window.rowsBetween(-6, 0))
    ma30 = F.avg("close").over(window.rowsBetween(-29, 0))
    df = df.withColumn("ma7", ma7)
    df = df.withColumn("ma30", ma30)
    df = df.withColumn("ma_ratio", ma7 / ma30)

    # Rolling volatility (7-period)
    vol7 = F.stddev("log_ret").over(window.rowsBetween(-6, 0))
    df = df.withColumn("volatility_7d", vol7)

    # Target: next period return
    df = df.withColumn("target", F.lead("log_ret", 1).over(window))

    # Drop nulls
    df = df.dropna()

    feature_cols = ["ret_lag1", "ret_lag2", "ret_lag3", "ma_ratio", "volatility_7d"]

    return df, feature_cols

def stage3e_portfolio_gbt(long_df, lookback_start, lookback_end):
    """
    GBT prediction-based strategy.

    Algorithm:
    1. Engineer technical features over a 90-day window (lagged returns, moving averages, volatility)
    2. Train GBT Regressor to predict next-period returns
    3. Predict expected returns for all assets
    4. Select top N assets by predicted return
    5. Weight by normalized predicted return (higher prediction = higher weight)

    Returns:
        DataFrame with (symbol, weight, predicted_return)
    """
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.feature import VectorAssembler

    try:
        # Engineer features over lookback window
        df, feature_cols = engineer_features_for_gbt(long_df, lookback_start, lookback_end)

        # Assemble features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df = assembler.transform(df).select("symbol", "features", "target")

        # Train GBT
        gbt = GBTRegressor(featuresCol="features", labelCol="target",
                          maxIter=20, maxDepth=5, seed=42)
        model = gbt.fit(df)

        # Make predictions on all data
        predictions = model.transform(df.select("symbol", "features"))

        # Aggregate predictions by symbol (take mean prediction)
        symbol_predictions = predictions.groupBy("symbol").agg(
            F.mean("prediction").alias("predicted_return")
        )

        # Convert to pandas for portfolio construction
        pred_pdf = symbol_predictions.toPandas()

        # Select top N by predicted return
        pred_pdf = pred_pdf.sort_values('predicted_return', ascending=False).head(TOP_GBT_ASSETS)

        # Weight by normalized predicted return (shift to make all positive)
        # Add constant to ensure all weights are positive
        min_pred = pred_pdf['predicted_return'].min()
        if min_pred < 0:
            pred_pdf['adj_pred'] = pred_pdf['predicted_return'] - min_pred + 0.001
        else:
            pred_pdf['adj_pred'] = pred_pdf['predicted_return'] + 0.001

        pred_pdf['weight'] = pred_pdf['adj_pred'] / pred_pdf['adj_pred'].sum()

        return pred_pdf[['symbol', 'weight', 'predicted_return']]

    except Exception as e:
        # If GBT fails, return empty DataFrame
        return pd.DataFrame({'symbol': [], 'weight': [], 'predicted_return': []})

# =============================================================================
# STAGE 4: BACKTEST EXECUTION
# =============================================================================

def stage4_backtest_month(long_df, weights_pdf, holding_start, holding_end):
    """
    Execute backtest for a single month.

    Execution policy (strict, no optimistic fills):
    1) Tradability at rebalance:
       A symbol is investable only if it has an exact 4h candle at
       `holding_start` (month start rebalance timestamp).
       If missing, we do NOT allocate to it.
    2) Delisting / missing month-end bar:
       If a selected symbol has no exact 4h candle at `holding_end - 4h`,
       we assign 0% monthly return for that symbol
       (implemented via close_end := close_start).

    Implication:
    Final invested assets can be < target size (e.g. <20) in a month when
    some selected symbols are not tradable exactly at rebalance time.

    Args:
        long_df: Full dataset
        weights_pdf: Pandas DataFrame with (symbol, weight)
        holding_start: Start timestamp of holding month (inclusive)
        holding_end: End timestamp of holding month (exclusive)

    Returns:
        month_return: Portfolio return for the month
        num_assets: Number of assets in portfolio
    """
    month_data = filter_by_time_window(long_df, holding_start, holding_end)
    end_bar_ts = holding_end - timedelta(hours=4)

    # ---------------------------------------------------------------------
    # Rule 2 (requested): only invest in symbols tradable AT REBALANCE DATE.
    # We define tradable as having a 4h close exactly at holding_start.
    # Symbols without this start bar are removed before capital allocation.
    # ---------------------------------------------------------------------
    start_prices = month_data.filter(F.col("timestamp") == F.lit(holding_start)) \
                            .select("symbol", F.col("close").alias("close_start"))
    start_pdf = start_prices.toPandas()
    if start_pdf.empty:
        return 0.0, 0

    tradable = weights_pdf.merge(start_pdf, on="symbol", how="inner")
    if tradable.empty:
        return 0.0, 0

    # Re-normalize after removing non-tradable-at-rebalance symbols.
    weight_sum = tradable["weight"].sum()
    if weight_sum <= 0:
        return 0.0, 0
    tradable["weight"] = tradable["weight"] / weight_sum

    # ---------------------------------------------------------------------
    # Rule 1 (requested): if a selected symbol gets delisted / disappears
    # before month-end, assign 0% return for that month.
    # We enforce this by requiring an exact close on the final 4h bar.
    # Missing close_end -> close_end = close_start -> return = 0%.
    # ---------------------------------------------------------------------
    end_prices = month_data.filter(F.col("timestamp") == F.lit(end_bar_ts)) \
                          .select("symbol", F.col("close").alias("close_end"))
    end_pdf = end_prices.toPandas()

    portfolio = tradable.merge(end_pdf, on="symbol", how="left")
    portfolio["close_end"] = portfolio["close_end"].fillna(portfolio["close_start"])

    # Defensive cleanup for invalid prices
    portfolio = portfolio[np.isfinite(portfolio["close_start"])]
    portfolio = portfolio[np.isfinite(portfolio["close_end"])]
    portfolio = portfolio[portfolio["close_start"] > 0]
    if portfolio.empty:
        return 0.0, 0

    portfolio["return"] = (portfolio["close_end"] - portfolio["close_start"]) / portfolio["close_start"]
    portfolio_return = (portfolio["weight"] * portfolio["return"]).sum()

    return portfolio_return, len(portfolio)

def stage4_run_backtest(long_df, symbols, rebalance_windows):
    """
    Run full backtest across all months and all strategies.

    Returns:
        results_df: Pandas DataFrame with monthly results
    """
    print("\n" + "="*60)
    print("STAGE 4: BACKTEST EXECUTION")
    print("="*60)
    print("Execution rules:")
    print("  - Invest only if symbol has exact 4h price at rebalance timestamp.")
    print("  - Missing month-end exact 4h price => symbol monthly return = 0%.")
    print("  - Therefore realized asset count may be below target size.")

    results = []

    for backtest_month, lookback_start, lookback_end, holding_start, holding_end in rebalance_windows:
        month_name = holding_start.strftime("%b %Y")
        print(f"\nProcessing Month {backtest_month}/12 ({month_name})...")
        print(
            f"  Lookback: {lookback_start:%Y-%m-%d} -> {(lookback_end - timedelta(days=1)):%Y-%m-%d}, "
            f"Holding: {holding_start:%Y-%m-%d} -> {(holding_end - timedelta(days=1)):%Y-%m-%d}"
        )

        # Generate portfolios using lookback window
        print("  Generating portfolios...")

        monthly_stats = {}

        # Strategy A: BTC Buy & Hold benchmark
        try:
            weights_a = stage3d_portfolio_btc()
            ret_a, n_a = stage4_backtest_month(long_df, weights_a, holding_start, holding_end)
            print(f"    Portfolio A (BTC): {n_a} asset, return: {ret_a:+.4f}")
        except Exception as e:
            print(f"    Portfolio A FAILED: {e}")
            ret_a, n_a = 0.0, 0
        monthly_stats['A'] = (ret_a, n_a)

        # Strategy B: Correlation-based
        try:
            weights_b = stage3a_portfolio_correlation(long_df, symbols, lookback_start, lookback_end)
            ret_b, n_b = stage4_backtest_month(long_df, weights_b, holding_start, holding_end)
            print(f"    Portfolio B: {n_b} assets, return: {ret_b:+.4f}")
        except Exception as e:
            print(f"    Portfolio B FAILED: {e}")
            ret_b, n_b = 0.0, 0
        monthly_stats['B'] = (ret_b, n_b)

        # Strategy C: K-Means
        try:
            weights_c = stage3b_portfolio_kmeans(long_df, lookback_start, lookback_end)
            ret_c, n_c = stage4_backtest_month(long_df, weights_c, holding_start, holding_end)
            print(f"    Portfolio C: {n_c} assets, return: {ret_c:+.4f}")
        except Exception as e:
            print(f"    Portfolio C FAILED: {e}")
            ret_c, n_c = 0.0, 0
        monthly_stats['C'] = (ret_c, n_c)

        # Strategy D: PCA Residual Variance
        try:
            weights_d = stage3c_portfolio_pca_residual(long_df, symbols, lookback_start, lookback_end)
            ret_d, n_d = stage4_backtest_month(long_df, weights_d, holding_start, holding_end)
            print(f"    Portfolio D: {n_d} assets, return: {ret_d:+.4f}")
        except Exception as e:
            print(f"    Portfolio D FAILED: {e}")
            ret_d, n_d = 0.0, 0
        monthly_stats['D'] = (ret_d, n_d)

        # Strategy E: GBT Prediction
        try:
            weights_e = stage3e_portfolio_gbt(long_df, lookback_start, lookback_end)
            if len(weights_e) > 0:
                ret_e, n_e = stage4_backtest_month(long_df, weights_e, holding_start, holding_end)
                print(f"    Portfolio E (GBT): {n_e} assets, return: {ret_e:+.4f}")
            else:
                ret_e, n_e = 0.0, 0
                print("    Portfolio E (GBT): FAILED (no predictions)")
        except Exception as e:
            print(f"    Portfolio E FAILED: {e}")
            ret_e, n_e = 0.0, 0
        monthly_stats['E'] = (ret_e, n_e)

        # Store results in the display order used by metrics/plots.
        for strategy in STRATEGY_DISPLAY_ORDER:
            ret, n_assets = monthly_stats[strategy]
            results.append({
                'month': backtest_month,
                'strategy': strategy,
                'month_return': ret,
                'num_assets': n_assets
            })

    results_df = pd.DataFrame(results)
    return results_df

# =============================================================================
# STAGE 5: PERFORMANCE METRICS
# =============================================================================

def stage5_calculate_performance(results_df):
    """
    Calculate performance metrics for each strategy.

    Metrics:
    - Total return
    - Annualized return
    - Annualized volatility
    - Sharpe ratio
    - Maximum drawdown
    - Win rate
    - Final portfolio value

    Returns:
        performance_df: Summary metrics by strategy
        equity_df: Portfolio values over time
    """
    print("\n" + "="*60)
    print("STAGE 5: PERFORMANCE METRICS")
    print("="*60)

    performance = []
    equity_data = []

    for strategy in STRATEGY_DISPLAY_ORDER:
        strategy_data = results_df[results_df['strategy'] == strategy].copy()
        strategy_data = strategy_data.sort_values('month')

        # Calculate cumulative portfolio value
        portfolio_value = INITIAL_CAPITAL
        values = [portfolio_value]

        for _, row in strategy_data.iterrows():
            portfolio_value *= (1 + row['month_return'])
            values.append(portfolio_value)

            equity_data.append({
                'month': row['month'],
                'strategy': strategy,
                'portfolio_value': portfolio_value,
                'month_return': row['month_return']
            })

        # Performance metrics
        returns = strategy_data['month_return'].values
        final_value = values[-1]

        total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # Use geometric mean for annualized return (accounts for compounding)
        n_months = len(returns)
        geometric_return = np.prod(1 + returns) ** (1/n_months) - 1
        ann_return = geometric_return * 12

        ann_vol = np.std(returns) * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Maximum drawdown
        cumulative = np.array(values)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Win rate
        win_rate = np.sum(returns > 0) / len(returns)

        performance.append({
            'strategy': strategy,
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_volatility': ann_vol,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'final_value': final_value
        })

        print(f"\nStrategy {strategy}:")
        print(f"  Final Value:     ${final_value:,.2f}")
        print(f"  Total Return:    {total_return*100:+.2f}%")
        print(f"  Ann. Return:     {ann_return*100:+.2f}%")
        print(f"  Ann. Volatility: {ann_vol*100:.2f}%")
        print(f"  Sharpe Ratio:    {sharpe:.2f}")
        print(f"  Max Drawdown:    {max_drawdown*100:.2f}%")
        print(f"  Win Rate:        {win_rate*100:.1f}%")

    performance_df = pd.DataFrame(performance)
    equity_df = pd.DataFrame(equity_data)

    return performance_df, equity_df

# =============================================================================
# STAGE 6: VISUALIZATION
# =============================================================================

def stage6_visualize(results_df, performance_df, equity_df, output_folder):
    """
    Create comprehensive visualization of backtest results.

    Plots:
    1. Equity curves
    2. Monthly returns heatmap
    3. Drawdown analysis (month-end underwater chart)
    4. Performance summary table
    """
    print("\n" + "="*60)
    print("STAGE 6: VISUALIZATION")
    print("="*60)

    # Create figure with 4 subplots - larger and clearer
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.35)

    # Define colors for strategies
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728', 'E': '#9467bd'}
    strategy_names = {
        'A': 'BTC Buy & Hold (Benchmark)',
        'B': 'Correlation-based (Top 20)',
        'C': 'K-Means Cluster-Balanced (Top 20)',
        'D': 'PCA Residual Variance (Top 20)',
        'E': 'GBT Prediction (Top 20)'
    }

    # Plot 1: Equity Curves
    ax1 = fig.add_subplot(gs[0, :])

    for strategy in STRATEGY_DISPLAY_ORDER:
        strategy_equity = equity_df[equity_df['strategy'] == strategy].sort_values('month')
        months = [0] + strategy_equity['month'].tolist()
        values = [INITIAL_CAPITAL] + strategy_equity['portfolio_value'].tolist()
        ax1.plot(months, values, marker='o', linewidth=3, markersize=6,
                label=strategy_names[strategy], color=colors[strategy], alpha=0.8)

    ax1.axhline(y=INITIAL_CAPITAL, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Initial Capital ($10k)')
    ax1.set_xlabel('Month (0=Start, 1=Jan, 2=Feb, ..., 12=Dec 2025)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12, fontweight='bold')
    ax1.set_title('Equity Curves - Monthly Rebalancing Backtest', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(0, 12)

    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

    # Plot 2: Monthly Returns
    ax2 = fig.add_subplot(gs[1, 0])

    strategy_order = STRATEGY_DISPLAY_ORDER
    month_order = sorted(results_df['month'].unique())
    returns_heatmap = (
        results_df.pivot(index='strategy', columns='month', values='month_return')
        .reindex(strategy_order)
        .reindex(columns=month_order)
        * 100
    )
    returns_heatmap.index = [strategy_names[s] for s in strategy_order]
    returns_heatmap.columns = [datetime(BACKTEST_YEAR, int(m), 1).strftime('%b') for m in month_order]

    sns.heatmap(
        returns_heatmap,
        ax=ax2,
        cmap='RdYlGn',
        center=0,
        annot=True,
        fmt='.1f',
        linewidths=0.6,
        linecolor='white',
        cbar_kws={'label': 'Monthly Return (%)'},
        annot_kws={'fontsize': 8}
    )
    ax2.set_xlabel('Month (2025)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Strategy', fontsize=12, fontweight='bold')
    ax2.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', labelrotation=0, labelsize=9)
    ax2.tick_params(axis='y', labelsize=8)

    # Plot 3: Drawdown Analysis
    ax3 = fig.add_subplot(gs[1, 1])

    for strategy in STRATEGY_DISPLAY_ORDER:
        strategy_equity = equity_df[equity_df['strategy'] == strategy].sort_values('month')
        values = np.array([INITIAL_CAPITAL] + strategy_equity['portfolio_value'].tolist())
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        months_plot = list(range(len(values)))
        ax3.plot(
            months_plot,
            drawdown,
            marker='o',
            linewidth=2.2,
            markersize=4,
            label=strategy_names[strategy],
            color=colors[strategy],
            alpha=0.9
        )

    ax3.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Drawdown (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Drawdown Analysis (Month-End Underwater)', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(loc='lower left', fontsize=8, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax3.set_xlim(0, 12)
    ax3.axhline(y=0, color='black', linewidth=1)
    ax3.set_xticks(range(0, 13))
    ax3.set_xticklabels(['Start', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=9)

    # Plot 4: Performance Summary Table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('tight')
    ax4.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Strategy', 'Final Value', 'Total Return', 'Ann. Return', 'Ann. Vol',
              'Sharpe', 'Max DD', 'Win Rate']

    for _, row in performance_df.iterrows():
        table_data.append([
            strategy_names[row['strategy']],
            f"${row['final_value']:,.0f}",
            f"{row['total_return']*100:+.2f}%",
            f"{row['ann_return']*100:+.2f}%",
            f"{row['ann_volatility']*100:.2f}%",
            f"{row['sharpe']:.2f}",
            f"{row['max_drawdown']*100:.2f}%",
            f"{row['win_rate']*100:.1f}%"
        ])

    table = ax4.table(cellText=table_data, colLabels=headers, cellLoc='center',
                     loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.35)

    # Wider first column to prevent clipped strategy names
    col_widths = [0.22, 0.11, 0.11, 0.11, 0.10, 0.08, 0.10, 0.09]
    for col, width in enumerate(col_widths):
        table[(0, col)].set_width(width)
        for row in range(1, len(table_data) + 1):
            table[(row, col)].set_width(width)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=11)
    table[(0, 0)].get_text().set_ha('left')

    # Color rows - highlight best strategy
    best_strategy = performance_df.loc[performance_df['sharpe'].idxmax(), 'strategy']
    for i, row in enumerate(performance_df.itertuples(index=False)):
        is_best = row.strategy == best_strategy
        bg_color = '#e8f5e9' if is_best else ('#f5f5f5' if i % 2 == 0 else 'white')
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(bg_color)
            if is_best:
                table[(i+1, j)].set_text_props(weight='bold')
        table[(i+1, 0)].get_text().set_ha('left')

    ax4.set_title('Performance Summary (12-Month Backtest)', fontsize=14, fontweight='bold', pad=20)

    # Save figure with higher DPI for clarity
    output_path = os.path.join(output_folder, 'backtest_comparison.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"\nVisualization saved: {output_path}")

    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution pipeline."""
    spark = None

    try:
        # Initialize Spark
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("ERROR")

        print("\n" + "="*60)
        print("CRYPTO INDEX BACKTESTER - MONTHLY REBALANCING")
        print("="*60)
        print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
        print(f"Backtest Period: Jan 2025 - Dec 2025 (12 months)")
        print(f"Lookback Window: {LOOKBACK_DAYS} days (~{LOOKBACK_DAYS * CANDLES_PER_DAY} observations with 4-hour data)")
        print(
            f"Wide-Matrix Cap (B/D only): {MAX_WIDE_MATRIX_SYMBOLS} symbols "
            f"(stability guard for Spark codegen on large universes)"
        )
        print("Strategies (display order): A (BTC), B (Correlation), C (K-Means), D (PCA Residual), E (GBT)")

        # Stage 1: Ingest data
        long_df, symbols = stage1_ingest_data(spark)

        # Stage 2: Create rebalance windows
        rebalance_windows = stage2_create_rebalance_windows()

        # Stage 3 & 4: Run backtest
        results_df = stage4_run_backtest(long_df, symbols, rebalance_windows)

        # Stage 5: Calculate performance
        performance_df, equity_df = stage5_calculate_performance(results_df)

        # Save results
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

        results_path = os.path.join(OUTPUT_FOLDER, 'backtest_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\n✅ Monthly results saved: {results_path}")

        performance_path = os.path.join(OUTPUT_FOLDER, 'backtest_performance.csv')
        performance_df.to_csv(performance_path, index=False)
        print(f"✅ Performance summary saved: {performance_path}")

        equity_path = os.path.join(OUTPUT_FOLDER, 'backtest_equity.csv')
        equity_df.to_csv(equity_path, index=False)
        print(f"✅ Equity curve data saved: {equity_path}")

        # Stage 6: Visualize
        stage6_visualize(results_df, performance_df, equity_df, OUTPUT_FOLDER)

        # Final summary
        print("\n" + "="*60)
        print("BACKTEST COMPLETE")
        print("="*60)
        print("\nBest Strategy:")
        best = performance_df.loc[performance_df['sharpe'].idxmax()]
        print(f"  {strategy_names[best['strategy']]}")
        print(f"  Sharpe Ratio: {best['sharpe']:.2f}")
        print(f"  Final Value: ${best['final_value']:,.2f}")
        print(f"  Total Return: {best['total_return']*100:+.2f}%")

        print("\n✅ All outputs saved to output/")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    # Define strategy names globally for table formatting
    strategy_names = {
        'A': 'BTC Buy & Hold (Benchmark)',
        'B': 'Correlation-based (Top 20)',
        'C': 'K-Means Cluster-Balanced (Top 20)',
        'D': 'PCA Residual Variance (Top 20)',
        'E': 'GBT Prediction (Top 20)'
    }
    main()
