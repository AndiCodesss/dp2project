"""
Crypto Index Backtester - Monthly Rebalancing Engine
=====================================================
Simulates 3 ML-based portfolio strategies with monthly rebalancing.

Strategies:
A. Correlation-based tournament filtering (top 20 assets, inverse-vol weights)
B. K-Means clustering (5 clusters, select top Sharpe from each)
C. PCA component loading (top 15 assets by PC1 loading magnitude)

Features:
- Monthly rebalancing using rolling 1-month lookback window
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
from datetime import datetime
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
LOOKBACK_MONTHS = 1  # Rolling window for weight calculation
BACKTEST_START_MONTH = 2   # February 2025 (month_id=2) - start here due to data availability
BACKTEST_END_MONTH = 12    # December 2025 (month_id=12)

# Strategy parameters
CORRELATION_THRESHOLD = 0.85
TOP_N_ASSETS = 20
N_CLUSTERS = 5
N_PCA_COMPONENTS = 10
TOP_PCA_ASSETS = 15

# Constants
CANDLES_PER_DAY = 6  # 4-hour candles
ANNUALIZATION_FACTOR = 365 * CANDLES_PER_DAY

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
            .getOrCreate())

# =============================================================================
# STAGE 1: DATA INGESTION
# =============================================================================

def stage1_ingest_data(spark):
    """
    Load all 13 months of data and create month_id column.

    Month mapping:
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

    # Create month_id: Jan 2025 = 1, Feb 2025 = 2, ..., Dec 2025 = 12
    # We only have 2025 data, so month_id = month
    df = df.withColumn("month_id", F.col("month"))

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
    print(f"Data spans {len(month_counts)} months (month_id 1-12)")
    print("\nMonth distribution:")
    for row in month_counts:
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May",
                      "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        month_name = month_names[row.month_id] if row.month_id <= 12 else f"Month {row.month_id}"
        print(f"  Month {row.month_id:2d} ({month_name:10s}): {row['count']:,} rows")

    return long_df, symbols

# =============================================================================
# STAGE 2: MONTHLY WINDOWING LOGIC
# =============================================================================

def stage2_create_month_pairs():
    """
    Create mapping of (backtest_month, lookback_month, holding_month).

    Example:
    - Month 2 (Feb 2025): Use Jan 2025 (month_id=1) for weights,
                          hold portfolio through Feb 2025 (month_id=2)
    - Month 3 (Mar 2025): Use Feb 2025 (month_id=2) for weights,
                          hold portfolio through Mar 2025 (month_id=3)

    Returns:
        List of (backtest_month, lookback_month, holding_month) tuples
    """
    print("\n" + "="*60)
    print("STAGE 2: MONTHLY WINDOWING")
    print("="*60)

    month_pairs = []
    for month in range(BACKTEST_START_MONTH, BACKTEST_END_MONTH + 1):
        lookback = month - LOOKBACK_MONTHS
        holding = month
        month_pairs.append((month, lookback, holding))

    print(f"\nCreated {len(month_pairs)} month pairs:")
    print("  Backtest Month | Lookback Month | Holding Month")
    print("  " + "-"*50)
    for bm, lm, hm in month_pairs[:3]:
        print(f"       {bm:2d}       |       {lm:2d}       |      {hm:2d}")
    if len(month_pairs) > 3:
        print("       ...      |      ...       |     ...")
        bm, lm, hm = month_pairs[-1]
        print(f"       {bm:2d}       |       {lm:2d}       |      {hm:2d}")

    return month_pairs

# =============================================================================
# HELPER: CALCULATE METRICS FOR LOOKBACK PERIOD
# =============================================================================

def calculate_metrics(long_df, month_id):
    """
    Calculate returns, volatility, and Sharpe ratio for assets in a specific month.

    Args:
        long_df: Full dataset
        month_id: Month to analyze

    Returns:
        DataFrame with columns: (symbol, annual_ret, annual_vol, sharpe, count)
    """
    # Filter to specific month
    month_data = long_df.filter(F.col("month_id") == month_id)

    # Calculate log returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = month_data.withColumn("prev_close", F.lag("close").over(window))
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

def calculate_correlation_matrix(long_df, month_id, symbols):
    """
    Calculate Pearson correlation matrix on returns for a specific month.

    Returns:
        corr_mat: NumPy correlation matrix
        valid_cols: List of symbols (column order)
    """
    # Filter to specific month
    month_data = long_df.filter(F.col("month_id") == month_id)

    # Calculate returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = month_data.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()).select("timestamp", "symbol", "ret")

    # Pivot returns matrix
    wide_ret = df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("ret"))
    wide_ret = wide_ret.fillna(0)  # Forward fill (0 return = constant price)

    # VectorAssembler
    valid_cols = [c for c in symbols if c in wide_ret.columns]
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")

    # Correlation
    corr_mat = Correlation.corr(vec_df, "features", "pearson").head()[0].toArray()

    return corr_mat, valid_cols

# =============================================================================
# STAGE 3A: PORTFOLIO A - CORRELATION-BASED STRATEGY
# =============================================================================

def stage3a_portfolio_correlation(long_df, symbols, month_id):
    """
    Correlation-based tournament filtering strategy.

    Algorithm:
    1. Calculate metrics (returns, volatility, Sharpe)
    2. Build correlation matrix on returns
    3. Apply tournament filter: for each highly correlated pair (|ρ| > 0.85),
       drop the asset with lower Sharpe ratio
    4. Select top 20 survivors by Sharpe
    5. Apply inverse-volatility weights

    Returns:
        Pandas DataFrame with (symbol, weight, sharpe, annual_vol)
    """
    # Calculate metrics
    metrics = calculate_metrics(long_df, month_id)

    # Calculate correlation
    corr_mat, valid_symbols = calculate_correlation_matrix(long_df, month_id, symbols)

    # Bring metrics to driver (small data)
    pdf = metrics.toPandas().set_index("symbol")

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

    # Inverse volatility weights
    final_df["inv_vol"] = 1.0 / final_df["annual_vol"]
    final_df["weight"] = final_df["inv_vol"] / final_df["inv_vol"].sum()

    # Return as Pandas DataFrame
    result_df = final_df.reset_index()[["symbol", "weight", "sharpe", "annual_vol"]]

    return result_df

# =============================================================================
# STAGE 3B: PORTFOLIO B - K-MEANS CLUSTERING STRATEGY
# =============================================================================

def stage3b_portfolio_kmeans(long_df, month_id):
    """
    K-Means clustering strategy.

    Algorithm:
    1. Calculate risk/return metrics
    2. Standardize features: [annual_ret, annual_vol, sharpe]
    3. Train K-Means with k=5
    4. Select top Sharpe asset from each cluster
    5. Apply inverse-volatility weights across 5 assets

    Returns:
        DataFrame with (symbol, weight, cluster, sharpe, annual_vol)
    """
    # Calculate metrics
    metrics = calculate_metrics(long_df, month_id)

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

    # Select top asset from each cluster
    window = Window.partitionBy("cluster").orderBy(F.desc("sharpe"))
    cluster_leaders = clustered.withColumn("rank", F.row_number().over(window)) \
                              .filter(F.col("rank") == 1) \
                              .select("symbol", "cluster", "sharpe", "annual_vol")

    # Convert to pandas for weighting
    leaders_pdf = cluster_leaders.toPandas()
    leaders_pdf["inv_vol"] = 1.0 / leaders_pdf["annual_vol"]
    leaders_pdf["weight"] = leaders_pdf["inv_vol"] / leaders_pdf["inv_vol"].sum()

    return leaders_pdf[["symbol", "weight", "cluster", "sharpe", "annual_vol"]]

# =============================================================================
# STAGE 3C: PORTFOLIO C - PCA COMPONENT STRATEGY
# =============================================================================

def stage3c_portfolio_pca(long_df, symbols, month_id):
    """
    PCA component loading strategy.

    Algorithm:
    1. Build returns matrix (timestamp × symbol pivot)
    2. Apply PCA with k=10 components
    3. Extract PC1 loadings (eigenvector coefficients)
    4. Select top 15 assets by absolute loading magnitude
    5. Weight by normalized absolute loading

    Rationale:
    Assets with high PC1 loadings are "market factor representatives" - they
    best explain the primary source of variance in crypto returns.

    Returns:
        DataFrame with (symbol, weight, pc1_loading, sharpe, annual_vol)
    """
    # Filter to specific month
    month_data = long_df.filter(F.col("month_id") == month_id)

    # Calculate returns
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = month_data.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()).select("timestamp", "symbol", "ret")

    # Pivot returns
    wide_ret = df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("ret"))
    wide_ret = wide_ret.fillna(0)

    # VectorAssembler
    valid_cols = [c for c in symbols if c in wide_ret.columns]
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")

    # Apply PCA
    pca = PCA(k=min(N_PCA_COMPONENTS, len(valid_cols)), inputCol="features", outputCol="pca_features")
    model = pca.fit(vec_df)

    # Get PC1 loadings (first row of principal components matrix)
    # pc.shape = (n_components, n_features)
    pc_matrix = model.pc.toArray()
    pc1_loadings = pc_matrix[:, 0]  # First principal component loadings

    # Create DataFrame with loadings
    loading_df = pd.DataFrame({
        'symbol': valid_cols,
        'pc1_loading': pc1_loadings,
        'abs_loading': np.abs(pc1_loadings)
    })

    # Select top N by absolute loading
    loading_df = loading_df.sort_values('abs_loading', ascending=False).head(TOP_PCA_ASSETS)

    # Weight by normalized absolute loading
    loading_df['weight'] = loading_df['abs_loading'] / loading_df['abs_loading'].sum()

    # Get metrics for selected assets
    metrics = calculate_metrics(long_df, month_id)
    metrics_pdf = metrics.toPandas()

    # Merge with metrics
    result_df = loading_df.merge(metrics_pdf[['symbol', 'sharpe', 'annual_vol']],
                                 on='symbol', how='left')

    return result_df[['symbol', 'weight', 'pc1_loading', 'sharpe', 'annual_vol']]

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
# STAGE 4: BACKTEST EXECUTION
# =============================================================================

def stage4_backtest_month(long_df, weights_pdf, holding_month):
    """
    Execute backtest for a single month.

    Args:
        long_df: Full dataset
        weights_pdf: Pandas DataFrame with (symbol, weight)
        holding_month: Month to hold portfolio (month_id)

    Returns:
        month_return: Portfolio return for the month
        num_assets: Number of assets in portfolio
    """
    # Filter holding month data
    month_data = long_df.filter(F.col("month_id") == holding_month)

    # Get first and last close for each asset
    window_asc = Window.partitionBy("symbol").orderBy("timestamp")
    window_desc = Window.partitionBy("symbol").orderBy(F.desc("timestamp"))

    first_close = month_data.withColumn("rn", F.row_number().over(window_asc)) \
                           .filter(F.col("rn") == 1) \
                           .select("symbol", F.col("close").alias("close_start"))

    last_close = month_data.withColumn("rn", F.row_number().over(window_desc)) \
                          .filter(F.col("rn") == 1) \
                          .select("symbol", F.col("close").alias("close_end"))

    # Join to get returns
    asset_returns = first_close.join(last_close, "symbol", "inner")
    asset_returns = asset_returns.withColumn(
        "return",
        (F.col("close_end") - F.col("close_start")) / F.col("close_start")
    )

    # Convert to pandas for weighting calculation
    returns_pdf = asset_returns.select("symbol", "return").toPandas()

    # Merge with weights
    portfolio = weights_pdf.merge(returns_pdf, on="symbol", how="inner")

    # Handle missing assets (renormalize weights)
    if len(portfolio) < len(weights_pdf):
        portfolio['weight'] = portfolio['weight'] / portfolio['weight'].sum()

    # Calculate portfolio return
    portfolio_return = (portfolio['weight'] * portfolio['return']).sum()

    return portfolio_return, len(portfolio)

def stage4_run_backtest(long_df, symbols, month_pairs):
    """
    Run full backtest across all months and all strategies.

    Returns:
        results_df: Pandas DataFrame with monthly results
    """
    print("\n" + "="*60)
    print("STAGE 4: BACKTEST EXECUTION")
    print("="*60)

    results = []

    for backtest_month, lookback_month, holding_month in month_pairs:
        month_name = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][backtest_month]
        print(f"\nProcessing Month {backtest_month}/12 ({month_name} 2025)...")
        print(f"  Lookback: month_id={lookback_month}, Holding: month_id={holding_month}")

        # Generate portfolios using lookback month
        print("  Generating portfolios...")

        # Strategy A: Correlation-based
        try:
            weights_a = stage3a_portfolio_correlation(long_df, symbols, lookback_month)
            ret_a, n_a = stage4_backtest_month(long_df, weights_a, holding_month)
            print(f"    Portfolio A: {n_a} assets, return: {ret_a:+.4f}")
        except Exception as e:
            print(f"    Portfolio A FAILED: {e}")
            ret_a, n_a = 0.0, 0

        # Strategy B: K-Means
        try:
            weights_b = stage3b_portfolio_kmeans(long_df, lookback_month)
            ret_b, n_b = stage4_backtest_month(long_df, weights_b, holding_month)
            print(f"    Portfolio B: {n_b} assets, return: {ret_b:+.4f}")
        except Exception as e:
            print(f"    Portfolio B FAILED: {e}")
            ret_b, n_b = 0.0, 0

        # Strategy C: PCA
        try:
            weights_c = stage3c_portfolio_pca(long_df, symbols, lookback_month)
            ret_c, n_c = stage4_backtest_month(long_df, weights_c, holding_month)
            print(f"    Portfolio C: {n_c} assets, return: {ret_c:+.4f}")
        except Exception as e:
            print(f"    Portfolio C FAILED: {e}")
            ret_c, n_c = 0.0, 0

        # Strategy D: BTC Buy & Hold (no rebalancing needed)
        try:
            weights_d = stage3d_portfolio_btc()
            ret_d, n_d = stage4_backtest_month(long_df, weights_d, holding_month)
            print(f"    Portfolio D (BTC): {n_d} asset, return: {ret_d:+.4f}")
        except Exception as e:
            print(f"    Portfolio D FAILED: {e}")
            ret_d, n_d = 0.0, 0

        # Store results
        results.append({
            'month': backtest_month,
            'strategy': 'A',
            'month_return': ret_a,
            'num_assets': n_a
        })
        results.append({
            'month': backtest_month,
            'strategy': 'B',
            'month_return': ret_b,
            'num_assets': n_b
        })
        results.append({
            'month': backtest_month,
            'strategy': 'C',
            'month_return': ret_c,
            'num_assets': n_c
        })
        results.append({
            'month': backtest_month,
            'strategy': 'D',
            'month_return': ret_d,
            'num_assets': n_d
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

    for strategy in ['A', 'B', 'C', 'D']:
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
        ann_return = np.mean(returns) * 12
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
    1. Equity curves (all 3 strategies)
    2. Monthly returns (grouped bar chart)
    3. Drawdown analysis (underwater chart)
    4. Performance summary table
    """
    print("\n" + "="*60)
    print("STAGE 6: VISUALIZATION")
    print("="*60)

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Define colors for strategies
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728'}
    strategy_names = {
        'A': 'Correlation-based (Top 20)',
        'B': 'K-Means Clustering (5 clusters)',
        'C': 'PCA Components (Top 15)',
        'D': 'BTC Buy & Hold (Benchmark)'
    }

    # Plot 1: Equity Curves
    ax1 = fig.add_subplot(gs[0, :])

    for strategy in ['A', 'B', 'C', 'D']:
        strategy_equity = equity_df[equity_df['strategy'] == strategy].sort_values('month')
        months = [0] + strategy_equity['month'].tolist()
        values = [INITIAL_CAPITAL] + strategy_equity['portfolio_value'].tolist()
        ax1.plot(months, values, marker='o', linewidth=2, label=strategy_names[strategy],
                color=colors[strategy])

    ax1.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_xlabel('Month (0=Dec 2024, 1=Jan 2025, ..., 12=Dec 2025)', fontsize=10)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=10)
    ax1.set_title('Equity Curves - Monthly Rebalancing Backtest', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 12)

    # Plot 2: Monthly Returns
    ax2 = fig.add_subplot(gs[1, 0])

    months = sorted(results_df['month'].unique())
    width = 0.2
    x = np.arange(len(months))

    for i, strategy in enumerate(['A', 'B', 'C', 'D']):
        strategy_returns = results_df[results_df['strategy'] == strategy].sort_values('month')
        returns = strategy_returns['month_return'].values * 100
        ax2.bar(x + i*width, returns, width, label=strategy_names[strategy], color=colors[strategy])

    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('Return (%)', fontsize=10)
    ax2.set_title('Monthly Returns Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels(months)
    ax2.legend(loc='best', fontsize=7)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Drawdown Analysis
    ax3 = fig.add_subplot(gs[1, 1])

    for strategy in ['A', 'B', 'C', 'D']:
        strategy_equity = equity_df[equity_df['strategy'] == strategy].sort_values('month')
        values = np.array([INITIAL_CAPITAL] + strategy_equity['portfolio_value'].tolist())
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        months_plot = list(range(len(values)))
        ax3.fill_between(months_plot, drawdown, 0, alpha=0.3, color=colors[strategy])
        ax3.plot(months_plot, drawdown, linewidth=2, label=strategy_names[strategy],
                color=colors[strategy])

    ax3.set_xlabel('Month', fontsize=10)
    ax3.set_ylabel('Drawdown (%)', fontsize=10)
    ax3.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 12)

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
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows
    for i in range(len(table_data)):
        strategy = ['A', 'B', 'C'][i]
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    # Save figure
    output_path = os.path.join(output_folder, 'backtest_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
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
        print(f"Backtest Period: Feb 2025 - Dec 2025 (11 months)")
        print(f"Lookback Window: {LOOKBACK_MONTHS} month(s)")
        print(f"Strategies: A (Correlation), B (K-Means), C (PCA), D (BTC Benchmark)")

        # Stage 1: Ingest data
        long_df, symbols = stage1_ingest_data(spark)

        # Stage 2: Create month pairs
        month_pairs = stage2_create_month_pairs()

        # Stage 3 & 4: Run backtest
        results_df = stage4_run_backtest(long_df, symbols, month_pairs)

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
        'A': 'Correlation-based (Top 20)',
        'B': 'K-Means Clustering (5 clusters)',
        'C': 'PCA Components (Top 15)',
        'D': 'BTC Buy & Hold (Benchmark)'
    }
    main()
