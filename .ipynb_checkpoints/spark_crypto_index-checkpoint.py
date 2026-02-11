"""
Crypto Index Construction Engine (Scalable Version)
====================================================
A fully distributed PySpark analytics engine to construct a monthly rebalanced
"Smart Index" of cryptocurrencies.

Features:
- Scales to 2000+ trading pairs
- Fully distributed Spark pipeline (no Pandas in hot path)
- Robust error handling for Windows/Local execution

Pipeline Stages:
1. Ingestion & Temporal Alignment (Spark SQL)
2. Feature Engineering (Spark Window Functions)
3. Correlation Pruning (Spark MLlib)
4. Portfolio Construction (Inverse Volatility Weighting)
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data source
DATA_SOURCE = "crypto_data_parquet"  # Partitioned Parquet
CSV_FALLBACK = "crypto_data_4h"      # CSV Folder

OUTPUT_FOLDER = "output"

# Index parameters
CORRELATION_THRESHOLD = 0.85
TOP_N_ASSETS = 20
CANDLES_PER_DAY = 6
ANNUALIZATION_FACTOR = np.sqrt(365 * CANDLES_PER_DAY)

# =============================================================================
# UTILS
# =============================================================================

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
import findspark

# Initialize findspark if available (standard practice for local dev, harmless on server)
try:
    findspark.init()
except:
    pass

def create_spark_session() -> SparkSession:
    """Initialize Spark session."""
    return (SparkSession.builder
            .appName("CryptoIndexEngine")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.shuffle.partitions", "20") # Adjust based on data size
            .getOrCreate())

# =============================================================================
# STAGE 1: INGESTION
# =============================================================================

def stage1_ingest_data(spark: SparkSession):
    print("\n--- Stage 1: Ingestion ---")
    
    # Try Parquet first
    if os.path.exists(DATA_SOURCE) and os.listdir(DATA_SOURCE):
        print(f"Reading Parquet from: {DATA_SOURCE}")
        df = spark.read.parquet(DATA_SOURCE)
    elif os.path.exists(CSV_FALLBACK):
        print(f"Reading CSV from: {CSV_FALLBACK}")
        csv_files = glob.glob(os.path.join(CSV_FALLBACK, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSVs in {CSV_FALLBACK}")
        
        # Define schema for CSV (new datetime-based format)
        from pyspark.sql.types import StructType, StructField, DoubleType, LongType, StringType
        schema = StructType([
            StructField("datetime", StringType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("quote_volume", DoubleType(), True),
            StructField("trades", LongType(), True),
            StructField("taker_buy_base", DoubleType(), True),
            StructField("taker_buy_quote", DoubleType(), True),
            StructField("ignore", DoubleType(), True),
        ])
        
        df = spark.read.csv(csv_files, schema=schema, header=True)
        df = df.withColumn("_filename", F.input_file_name())
        df = df.withColumn("symbol", F.regexp_extract(F.col("_filename"), r"([A-Z0-9]+)-\d+h-", 1))
        # Convert datetime string to timestamp
        df = df.withColumn("timestamp", F.to_timestamp(F.col("datetime"), "yyyy-MM-dd HH:mm:ss"))
    else:
        raise FileNotFoundError("No data found!")

    # Select necessary columns - timestamp is proper datetime type from parquet
    long_df = df.select(
        F.col("timestamp"), 
        F.col("symbol"), 
        F.col("close").cast("double")
    ).filter(F.col("close").isNotNull())
    
    # Cache and count
    long_df = long_df.cache()
    count = long_df.count()
    symbols = [r.symbol for r in long_df.select("symbol").distinct().collect()]
    
    print(f"Loaded {count:,} rows across {len(symbols)} assets.")
    return long_df, symbols

def stage1_pivot(long_df, symbols):
    print("Pivoting to wide format...")
    # Manual pivot or Spark pivot? Spark pivot is fine for 100 cols.
    # For 2000, we might prefer GroupBy.
    master_matrix = long_df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("close"))
    master_matrix = master_matrix.orderBy("timestamp").cache()
    print(f"Master Matrix: {master_matrix.count()} rows")
    return master_matrix

# =============================================================================
# STAGE 2: METRICS
# =============================================================================

def stage2_metrics(long_df):
    print("\n--- Stage 2: Metrics (Distributed) ---")
    
    window = Window.partitionBy("symbol").orderBy("timestamp")
    
    # Calculate log returns
    df = long_df.withColumn("prev_close", F.lag("close").over(window))
    df = df.withColumn("log_ret", F.log(F.col("close") / F.col("prev_close")))
    df = df.filter(F.col("log_ret").isNotNull())
    
    # Aggregations
    # Annualized Factor
    ann_factor = 365 * 6 # periods per year
    
    metrics = df.groupBy("symbol").agg(
        F.count("log_ret").alias("count"),
        F.sum("log_ret").alias("sum_log_ret"),
        F.stddev("log_ret").alias("std_log_ret")
    )
    
    metrics = metrics.withColumn("annual_ret", F.col("sum_log_ret") * (ann_factor / F.col("count")))
    metrics = metrics.withColumn("annual_vol", F.col("std_log_ret") * np.sqrt(ann_factor))
    metrics = metrics.withColumn("sharpe", F.col("annual_ret") / F.col("annual_vol"))
    
    metrics = metrics.filter(F.col("count") > 50) # Minimum data points
    
    print("Metrics calculated. Top 5:")
    metrics.orderBy(F.desc("sharpe")).show(5)
    return metrics

# =============================================================================
# STAGE 3: CORRELATION
# =============================================================================

# =============================================================================
# STAGE 3: CORRELATION
# =============================================================================

def stage3_correlation(long_df, symbols, spark):
    """Correlation on Log Returns with Forward Imputation."""
    print("\n--- Stage 3: Correlation on Returns ---")
    
    # Calculate returns first (same as stage 2)
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = long_df.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()).select("timestamp", "symbol", "ret")
    
    # Pivot Returns
    # Matrix: Rows = Time, Cols = Symbols
    wide_ret = df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("ret"))
    
    # FORWARD IMPUTATION:
    # Filling missing returns with 0 implies the price remained constant (carried forward).
    # P_t = P_{t-1} => log(P_t / P_{t-1}) = log(1) = 0
    wide_ret = wide_ret.fillna(0)
    
    # VectorAssembler
    valid_cols = [c for c in symbols if c in wide_ret.columns]
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")
    
    # Correlation
    print("Calculating Correlation Matrix (Pearson)...")
    corr_mat = Correlation.corr(vec_df, "features", "pearson").head()[0].toArray()
    
    return corr_mat, valid_cols

# =============================================================================
# STAGE 3B: PCA ANALYSIS (ML Model #2)
# =============================================================================

def stage3b_pca_analysis(long_df, symbols, n_components=10):
    """
    Apply PCA to returns matrix to find independent market factors.
    Alternative to correlation filtering.
    """
    print(f"\n--- Stage 3B: PCA Analysis (n_components={n_components}) ---")
    
    from pyspark.ml.feature import PCA
    
    # Prepare returns matrix (same as correlation)
    window = Window.partitionBy("symbol").orderBy("timestamp")
    df = long_df.withColumn("prev", F.lag("close").over(window))
    df = df.withColumn("ret", F.log(F.col("close") / F.col("prev")))
    df = df.filter(F.col("ret").isNotNull()).select("timestamp", "symbol", "ret")
    
    # Pivot Returns
    wide_ret = df.groupBy("timestamp").pivot("symbol", symbols).agg(F.first("ret"))
    wide_ret = wide_ret.fillna(0)
    
    # VectorAssembler
    valid_cols = [c for c in symbols if c in wide_ret.columns]
    assembler = VectorAssembler(inputCols=valid_cols, outputCol="features")
    vec_df = assembler.transform(wide_ret).select("features")
    
    # Apply PCA
    pca = PCA(k=min(n_components, len(valid_cols)), inputCol="features", outputCol="pca_features")
    model = pca.fit(vec_df)
    
    # Get explained variance
    explained_variance = model.explainedVariance.toArray()
    cumsum_variance = np.cumsum(explained_variance)
    
    print(f"\nExplained Variance by Component:")
    for i, (var, cum_var) in enumerate(zip(explained_variance[:5], cumsum_variance[:5])):
        print(f"  PC{i+1}: {var*100:.2f}% (Cumulative: {cum_var*100:.2f}%)")
    
    print(f"\nTotal variance explained by {n_components} components: {cumsum_variance[-1]*100:.2f}%")
    
    # Transform data
    transformed = model.transform(vec_df)
    
    return model, explained_variance, valid_cols


# =============================================================================
# STAGE 4: K-MEANS CLUSTERING
# =============================================================================

def stage4_kmeans_clustering(metrics, n_clusters=5):
    """
    Cluster cryptocurrencies by risk/return profiles.
    Select top performer from each cluster for diversification.
    """
    print(f"\n--- Stage 4: K-Means Clustering (k={n_clusters}) ---")
    
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.evaluation import ClusteringEvaluator
    
    # Prepare features: annual_ret, annual_vol, sharpe
    features = ["annual_ret", "annual_vol", "sharpe"]
    
    # Assemble features
    assembler = VectorAssembler(inputCols=features, outputCol="features_raw")
    df_features = assembler.transform(metrics)
    
    # Standardize (critical for K-Means - features have different scales)
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df_features)
    df_scaled = scaler_model.transform(df_features)
    
    # Train K-Means
    kmeans = KMeans(k=n_clusters, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df_scaled)
    
    # Make predictions
    clustered = model.transform(df_scaled)
    
    # Evaluate clustering quality (Silhouette score)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
    silhouette = evaluator.evaluate(clustered)
    print(f"Silhouette Score: {silhouette:.3f} (higher is better, range: -1 to 1)")
    
    # Show cluster statistics
    cluster_stats = clustered.groupBy("cluster").agg(
        F.count("symbol").alias("count"),
        F.avg("sharpe").alias("avg_sharpe"),
        F.avg("annual_vol").alias("avg_vol"),
        F.avg("annual_ret").alias("avg_ret")
    ).orderBy("cluster")
    
    print("\nCluster Statistics:")
    cluster_stats.show()
    
    # Select top asset from each cluster (by Sharpe ratio)
    from pyspark.sql import Window
    window = Window.partitionBy("cluster").orderBy(F.desc("sharpe"))
    
    cluster_leaders = clustered.withColumn("rank", F.row_number().over(window)) \
                              .filter(F.col("rank") == 1) \
                              .select("symbol", "cluster", "sharpe", "annual_ret", "annual_vol")
    
    print(f"\nSelected {cluster_leaders.count()} cluster representatives:")
    cluster_leaders.show()
    
    return clustered, cluster_leaders, model

# =============================================================================
# STAGE 4B: FEATURE ENGINEERING FOR GBT
# =============================================================================

def engineer_features_for_gbt(long_df):
    """Create technical features from OHLCV data for GBT prediction."""
    print("\n--- Feature Engineering for GBT ---")
    
    window = Window.partitionBy("symbol").orderBy("timestamp")
    
    # Lagged returns
    df = long_df.withColumn("prev_close", F.lag("close").over(window))
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
    print(f"Engineered {len(feature_cols)} features: {feature_cols}")
    
    return df, feature_cols

# =============================================================================
# STAGE 4C: GBT PREDICTION (ML Model #3)
# =============================================================================

def stage4c_gbt_prediction(long_df):
    """
    Train Gradient Boosted Trees to predict future returns.
    Use predictions for forward-looking portfolio weights.
    """
    print("\n--- Stage 4C: Gradient Boosted Trees Regression ---")
    
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.feature import VectorAssembler
    
    # Engineer features
    df, feature_cols = engineer_features_for_gbt(long_df)
    
    # Assemble features
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = assembler.transform(df).select("symbol", "features", "target")
    
    # Train/Test split (80/20)
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    
    print(f"Training set: {train.count()} rows")
    print(f"Test set: {test.count()} rows")
    
    # Train GBT
    gbt = GBTRegressor(featuresCol="features", labelCol="target",
                       maxIter=20, maxDepth=5, seed=42)
    model = gbt.fit(train)
    
    # Evaluate
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(labelCol="target", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    print(f"\nModel Performance:")
    print(f"  RMSE: {rmse:.6f}")
    
    # Feature importance
    importance = model.featureImportances.toArray()
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(importance_df.to_string(index=False))
    
    # Predict on all data for portfolio construction
    full_predictions = model.transform(df.select("symbol", "features"))
    
    return model, rmse, importance_df, full_predictions

# =============================================================================
# STAGE 5: PORTFOLIO (CORRELATION-BASED)
# =============================================================================

def stage5_portfolio(metrics, corr_mat, symbols):
    print("\n--- Stage 4: Tournament Filter ---")
    
    # Bring metrics to driver for filtering (small data: 2000 rows)
    pdf = metrics.toPandas().set_index("symbol")
    
    # Symbols list matching the matrix
    idx_to_sym = {i: s for i, s in enumerate(symbols)}
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    
    keep = set(symbols)
    dropped = set()
    
    # Iterate upper triangle
    cnt = 0
    import numpy as np
    
    rows, cols = corr_mat.shape
    for i in range(rows):
        for j in range(i+1, cols):
            if abs(corr_mat[i, j]) > CORRELATION_THRESHOLD:
                s1 = idx_to_sym[i]
                s2 = idx_to_sym[j]
                
                if s1 in dropped or s2 in dropped:
                    continue
                
                # Compare Sharpe
                try:
                    sh1 = pdf.loc[s1, "sharpe"]
                    sh2 = pdf.loc[s2, "sharpe"]
                except KeyError:
                    continue
                
                cnt += 1
                if sh1 < sh2:
                    dropped.add(s1)
                    if s1 in keep: keep.remove(s1)
                else:
                    dropped.add(s2)
                    if s2 in keep: keep.remove(s2)
    
    print(f"Evaluated pairs. Dropped {len(dropped)} high-correlation assets.")
    
    # Final Weights (Inv Vol)
    final_df = pdf.loc[list(keep)].copy()
    final_df = final_df.sort_values("sharpe", ascending=False).head(TOP_N_ASSETS)
    
    final_df["inv_vol"] = 1.0 / final_df["annual_vol"]
    final_df["weight"] = final_df["inv_vol"] / final_df["inv_vol"].sum()
    
    print("\nTop 10 Final Portfolio:")
    print(final_df[["weight", "sharpe", "annual_vol"]].head(10))
    
    return final_df, corr_mat

# =============================================================================
# MAIN
# =============================================================================

def main():
    spark = None
    try:
        spark = create_spark_session()
        spark.sparkContext.setLogLevel("ERROR")
        
        # 1. Ingest
        long_df, symbols = stage1_ingest_data(spark)
        
        # 2. Metrics
        metrics = stage2_metrics(long_df)
        
        # 3. Correlation (On Returns!)
        corr_mat, valid_symbols = stage3_correlation(long_df, symbols, spark)
        
        # 3B. PCA Analysis (ML Model #2)
        pca_model, explained_var, pca_symbols = stage3b_pca_analysis(long_df, symbols, n_components=10)
        
        # 4. K-Means Clustering (ML Model #1)
        clustered, cluster_leaders, kmeans_model = stage4_kmeans_clustering(metrics, n_clusters=5)
        
        # 4C. GBT Prediction (ML Model #3)
        gbt_model, gbt_rmse, feature_importance, gbt_predictions = stage4c_gbt_prediction(long_df)
        
        # 5. Portfolio (Correlation-based baseline)
        final_df, _ = stage5_portfolio(metrics, corr_mat, valid_symbols)
        
        # Save
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        final_df.to_csv(f"{OUTPUT_FOLDER}/smart_index.csv")
        
        # Save K-Means results
        clustered.select("symbol", "cluster", "sharpe", "annual_ret", "annual_vol") \
                .toPandas().to_csv(f"{OUTPUT_FOLDER}/kmeans_clusters.csv", index=False)
        cluster_leaders.toPandas().to_csv(f"{OUTPUT_FOLDER}/kmeans_portfolio.csv", index=False)
        
        # Visualize
        plt.figure(figsize=(10,10))
        sns.heatmap(corr_mat, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix")
        plt.savefig(f"{OUTPUT_FOLDER}/correlation_matrix.png")
        
        # K-Means scatter plot
        cluster_pdf = clustered.select("annual_ret", "annual_vol", "sharpe", "cluster").toPandas()
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cluster_pdf["annual_vol"], cluster_pdf["sharpe"], 
                             c=cluster_pdf["cluster"], cmap="viridis", s=100, alpha=0.6)
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Sharpe Ratio")
        plt.title("K-Means Clustering: Risk vs Return Profile")
        plt.colorbar(scatter, label="Cluster")
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_FOLDER}/kmeans_clusters_viz.png")
        
        # PCA Scree Plot
        plt.figure(figsize=(10, 6))
        cumsum_var = np.cumsum(explained_var)
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_var)+1), explained_var)
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance")
        plt.title("PCA: Explained Variance by Component")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(cumsum_var)+1), cumsum_var * 100, marker='o')
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance (%)")
        plt.title("PCA: Cumulative Explained Variance")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=90, color='r', linestyle='--', label='90% threshold')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/pca_scree_plot.png")
        
        # Save PCA results
        pca_results = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(explained_var))],
            'explained_variance': explained_var,
            'cumulative_variance': cumsum_var
        })
        pca_results.to_csv(f"{OUTPUT_FOLDER}/pca_components.csv", index=False)
        
        # GBT Feature Importance Plot
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"GBT Feature Importance (RMSE: {gbt_rmse:.6f})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_FOLDER}/gbt_feature_importance.png")
        
        # Save GBT results
        feature_importance.to_csv(f"{OUTPUT_FOLDER}/gbt_feature_importance.csv", index=False)
        
        print("\n✅ Pipeline Done.")
        print(f"Outputs saved to {OUTPUT_FOLDER}/:")
        print("  - smart_index.csv (correlation-based portfolio)")
        print("  - kmeans_clusters.csv (cluster assignments)")
        print("  - kmeans_portfolio.csv (cluster-based portfolio)")
        print("  - kmeans_clusters_viz.png (cluster visualization)")
        print("  - pca_components.csv (principal components & variance)")
        print("  - pca_scree_plot.png (PCA variance visualization)")
        print("  - gbt_feature_importance.csv (GBT model feature importance)")
        print("  - gbt_feature_importance.png (feature importance visualization)")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
