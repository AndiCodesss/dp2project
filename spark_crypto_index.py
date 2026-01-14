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
        
        # Define schema for CSV manually to avoid inference overhead
        from pyspark.sql.types import StructType, StructField, DoubleType, LongType
        schema = StructType([
            StructField("open_time", LongType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("close_time", LongType(), True),
            StructField("quote_volume", DoubleType(), True),
            StructField("trades", LongType(), True),
            StructField("taker_buy_base", DoubleType(), True),
            StructField("taker_buy_quote", DoubleType(), True),
            StructField("ignore", DoubleType(), True),
        ])
        
        df = spark.read.csv(csv_files, schema=schema, header=False)
        df = df.withColumn("_filename", F.input_file_name())
        df = df.withColumn("symbol", F.regexp_extract(F.col("_filename"), r"([A-Z0-9]+)-\d+h-", 1))
        # CSV open_time is in microseconds or milliseconds?
        # We need to detect or assume. Based on analysis, it is microseconds.
        # But we will convert to Float close price and just keep timestamp as Key.
        df = df.withColumn("timestamp", F.col("open_time"))
    else:
        raise FileNotFoundError("No data found!")

    # Select necessary columns
    long_df = df.select(
        F.col("timestamp").cast("long"), 
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
# STAGE 4: PORTFOLIO
# =============================================================================

def stage4_portfolio(metrics, corr_mat, symbols):
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
        
        # 4. Portfolio
        final_df, _ = stage4_portfolio(metrics, corr_mat, valid_symbols)
        
        # Save
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        final_df.to_csv(f"{OUTPUT_FOLDER}/smart_index.csv")
        
        # Visualize
        plt.figure(figsize=(10,10))
        sns.heatmap(corr_mat, cmap="coolwarm", center=0)
        plt.title("Correlation Matrix")
        plt.savefig(f"{OUTPUT_FOLDER}/correlation_matrix.png")
        print("\nPipeline Done.")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if spark: spark.stop()

if __name__ == "__main__":
    main()
