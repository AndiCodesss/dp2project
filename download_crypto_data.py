"""
Crypto Data Downloader & Converter (Windows Compatible)
=======================================================
Downloads OHLCV data from Binance Public Data and converts to Parquet format.

Features:
- Multi-year, multi-coin support (scalable to 2000+ pairs)
- Parallel downloads with error handling
- robust CSV → Parquet conversion using Pandas/PyArrow (Windows safe)
- Automatic CSV cleanup to save space

Usage:
    python download_crypto_data.py              # Download + convert + delete CSVs
    python download_crypto_data.py --skip-convert  # Download only
    python download_crypto_data.py --convert-only  # Only convert existing CSVs
"""

import requests
import os
import zipfile
import io
import argparse
import shutil
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Tuple, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# Top 100 cryptocurrencies by market cap (USDT trading pairs)
TOP_PAIRS = [
    # Top 10
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    # 11-20
    "DOTUSDT", "MATICUSDT", "UNIUSDT", "LTCUSDT", "ATOMUSDT",
    "ETCUSDT", "XLMUSDT", "FILUSDT", "NEARUSDT", "APTUSDT",
    # 21-30
    "ARBUSDT", "OPUSDT", "VETUSDT", "ALGOUSDT", "FTMUSDT",
    "SANDUSDT", "MANAUSDT", "AXSUSDT", "AAVEUSDT", "EGLDUSDT",
    # 31-40
    "EOSUSDT", "XTZUSDT", "THETAUSDT", "ICPUSDT", "GRTUSDT",
    "FLOWUSDT", "NEOUSDT", "MKRUSDT", "SNXUSDT", "KAVAUSDT",
    # 41-50
    "RNDRUSDT", "INJUSDT", "SUIUSDT", "SEIUSDT", "TIAUSDT",
    "IMXUSDT", "LDOUSDT", "RUNEUSDT", "CFXUSDT", "MINAUSDT",
    # 51-60
    "APEUSDT", "GMXUSDT", "FETUSDT", "AGIXUSDT", "OCEANUSDT",
    "WOOUSDT", "CRVUSDT", "COMPUSDT", "LRCUSDT", "ENJUSDT",
    # 61-70
    "CHZUSDT", "GALAUSDT", "DYDXUSDT", "ZECUSDT", "DASHUSDT",
    "WAVESUSDT", "BATUSDT", "ZILUSDT", "IOSTUSDT", "ONTUSDT",
    # 71-80
    "HOTUSDT", "RVNUSDT", "ZENUSDT", "SCUSDT", "ICXUSDT",
    "ZRXUSDT", "SXPUSDT", "KSMUSDT", "CELRUSDT", "ONEUSDT",
    # 81-90
    "QTUMUSDT", "ANKRUSDT", "SKLUSDT", "COTIUSDT", "BAKEUSDT",
    "IOTAUSDT", "CTSIUSDT", "BANDUSDT", "STMXUSDT", "OGNUSDT",
    # 91-100
    "NKNUSDT", "DENTUSDT", "MTLUSDT", "REEFUSDT", "DGBUSDT",
    "1INCHUSDT", "SUSHIUSDT", "YFIUSDT", "AUDIOUSDT", "CELOUSDT",
]

# Time period configuration
START_YEAR = 2025
END_YEAR = 2025
START_MONTH = 12
END_MONTH = 12
INTERVAL = "4h"  # 4-hour candles

# Base URL for Binance Public Data
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

# Output folders
CSV_FOLDER = "crypto_data_4h"
PARQUET_FOLDER = "crypto_data_parquet"

# Download settings
MAX_WORKERS = 10

# Binance columns
COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def add_header_to_csv(csv_path: str, columns: List[str]):
    """Add header row to a CSV file if it doesn't have one."""
    try:
        # Read the file
        with open(csv_path, 'r') as f:
            content = f.read()
        
        # Check if first line looks like header (contains letters)
        first_line = content.split('\n')[0]
        if any(c.isalpha() for c in first_line):
            return  # Already has header
        
        # Add header
        header = ','.join(columns) + '\n'
        with open(csv_path, 'w') as f:
            f.write(header + content)
    except Exception as e:
        pass  # Silently fail to not interrupt download process

def generate_year_months(start_year: int, start_month: int, 
                          end_year: int, end_month: int) -> List[Tuple[str, str]]:
    periods = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        for month in range(m_start, m_end + 1):
            periods.append((str(year), f"{month:02d}"))
    return periods

def download_file(pair: str, year: str, month: str, interval: str, 
                  save_folder: str) -> Tuple[str, str, str, bool, str]:
    file_name = f"{pair}-{interval}-{year}-{month}.zip"
    url = f"{BASE_URL}/{pair}/{interval}/{file_name}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                z.extractall(save_folder)
            
            # Add header to extracted CSV
            csv_name = file_name.replace('.zip', '.csv')
            csv_path = os.path.join(save_folder, csv_name)
            if os.path.exists(csv_path):
                add_header_to_csv(csv_path, COLUMNS)
            
            return (pair, year, month, True, "OK")
        else:
            return (pair, year, month, False, f"HTTP {response.status_code}")
    except Exception as e:
        return (pair, year, month, False, str(e)[:50])

def download_all_data(pairs: List[str], periods: List[Tuple[str, str]], interval: str, 
                      save_folder: str, max_workers: int = 10) -> Dict[str, int]:
    os.makedirs(save_folder, exist_ok=True)
    tasks = [(pair, year, month) for pair in pairs for (year, month) in periods]
    total_tasks = len(tasks)
    
    print(f"\n{'='*60}\nDOWNLOADING DATA\n{'='*60}")
    print(f"Total files: {total_tasks}")
    
    success_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_file, pair, year, month, interval, save_folder): 
            (pair, year, month) for (pair, year, month) in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            pair, year, month, success, message = future.result()
            if success:
                success_count += 1
            else:
                failed_files.append((pair, year, month, message))
            
            if i % 20 == 0 or i == total_tasks:
                print(f"Progress: {i}/{total_tasks} - {success_count} OK")

    print(f"\nDownload finished. Success: {success_count}, Failed: {len(failed_files)}")
    return {"success": success_count, "failed": len(failed_files)}

# =============================================================================
# PARQUET CONVERSION (PANDAS/PYARROW)
# =============================================================================

def convert_single_csv(csv_file, parquet_folder):
    """
    Read a single CSV, add partition columns, save to partitioned parquet structure.
    Returns (success, symbol)
    """
    try:
        # Extract metadata from filename: BTCUSDT-4h-2025-12.csv
        filename = os.path.basename(csv_file)
        parts = filename.split("-")
        if len(parts) < 4:
            return False, filename, "Invalid filename format"
            
        symbol = parts[0]
        
        # Check if CSV has header by reading first line
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            has_header = any(c.isalpha() for c in first_line)
        
        # Read CSV with appropriate parameters
        if has_header:
            df = pd.read_csv(csv_file, header=0)  # Use first row as header
        else:
            df = pd.read_csv(csv_file, names=COLUMNS, header=None)  # No header
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Convert timestamp to datetime (Input is in microseconds for 4h data?)
        # Verified: 1764547200000000 (us) -> 2025/12.
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="us")
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        
        # Select columns
        keep_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "year", "month"]
        df = df[keep_cols]
        
        # Write to partitioned Parquet
        # Partition schema: parquet_folder/year=2025/month=12/part-xyz.parquet
        df.to_parquet(
            parquet_folder,
            partition_cols=["year", "month"],
            engine="pyarrow",
            compression="snappy",
            index=False
        )
        return True, symbol, "OK"
    except Exception as e:
        return False, filename, str(e)

def convert_to_parquet_pandas(csv_folder, parquet_folder, delete_csv=True):
    """
    Convert all CSV files to Parquet using Pandas (reliable on Windows).
    Refactored to process file-by-file to avoid huge memory usage.
    """
    print(f"\n{'='*60}\nCONVERTING TO PARQUET (PANDAS)\n{'='*60}")
    
    csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    print(f"Found {len(csv_files)} CSV files to convert.")
    
    # Clean output folder if exists to start fresh
    if os.path.exists(parquet_folder):
        print(f"Cleaning existing parquet folder: {parquet_folder}")
        shutil.rmtree(parquet_folder)
    
    success_count = 0
    failed_count = 0
    
    # Process sequentially or in parallel (Pandas is fast enough sequentially for 100 files)
    # Using ThreadPool for speed
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(convert_single_csv, f, parquet_folder): f for f in csv_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            f = futures[future]
            success, name, msg = future.result()
            
            if success:
                success_count += 1
                # Delete CSV immediately if successful to save space
                if delete_csv:
                    try:
                        os.remove(f)
                    except:
                        pass 
            else:
                failed_count += 1
                print(f"Failed {name}: {msg}")
                
            if i % 20 == 0:
                print(f"Converted {i}/{len(csv_files)}")
    
    print(f"\nConversion Complete.")
    print(f"✅ Converted: {success_count}")
    print(f"❌ Failed: {failed_count}")
    
    if delete_csv and failed_count == 0:
        print("Cleaning up empty CSV folder...")
        try:
            shutil.rmtree(csv_folder)
        except:
            pass

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--convert-only", action="store_true")
    parser.add_argument("--add-headers", action="store_true", help="Add headers to existing CSV files")
    args = parser.parse_args()
    
    # Add headers to existing CSVs if requested
    if args.add_headers:
        print(f"\n{'='*60}\nADDING HEADERS TO EXISTING CSVs\n{'='*60}")
        csv_files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
        for csv_file in csv_files:
            add_header_to_csv(csv_file, COLUMNS)
        print(f"✅ Added headers to {len(csv_files)} files.")
        return
    
    # 1. Download
    if not args.convert_only:
        periods = generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
        download_all_data(TOP_PAIRS, periods, INTERVAL, CSV_FOLDER, MAX_WORKERS)
    
    # 2. Convert & Cleanup
    if not args.skip_convert:
        convert_to_parquet_pandas(CSV_FOLDER, PARQUET_FOLDER, delete_csv=False)
    
    print("\n✅ Process Complete!")

if __name__ == "__main__":
    main()
