"""
Crypto Data Downloader & Converter (Windows Compatible)
=======================================================
Downloads OHLCV data from Binance Public Data and converts to Parquet format.

Features:
- Multi-year, multi-coin support (scalable to 2000+ pairs)
- Automatic Binance pair discovery (USDT spot universe)
- Parallel downloads with error handling
- robust CSV → Parquet conversion using Pandas/PyArrow (Windows safe)
- Automatic CSV cleanup to save space

Usage:
    python download_crypto_data.py              # Download + convert + delete CSVs
    python download_crypto_data.py --pair-mode diversified --target-pairs 300
    python download_crypto_data.py --pair-mode all-usdt
    python download_crypto_data.py --pair-mode historical-usdt
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
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any, Set

# =============================================================================
# CONFIGURATION
# =============================================================================

# Fallback pair list (used only if live Binance pair discovery fails)
FALLBACK_TOP_PAIRS = [
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

# Curated seeds to enforce cross-sector diversification.
# Symbols not listed on Binance are ignored automatically.
DIVERSIFIED_SEED_PAIRS = {
    "large_cap": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT", "ADAUSDT",
        "TRXUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"
    ],
    "meme": [
        "DOGEUSDT", "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BONKUSDT",
        "WIFUSDT", "MEMEUSDT", "BOMEUSDT", "1000SATSUSDT", "TURBOUSDT"
    ],
    "stable": [
        "USDCUSDT", "FDUSDUSDT", "USDPUSDT", "TUSDUSDT", "DAIUSDT",
        "USDEUSDT", "AEURUSDT"
    ],
    "commodity": [
        "PAXGUSDT", "XAUTUSDT"
    ],
    "infra_exchange": [
        "BNBUSDT", "ATOMUSDT", "NEARUSDT", "DOTUSDT", "INJUSDT",
        "TIAUSDT", "SEIUSDT", "SUIUSDT", "APTUSDT"
    ],
    "defi_lending_dex": [
        "UNIUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "CRVUSDT",
        "COMPUSDT", "SNXUSDT", "RUNEUSDT", "DYDXUSDT"
    ],
    "ai_data": [
        "FETUSDT", "AGIXUSDT", "OCEANUSDT", "TAOUSDT", "RNDRUSDT"
    ],
    "gaming_nft": [
        "IMXUSDT", "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT",
        "APEUSDT", "ENJUSDT"
    ],
}

# Time period configuration
START_YEAR = 2024
END_YEAR = 2025
START_MONTH = 10  # October 2024 warmup for Jan 2025 (supports 90-day lookback)
END_MONTH = 12    # December 2025
INTERVAL = "4h"   # 4-hour candles

# Base URL for Binance Public Data
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
S3_BUCKET_LIST_URL = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"
TICKER_24HR_URL = "https://api.binance.com/api/v3/ticker/24hr"

# Pair universe defaults
PAIR_MODE_DEFAULT = "diversified"  # options: top100, diversified, all-usdt, historical-usdt
DEFAULT_TARGET_PAIRS = 300
DEFAULT_QUOTE_ASSET = "USDT"

# Output folders
CSV_FOLDER = "crypto_data_4h"
PARQUET_FOLDER = "crypto_data_parquet"

# Download settings
MAX_WORKERS = 10

# Binance columns - simplified with human-readable datetime
COLUMNS = [
    "datetime", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore"
]

# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def convert_binance_timestamp_to_datetime(timestamp: int) -> pd.Timestamp:
    """
    Convert Binance open_time timestamp to pandas datetime.
    Supports seconds, milliseconds, and microseconds.
    """
    if timestamp >= 10**15:
        return pd.to_datetime(timestamp, unit='us')
    if timestamp >= 10**12:
        return pd.to_datetime(timestamp, unit='ms')
    return pd.to_datetime(timestamp, unit='s')


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """
    Parse a datetime-like series from Binance CSV into pandas datetime.
    Handles both numeric timestamps (s/ms/us) and preformatted datetime strings.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    numeric_ratio = numeric.notna().mean()

    if numeric_ratio > 0.95:
        sample = numeric.dropna()
        if sample.empty:
            return pd.to_datetime(series, errors="coerce")

        median_ts = sample.median()
        if median_ts >= 10**15:
            return pd.to_datetime(numeric, unit="us", errors="coerce")
        if median_ts >= 10**12:
            return pd.to_datetime(numeric, unit="ms", errors="coerce")
        return pd.to_datetime(numeric, unit="s", errors="coerce")

    return pd.to_datetime(series, errors="coerce")


def add_header_to_csv(csv_file, columns):
    """Convert raw timestamp CSV to datetime format with headers."""
    try:
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Check if already processed (has datetime in header)
        if lines and 'datetime' in lines[0]:
            return

        # Convert timestamps and write atomically to avoid partial-file corruption
        tmp_file = f"{csv_file}.tmp"
        converted_rows = 0
        with open(tmp_file, 'w', newline='') as f:
            f.write(','.join(columns) + '\n')
            
            for line in lines:
                if line.strip() and not any(c.isalpha() for c in line.split(',')[0]):
                    parts = line.strip().split(',')
                    if len(parts) >= 12:  # Original Binance format
                        timestamp = int(parts[0])
                        dt = convert_binance_timestamp_to_datetime(timestamp)
                        datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Reconstruct line: datetime, open, high, low, close, volume, quote_volume, ...
                        # Skip open_time (parts[0]) and close_time (parts[6])
                        new_line = f"{datetime_str},{parts[1]},{parts[2]},{parts[3]},{parts[4]},{parts[5]},{parts[7]},{parts[8]},{parts[9]},{parts[10]},{parts[11]}\n"
                        f.write(new_line)
                        converted_rows += 1

        if converted_rows == 0:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
            raise ValueError("No rows converted from source CSV")

        os.replace(tmp_file, csv_file)
    except Exception as e:
        tmp_file = f"{csv_file}.tmp"
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except Exception:
                pass
        print(f"Error processing {csv_file}: {e}")

def generate_year_months(start_year: int, start_month: int, 
                          end_year: int, end_month: int) -> List[Tuple[str, str]]:
    periods = []
    for year in range(start_year, end_year + 1):
        m_start = start_month if year == start_year else 1
        m_end = end_month if year == end_year else 12
        for month in range(m_start, m_end + 1):
            periods.append((str(year), f"{month:02d}"))
    return periods

def is_likely_leveraged_token(symbol: str, quote_asset: str = DEFAULT_QUOTE_ASSET) -> bool:
    """
    Heuristic filter for Binance leveraged token symbols.
    Excludes patterns like BTCUPUSDT, ETHDOWNUSDT, BNBBULLUSDT, BEARUSDT.
    """
    if not symbol.endswith(quote_asset):
        return False

    base = symbol[:-len(quote_asset)]
    leveraged_roots = {"UP", "DOWN", "BULL", "BEAR"}

    if base in leveraged_roots:
        return True

    for suffix in leveraged_roots:
        if base.endswith(suffix):
            stem = base[:-len(suffix)]
            # Avoid false positives such as JUPUSDT (stem='J', suffix='UP').
            if len(stem) >= 2:
                return True
    return False

def list_s3_common_prefixes(prefix: str, delimiter: str = "/", timeout: int = 30) -> List[str]:
    """
    List S3 common prefixes for Binance public data bucket with pagination.
    """
    prefixes: List[str] = []
    marker = ""
    page = 0

    while True:
        params = {"prefix": prefix, "delimiter": delimiter}
        if marker:
            params["marker"] = marker

        response = requests.get(S3_BUCKET_LIST_URL, params=params, timeout=timeout)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        if "}" in root.tag:
            ns = root.tag.split("}")[0].strip("{")
            tag = lambda name: f"{{{ns}}}{name}"  # noqa: E731
        else:
            tag = lambda name: name  # noqa: E731

        for cp in root.findall(tag("CommonPrefixes")):
            prefix_el = cp.find(tag("Prefix"))
            if prefix_el is not None and prefix_el.text:
                prefixes.append(prefix_el.text)

        next_marker_el = root.find(tag("NextMarker"))
        marker = next_marker_el.text if (next_marker_el is not None and next_marker_el.text) else ""
        page += 1

        if not marker:
            break

    return prefixes

def fetch_historical_spot_pairs_from_binance_vision(
    quote_asset: str = DEFAULT_QUOTE_ASSET,
    include_leveraged_tokens: bool = False
) -> List[str]:
    """
    Fetch historical spot symbols from Binance public data bucket, including delisted pairs.
    """
    raw_prefixes = list_s3_common_prefixes("data/spot/monthly/klines/", delimiter="/")
    symbols = [p.rstrip("/").split("/")[-1] for p in raw_prefixes]
    quote_filtered = [s for s in symbols if s.endswith(quote_asset)]

    if include_leveraged_tokens:
        return sorted(set(quote_filtered))

    filtered = [
        s for s in quote_filtered
        if not is_likely_leveraged_token(s, quote_asset=quote_asset)
    ]
    return sorted(set(filtered))

def fetch_active_spot_pairs(quote_asset: str = DEFAULT_QUOTE_ASSET,
                            timeout: int = 20) -> List[str]:
    """
    Fetch active spot pairs from Binance Exchange Info endpoint.
    """
    response = requests.get(EXCHANGE_INFO_URL, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    symbols = data.get("symbols", [])
    active_pairs = [
        s["symbol"] for s in symbols
        if s.get("status") == "TRADING"
        and s.get("isSpotTradingAllowed")
        and s.get("quoteAsset") == quote_asset
    ]
    return sorted(active_pairs)

def fetch_volume_ranked_pairs(quote_asset: str = DEFAULT_QUOTE_ASSET,
                              timeout: int = 20) -> List[str]:
    """
    Fetch 24h quote-volume ranking for spot symbols, highest first.
    """
    response = requests.get(TICKER_24HR_URL, timeout=timeout)
    response.raise_for_status()
    rows = response.json()

    ranked = []
    for row in rows:
        symbol = row.get("symbol", "")
        if not symbol.endswith(quote_asset):
            continue
        try:
            qv = float(row.get("quoteVolume", 0.0))
        except Exception:
            qv = 0.0
        ranked.append((symbol, qv))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in ranked]

def add_unique_pair(pair: str, selected: List[str], selected_set: Set[str], active_set: Set[str]):
    """Append pair once if it exists in active set."""
    if pair in active_set and pair not in selected_set:
        selected.append(pair)
        selected_set.add(pair)

def build_diversified_pair_universe(active_pairs: List[str], target_pairs: int,
                                    quote_asset: str = DEFAULT_QUOTE_ASSET) -> List[str]:
    """
    Build diversified universe:
    1) Ensure cross-sector seeds (large-cap, meme, stable, commodity, etc.)
    2) Fill remaining slots by 24h quote volume.
    """
    target = max(1, target_pairs)
    active_set = set(active_pairs)
    selected: List[str] = []
    selected_set: Set[str] = set()

    print("\nSelecting diversified pair universe:")
    for category, seeds in DIVERSIFIED_SEED_PAIRS.items():
        added_before = len(selected)
        for pair in seeds:
            add_unique_pair(pair, selected, selected_set, active_set)
        added_now = len(selected) - added_before
        print(f"  Seed category '{category}': +{added_now}")

    ranked = fetch_volume_ranked_pairs(quote_asset=quote_asset)
    for pair in ranked:
        if len(selected) >= target:
            break
        add_unique_pair(pair, selected, selected_set, active_set)

    if len(selected) < target:
        # Fill tail alphabetically if some pairs are missing in ticker ranking
        for pair in active_pairs:
            if len(selected) >= target:
                break
            add_unique_pair(pair, selected, selected_set, active_set)

    return selected[:target]

def resolve_pair_universe(pair_mode: str = PAIR_MODE_DEFAULT,
                          target_pairs: int = DEFAULT_TARGET_PAIRS,
                          quote_asset: str = DEFAULT_QUOTE_ASSET,
                          include_leveraged_tokens: bool = False) -> List[str]:
    """
    Resolve symbol universe for download.
    Modes:
    - top100: static fallback list (stable + reproducible)
    - diversified: curated seed mix + high-volume fill to target count
    - all-usdt: all active Binance spot symbols quoted in USDT
    - historical-usdt: all historically listed spot symbols in Binance public data
      (includes delisted pairs still present in data archives)
    """
    quote_asset = quote_asset.upper()
    mode = pair_mode.lower()

    if mode == "top100":
        pairs = [p for p in FALLBACK_TOP_PAIRS if p.endswith(quote_asset)]
        print(f"\nPair mode: top100 (static). Using {len(pairs)} pairs.")
        return pairs

    if mode == "historical-usdt":
        try:
            pairs = fetch_historical_spot_pairs_from_binance_vision(
                quote_asset=quote_asset,
                include_leveraged_tokens=include_leveraged_tokens
            )
            print(
                f"\nPair mode: historical-usdt. Using {len(pairs)} historical {quote_asset} pairs "
                f"(include_leveraged_tokens={include_leveraged_tokens})."
            )
            return pairs
        except Exception as e:
            print(f"\nWARNING: Historical archive symbol discovery failed ({e}).")
            print("Falling back to active Binance spot symbols.")

    try:
        active_pairs = fetch_active_spot_pairs(quote_asset=quote_asset)
        print(f"\nDiscovered {len(active_pairs)} active Binance spot {quote_asset} pairs.")

        if mode == "all-usdt":
            print(f"Pair mode: all-usdt. Using all {len(active_pairs)} pairs.")
            return active_pairs

        if mode == "diversified":
            target = min(max(1, target_pairs), len(active_pairs))
            pairs = build_diversified_pair_universe(
                active_pairs=active_pairs,
                target_pairs=target,
                quote_asset=quote_asset
            )
            print(f"Pair mode: diversified. Using {len(pairs)} pairs (target={target}).")
            return pairs

        raise ValueError(f"Unsupported pair mode: {pair_mode}")
    except Exception as e:
        print(f"\nWARNING: Live Binance pair discovery failed ({e}).")
        print("Falling back to static top-100 list.")
        pairs = [p for p in FALLBACK_TOP_PAIRS if p.endswith(quote_asset)]
        print(f"Using {len(pairs)} fallback pairs.")
        return pairs

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
            df = pd.read_csv(csv_file, header=0)
        else:
            df = pd.read_csv(csv_file, names=COLUMNS, header=None)
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Parse datetime column (supports preformatted strings and raw timestamps)
        df["timestamp"] = parse_datetime_series(df["datetime"])
        df = df[df["timestamp"].notna()]
        df["year"] = df["timestamp"].dt.year
        df["month"] = df["timestamp"].dt.month
        
        # Select columns for parquet
        keep_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "year", "month"]
        df = df[keep_cols]
        
        # Write to partitioned Parquet with millisecond precision (Spark-compatible)
        # Partition schema: parquet_folder/year=2025/month=12/part-xyz.parquet
        df.to_parquet(
            parquet_folder,
            partition_cols=["year", "month"],
            engine="pyarrow",
            compression="snappy",
            index=False,
            coerce_timestamps='ms'  # Force millisecond precision for Spark compatibility
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
    parser.add_argument(
        "--pair-mode",
        choices=["top100", "diversified", "all-usdt", "historical-usdt"],
        default=PAIR_MODE_DEFAULT,
        help=(
            "Universe selection mode: "
            "'top100' (static), 'diversified' (seeded+volume), "
            "'all-usdt' (all active Binance spot USDT pairs), "
            "'historical-usdt' (historical+delisted USDT pairs from Binance data archives)"
        )
    )
    parser.add_argument(
        "--target-pairs",
        type=int,
        default=DEFAULT_TARGET_PAIRS,
        help="Target number of pairs for diversified mode (default: 300)"
    )
    parser.add_argument(
        "--quote-asset",
        type=str,
        default=DEFAULT_QUOTE_ASSET,
        help="Quote asset filter for dynamic modes (default: USDT)"
    )
    parser.add_argument(
        "--include-leveraged-tokens",
        action="store_true",
        help="Include leveraged token pairs (UP/DOWN/BULL/BEAR style) in historical mode"
    )
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
        pairs = resolve_pair_universe(
            pair_mode=args.pair_mode,
            target_pairs=args.target_pairs,
            quote_asset=args.quote_asset,
            include_leveraged_tokens=args.include_leveraged_tokens
        )
        periods = generate_year_months(START_YEAR, START_MONTH, END_YEAR, END_MONTH)
        print(
            f"\nDownload plan: {len(pairs)} pairs × {len(periods)} months "
            f"= {len(pairs) * len(periods)} files"
        )
        download_all_data(pairs, periods, INTERVAL, CSV_FOLDER, MAX_WORKERS)
    
    # 2. Convert & Cleanup
    if not args.skip_convert:
        convert_to_parquet_pandas(CSV_FOLDER, PARQUET_FOLDER, delete_csv=False)
    
    print("\n✅ Process Complete!")

if __name__ == "__main__":
    main()
