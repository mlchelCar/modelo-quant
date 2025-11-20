import databento as db
import pandas as pd
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path

class Candle:
    __slots__ = ('time', 'open', 'high', 'low', 'close', 'volume', 'symbol')

    def __init__(self, time, open_, high, low, close, volume, symbol):
        self.time = time
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.symbol = symbol
   
def separate_data(all_candles, number_days, first_date, last_date, fit_data_size):
    candles = all_candles[:]
    print(f"Separating data for {number_days} days...")

    first_date = pd.to_datetime(str(first_date), format="%Y%m%d").tz_localize("UTC")
    last_date = pd.to_datetime(str(last_date), format="%Y%m%d").tz_localize("UTC")
    days = pd.date_range(start=first_date, end=last_date, freq="D")
    
    # saturdays = pd.date_range(start=first_date, end=last_date, freq='W-SAT', tz='UTC')
    saturdays = [d for d in days if d.weekday() == 5]  # weekday(): Monday=0, Saturday=5, Sunday=6

    results = []

    p = 0
    for s in saturdays:
        weeks = []
        for c in candles[p:]:
            if c.time >= s:
                break
            weeks.append(c)
            p += 1
        results.append(weeks)
    results.append(candles[p:])

    v = int(len(results)*fit_data_size)

    fit_data = results[:v]
    test_data = results[v:]

    
    return  [cd for wk in fit_data for cd in wk], [cd for wk in test_data for cd in wk]

def load_raw_data(initial_date, final_date, path="./data", max_files=None):
    """
    Load all .dbn files within the date range [initial_date, final_date].

    Filenames must contain the date in positions [10:18], e.g.:
        something_YYYYMMDD.dbn
    """

    # --- Convert datetime to int YYYYMMDD --- #
    if not isinstance(initial_date, int):
        initial_date = int(initial_date.strftime("%Y%m%d"))
    if not isinstance(final_date, int):
        final_date = int(final_date.strftime("%Y%m%d"))

    print(f"Loading data from {path} with {initial_date} <= file_date <= {final_date}...")

    last_date = None
    data_path = Path(path)

    if not data_path.exists():
        raise ValueError(f"Path {path} does not exist")

    dfs = []
    count = 0

    for file in sorted(data_path.glob("*.dbn")):
        try:
            file_date = int(file.name[10:18])
        except ValueError:
            print(f"Skipping file (invalid date in name): {file.name}")
            continue

        # Check date range
        if initial_date <= file_date <= final_date:
            print(f"  [{count+1}] {file.name}")
            dfs.append(db.DBNStore.from_file(str(file)).to_df())
            last_date = file_date
            count += 1

            if max_files and count >= max_files:
                break

    if not dfs:
        raise ValueError(f"No .dbn files found between {initial_date} and {final_date} in {path}")

    print(f"Combining {len(dfs)} files...")
    result = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(result):,} rows")

    return result, count, initial_date, last_date

def permutate_candles():
    raise NotImplementedError

def make_candles(dataset, freq="25min", symbol=None, roll_schedule=None):
    print(f"\nMaking candles with freq={freq}...")
    df = dataset[["ts_event", "price", "size", "symbol"]].copy()
    df["datetime"] = pd.to_datetime(df["ts_event"], utc=True)  # ✅ ensure UTC

    # Auto-select most common symbol if none provided
    if symbol is None:
        symbol = df["symbol"].value_counts().idxmax()

    print(f"  Symbol: {symbol}")
    df = df[df["symbol"] == symbol]

    # 🔹 Filter by roll_schedule if provided
    if roll_schedule and symbol in roll_schedule:
        start_date, end_date = roll_schedule[symbol]
        # ✅ make both tz-aware (UTC)
        start_date = pd.to_datetime(start_date).tz_localize("UTC")
        end_date = pd.to_datetime(end_date).tz_localize("UTC")

        df = df[(df["datetime"] >= start_date) & (df["datetime"] < end_date)]
        print(f"  Date range for {symbol}: {start_date.date()} → {end_date.date()}")
    else:
        print(f"  No roll schedule found for {symbol}")

    # Safety check for empty data after filtering
    if df.empty:
        print(f"  ⚠️ No data for {symbol} in this range.")
        return []

    # 🔹 Resample into candles
    candles_df = (
        df.groupby(pd.Grouper(key="datetime", freq=freq))
        .agg(Open=("price", "first"),
             High=("price", "max"),
             Low=("price", "min"),
             Close=("price", "last"),
             Volume=("size", "sum"))
        .dropna()
        .reset_index()
    )

    print(f"  Created {len(candles_df)} candles for {symbol}")
    return [
        Candle(row['datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'], symbol)
        for _, row in candles_df.iterrows()
    ]

