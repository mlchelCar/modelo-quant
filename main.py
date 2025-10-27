import databento as db
import pandas as pd
from pathlib import Path
import re


class Candle:
    __slots__ = ('time', 'open', 'high', 'low', 'close', 'volume')

    def __init__(self, time, open_, high, low, close, volume):
        self.time = time
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


def load_raw_data(date, path="./data", max_files=None):
    print(f"Loading data from {path} with date >= {date}...")
    data_path = Path(path)
    if not data_path.exists():
        raise ValueError(f"Path {path} does not exist")

    dfs = []
    count = 0
    for file in sorted(data_path.glob("*.dbn")):
        match = re.search(r'(\d{8})', file.name)
        if match and int(match.group(1)) >= date:
            print(f"  [{count+1}] {file.name}")
            dfs.append(db.DBNStore.from_file(str(file)).to_df())
            count += 1
            if max_files and count >= max_files:
                break

    if not dfs:
        raise ValueError(f"No .dbn files found with date >= {date} in {path}")

    print(f"Combining {len(dfs)} files...")
    result = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(result):,} rows")
    return result


def make_candles(dataset, freq="25min", symbol=None):
    print(f"\nMaking candles with freq={freq}...")
    df = dataset[["ts_event", "price", "size", "symbol"]].copy()
    df["datetime"] = pd.to_datetime(df["ts_event"])

    if symbol is None:
        symbol = df["symbol"].value_counts().idxmax()

    print(f"  Symbol: {symbol}")
    df = df[df["symbol"] == symbol]



    candles_df = (
        df.groupby(pd.Grouper(key="datetime", freq=freq))
        .agg(Open=("price", "first"), High=("price", "max"), Low=("price", "min"),
             Close=("price", "last"), Volume=("size", "sum"))
        .dropna()
        .reset_index()
    )

    print(f"  Created {len(candles_df)} candles")
    return [Candle(row['datetime'], row['Open'], row['High'], row['Low'], row['Close'], row['Volume'])
            for _, row in candles_df.iterrows()]


if __name__ == "__main__":
    dataset = load_raw_data(20250723)
    candles = make_candles(dataset, freq="25min")
    print(f"Created {len(candles)} candles")