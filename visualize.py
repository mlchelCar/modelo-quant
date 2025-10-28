import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import load_raw_data, make_candles
import numpy as np


def calculate_volume_profile_from_trades(trades_df, A, B, symbol=None, bins=80):
    print(f"Calculating volume profile from {len(trades_df)} trades...")
    print(f"  A: {A}, B: {B}, symbol: {symbol}, bins: {bins}")

    df = trades_df.copy()

    # Ensure consistent naming (depends on your dataset)
    if "ts_event" in df.columns:
        df["time"] = pd.to_datetime(df["ts_event"])
    elif "time" not in df.columns:
        raise ValueError("No valid time column found in trades_df")

    # Remove timezone info (make tz-naive)
    df["time"] = df["time"].dt.tz_localize(None)

    # Convert A and B also to tz-naive Timestamps
    A = pd.Timestamp(A).tz_localize(None)
    B = pd.Timestamp(B).tz_localize(None)

    # Filter by time range and optional symbol
    mask = (df["time"] >= A) & (df["time"] <= B)
    if symbol:
        mask &= (df["symbol"] == symbol)
    df = df.loc[mask]

    if df.empty:
        print("⚠️ No trades found in this range.")
        return None

    # Use price and size (volume)
    prices = df["price"].to_numpy()
    volumes = df["size"].to_numpy()

    # Compute histogram (volume per price bin)
    vol, bin_edges = np.histogram(prices, bins=bins, weights=volumes)

    # Midpoints for plotting
    price_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {"price_bins": price_bins,"volumes": vol, "A": A, "B": B, "symbol": symbol,}

def calculate_moving_averages(candles, periods=(25, 100), ema=True):
    """Return a dict of moving averages (EMA or SMA) keyed by period."""
    df = pd.DataFrame({
        "close": [c.close for c in candles],
    })
    ma_dict = {}
    for p in periods:
        if ema:
            ma_dict[p] = df["close"].ewm(span=p, adjust=False).mean()  # EMA
        else:
            ma_dict[p] = df["close"].rolling(window=p).mean()          # SMA
    return ma_dict

def detect_ma_crossovers(ma_short, ma_long):
    """Return indices (or times) where short MA crosses long MA."""
    cross_up = []   # short crosses above long (buy)
    cross_down = [] # short crosses below long (sell)

    for i in range(1, len(ma_short)):
        if pd.isna(ma_short[i-1]) or pd.isna(ma_long[i-1]):
            continue

        # Upward crossover
        if ma_short[i-1] <= ma_long[i-1] and ma_short[i] > ma_long[i]:
            cross_up.append(i)
        # Downward crossover
        elif ma_short[i-1] >= ma_long[i-1] and ma_short[i] < ma_long[i]:
            cross_down.append(i)

    return cross_up, cross_down

def visualize_candles(candles, t="Candlestick Chart", moving_averages=None, cross_up=None, cross_down=None):
    times = [pd.Timestamp(c.time).tz_localize(None) for c in candles]
    opens, highs, lows, closes, volumes = zip(*[(c.open, c.high, c.low, c.close, c.volume) for c in candles])
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(candles))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=times, open=opens, high=highs, low=lows, close=closes, name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=times, y=volumes, name='Volume', marker_color=colors), row=2, col=1)
    
    if moving_averages:
        for period, ma_values in moving_averages.items():
            fig.add_trace(go.Scatter(
                x=times,
                y=ma_values,
                mode='lines',
                line=dict(width=1.5),
                name=f"MA{period}"
            ), row=1, col=1)

 # === Crossovers ===
    if cross_up:
        fig.add_trace(go.Scatter(
            x=[times[i] for i in cross_up],
            y=[closes[i] for i in cross_up],
            mode="markers",
            name="Bullish Crossover",
            marker=dict(symbol="triangle-up", color="lime", size=12, line=dict(width=1, color="black")),
        ), row=1, col=1)

    if cross_down:
        fig.add_trace(go.Scatter(
            x=[times[i] for i in cross_down],
            y=[closes[i] for i in cross_down],
            mode="markers",
            name="Bearish Crossover",
            marker=dict(symbol="triangle-down", color="red", size=12, line=dict(width=1, color="black")),
        ), row=1, col=1)

    fig.update_layout(title=t, xaxis_rangeslider_visible=False, height=800, hovermode='x unified', xaxis=dict(type='date'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.show()


if __name__ == "__main__":
    # Load last 60 days of data (faster rendering, still plenty of data)
    # Change max_files to load more/less data

    dataset = load_raw_data(20251001, max_files=None)

    symbols = ["6EH4", "6EM4", "6EU4", "6EZ4","6EH5", "6EM5", "6EU5", "6EZ5"]
    A = "2025-10-19 09:00:00"
    B = "2025-10-22 12:00:00"

    vp = calculate_volume_profile_from_trades(dataset, A, B, symbol="6EZ5", bins=80)

    freq="25min"
    for s in symbols:
        candles = make_candles(dataset, freq, symbol=s)
        
        if not candles:
            print(f"No candles for {s}")
            continue

        short_ma = 20
        long_ma = 50
        ma_dict = calculate_moving_averages(candles, periods=(short_ma, long_ma))
        cross_up, cross_down = detect_ma_crossovers(ma_dict[short_ma], ma_dict[long_ma])

        visualize_candles(candles, t=f"{s} {freq} Candlestick Chart", moving_averages=ma_dict, cross_up=cross_up, cross_down=cross_down)



'''
Todo

Resolver situacao dos simbolos OK
calcular Moving Average Crossings OK
Calcular Volume Profile OK
Plotar Volume Profile
'''