import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import load_raw_data, make_candles
import numpy as np
import sys

def calculate_volume_profile_from_trades(trades_df, A, B, symbol=None, bins=80):
    print(f"\nCalculating volume profile from {len(trades_df)} trades...")
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

def calculate_poc(vp, return_volume=False):
    """Return the price of the POC (and optionally the volume)."""
    if vp is None or len(vp["volumes"]) == 0:
        raise ValueError("Invalid or empty volume profile data")

    idx_max = np.argmax(vp["volumes"])
    poc_price = vp["price_bins"][idx_max]
    poc_volume = vp["volumes"][idx_max]

    return (poc_price, poc_volume) if return_volume else poc_price

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

def decide_entry_direction(b_candle_close, poc, current_candle):
    if not (current_candle.low <= poc <= current_candle.high):
        return None

    # Decide direction based on B candle close
    if b_candle_close > poc:
        return "long"
    elif b_candle_close < poc:
        return "short"
    
    return None

                    
def detect_entries(candles, volume_profiles):
    """
    Detects when candles touch a POC level from prior volume profiles.

    Parameters
    ----------
    candles : list
        List of Candle objects with attributes: time, open, high, low, close.
    volume_profiles : list of dict
        Each dict must contain: {"A": datetime, "B": datetime, "poc_price": float}

    Returns
    -------
    entries : list of dict
        [{"poc": float, "B": Timestamp, "entry_time": Timestamp, "entry_price": float, "candle_index": int}, ...]
    """

    entries = []
    times = [pd.Timestamp(c.time).tz_localize(None) for c in candles]

    for vp in volume_profiles:
        poc = vp["poc"]
        B_time = pd.Timestamp(vp["B"]).tz_localize(None)

        # Find the candle that matches B
        b_candle = next((c for c in candles if pd.Timestamp(c.time).tz_localize(None) == B_time), None)
        if b_candle is None:
            print(f"⚠️ Could not find B candle at {B_time}")
            continue

        b_close = b_candle.close

        for i in range(len(candles)):
            c = candles[i]
            if pd.Timestamp(c.time).tz_localize(None) <= B_time:
                continue  # only check after B

            entry_type = decide_entry_direction(b_close, poc, c)

            if entry_type:
                entries.append({"poc": poc, "B": B, "entry_time": times[i], "entry_price": c.close, "candle_index": i, "entry_type": entry_type})
                break  # stop after first touch

    return entries

def result_from_entries(entries, candles, stop_losses, rrratios):
    results = []
    for entry in entries:
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        

def visualize_candles(candles, t="Candlestick Chart", moving_averages=None, cross_up=None, cross_down=None, volume_profiles=None, entries=None):
    times = [pd.Timestamp(c.time).tz_localize(None) for c in candles]
    opens, highs, lows, closes, volumes = zip(*[(c.open, c.high, c.low, c.close, c.volume) for c in candles])
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(candles))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=times, open=opens, high=highs, low=lows, close=closes, name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=times, y=volumes, name='Volume', marker_color=colors), row=2, col=1)
    
    if moving_averages:
        for period, ma_values in moving_averages.items():
            fig.add_trace(go.Scatter(x=times, y=ma_values, mode='lines', line=dict(width=1.5), name=f"MA{period}"), row=1, col=1)

 # === Crossovers ===
    if cross_up or cross_down:
        indices = (cross_up or []) + (cross_down or [])
        fig.add_trace(go.Scatter(x=[times[i] for i in indices], y=[closes[i] for i in indices], mode="markers", name="Crossovers", marker=dict(symbol="circle", color="yellow", size=10, line=dict(width=1, color="black")), ), row=1, col=1)

    if entries:
        long_entries = [e for e in entries if e["entry_type"] == "long"]
        short_entries = [e for e in entries if e["entry_type"] == "short"]

        # Longs (green upward triangles)
        if long_entries:
            fig.add_trace(go.Scatter(x=[e["entry_time"] for e in long_entries], y=[e["poc"] for e in long_entries], mode="markers", name="Long Entry", marker=dict(symbol="triangle-up", color="lime", size=12, line=dict(width=1, color="black")), ), row=1, col=1)

        # Shorts (red downward triangles)
        if short_entries:
            fig.add_trace(go.Scatter( x=[e["entry_time"] for e in short_entries], y=[e["poc"] for e in short_entries], mode="markers", name="Short Entry", marker=dict(symbol="triangle-down", color="red", size=12, line=dict(width=1, color="black")), ), row=1, col=1)

        # --- POC lines ---
    if volume_profiles:
        for vp in volume_profiles:
            A = pd.Timestamp(vp["A"]).tz_localize(None)
            B = pd.Timestamp(vp["B"]).tz_localize(None)
            poc = vp["poc"]

            fig.add_shape(type="line", x0=A, x1=B, y0=poc, y1=poc, line=dict(color="orange", width=2, dash="dot"), row=1, col=1,)
            # Optional annotation:
            fig.add_annotation(x=B, y=poc, text=f"POC {poc:.5f}", showarrow=False, font=dict(size=10, color="orange"), xanchor="left", yanchor="bottom", row=1, col=1)

    fig.update_layout(title=t, xaxis_rangeslider_visible=False, height=800, hovermode='x unified', xaxis=dict(type='date'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.show()


if __name__ == "__main__":
    # Load last 60 days of data (faster rendering, still plenty of data)
    # Change max_files to load more/less data
    date = int(sys.argv[1])
    dataset = load_raw_data(date, max_files=None)

    symbols = ["6EH4", "6EM4", "6EU4", "6EZ4","6EH5", "6EM5", "6EU5", "6EZ5"]
    short_ma = 20
    long_ma = 50
    freq="25min"

    for s in symbols:
        candles = make_candles(dataset, freq, symbol=s)
        
        if not candles:
            print(f"No candles for {s}")
            continue


        ma_dict = calculate_moving_averages(candles, periods=(short_ma, long_ma))
        cross_up, cross_down = detect_ma_crossovers(ma_dict[short_ma], ma_dict[long_ma])

         # === Build Volume Profiles between crossovers ===
        volume_profiles = []
        all_crosses = sorted([(i, "up") for i in cross_up] + [(i, "down") for i in cross_down], key=lambda x: x[0])

        for i in range(len(all_crosses) - 1):
            idx_a, type_a = all_crosses[i]
            idx_b, type_b = all_crosses[i + 1]

            A = pd.Timestamp(candles[idx_a].time).tz_localize(None)
            B = pd.Timestamp(candles[idx_b].time).tz_localize(None)

            vp = calculate_volume_profile_from_trades(dataset, A, B, symbol=s, bins=160)
            if vp is None:
                continue

            poc = calculate_poc(vp)
            volume_profiles.append({"A": A, "B": B, "poc": poc})

        entries = detect_entries(candles, volume_profiles)

        stop_losses = [20 for i range(len(entries))]
        rrratios = [2 for i range(len(entries))]
        results = result_from_entries(entries, candles, stop_losses, rrratios)

        visualize_candles(candles, t=f"{s} {freq} Candlestick Chart", moving_averages=ma_dict, cross_up=cross_up, cross_down=cross_down, volume_profiles=volume_profiles, entries=entries)



'''
Todo

Resolver situacao dos simbolos OK
calcular Moving Average Crossings OK
Calcular Volume Profile OK
Plotar POC OK
Add entry signals OK
Fix decide_entry_direction OK
Fix price step (futures should be 0.0005) OK
'''