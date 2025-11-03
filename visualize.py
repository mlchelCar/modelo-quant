import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import load_raw_data, make_candles, separate_data
import numpy as np
import sys
from pandas import Timestamp
from scipy.stats import bootstrap
from numba import njit
import pandas.api.types as ptypes

@njit
def _volume_profile_numba(prices, volumes, bin_edges):
    n_bins = len(bin_edges) - 1
    vol = np.zeros(n_bins, dtype=np.float64)
    for i in range(len(prices)):
        p = prices[i]
        # np.histogram logic: bins are left-inclusive, right-exclusive (except last)
        idx = np.searchsorted(bin_edges, p, side='right') - 1
        if idx == n_bins:  # include right edge in last bin
            idx = n_bins - 1
        if 0 <= idx < n_bins:
            vol[idx] += volumes[i]
    return vol


def calculate_volume_profile_from_trades(trades_df, A, B, symbol=None, bins=80):
    print(f"\nCalculating volume profile from {len(trades_df)} trades...")
    print(f"  A: {A}, B: {B}, symbol: {symbol}, bins: {bins}")

    df = trades_df

    # === Safe datetime handling ===
    if "ts_event" in df.columns:
        if not ptypes.is_datetime64_any_dtype(df["ts_event"]):
            df["time"] = pd.to_datetime(df["ts_event"])
        else:
            df["time"] = df["ts_event"]
    elif "time" not in df.columns:
        raise ValueError("No valid time column found in trades_df")

    # Make timezone-naive if needed
    if getattr(df["time"].dt, "tz", None) is not None:
        df["time"] = df["time"].dt.tz_localize(None)

    # Convert A/B
    A = pd.Timestamp(A).tz_localize(None)
    B = pd.Timestamp(B).tz_localize(None)

    # Filter early
    mask = (df["time"] >= A) & (df["time"] <= B)
    if symbol is not None and "symbol" in df.columns:
        mask &= (df["symbol"] == symbol)
    df = df.loc[mask, ["price", "size"]]

    if df.empty:
        print("⚠️ No trades found in this range.")
        return None

    prices = df["price"].to_numpy(np.float64)
    volumes = df["size"].to_numpy(np.float64)

    # Compute bins and aggregate
    pmin, pmax = prices.min(), prices.max()
    bin_edges = np.linspace(pmin, pmax, bins + 1)
    vol = _volume_profile_numba(prices, volumes, bin_edges)

    price_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {"price_bins": price_bins, "volumes": vol, "A": A, "B": B, "symbol": symbol}


def calculate_poc(vp, return_volume=False):
    """Return the price of the POC (and optionally the volume)."""
    if vp is None or len(vp["volumes"]) == 0:
        raise ValueError("Invalid or empty volume profile data")

    idx_max = np.argmax(vp["volumes"])
    poc_price = round(vp["price_bins"][idx_max],5)
    poc_price = round(round(poc_price / 0.00005) * 0.00005, 5)
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
                entries.append({ "B": B_time, "entry_time": times[i], "entry_price": poc, "candle_index": i, "entry_type": entry_type})
                break  # stop after first touch

    return entries

def result_from_entries(entries, candles, stop_losses, rrratios,  win_size=125, costs=7, contracts=1, slipage_on_losses=12.5):
    results = []

    for entry, sl_dist, rr in zip(entries, stop_losses, rrratios):
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        direction = entry["entry_type"]

        # Normalize timezone
        if entry_time.tzinfo is not None:
            entry_time = entry_time.tz_convert(None)

        # --- 1️⃣ Determine closing time ---
        if entry_time.hour < 21:
            closing_time = entry_time.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            closing_time = (entry_time + pd.Timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)

        # --- 2️⃣ Define stop/target levels ---
        if direction == "long":
            stop_level = entry_price - sl_dist * 5
            target_level = entry_price + rr * sl_dist * 5
        else:
            stop_level = entry_price + sl_dist * 5
            target_level = entry_price - rr * sl_dist * 5

        result = 0
        print(f"\nEntry at {entry_time}, price={entry_price:.5f}, dir={direction}, " f"stop={stop_level:.5f}, target={target_level:.5f}, closing_time={closing_time}")

        prev_close = entry_price
        trade_closed = False

        for candle in candles:
            candle_time = candle.time
            if candle_time.tzinfo is not None:
                candle_time = candle_time.tz_convert(None)

            if candle_time <= entry_time:
                continue

            high, low, close = candle.high, candle.low, candle.close

            # --- Stop or Target first ---
            if direction == "long":
                if low <= stop_level:
                    result = -1 * win_size - costs * contracts - slipage_on_losses
                    trade_closed = True
                    break
                elif high >= target_level:
                    result = rr * win_size - costs * contracts
                    trade_closed = True
                    break
            else:
                if high >= stop_level:
                    result = -1 * win_size - costs * contracts - slipage_on_losses
                    trade_closed = True
                    break
                elif low <= target_level:
                    result = rr * win_size - costs * contracts
                    trade_closed = True
                    break

            # --- Time-based closure check ---
            if candle_time > closing_time:
                # close at previous candle close
                close = prev_close
                if direction == "long":
                    result = (close - entry_price) / (sl_dist * 5) * win_size - costs * contracts
                else:
                    result = (entry_price - close) / (sl_dist * 5) * win_size - costs * contracts
                print(f"Closing trade at {closing_time} (using prev close {close:.5f}), result={result:.2f}")
                trade_closed = True
                break

            prev_close = close  # keep track for next iteration

        # --- Fallback: still open at the end ---
        if not trade_closed:
            print("Trade remained open — closing at last candle price.")
            close = candles[-1].close
            if direction == "long":
                result = (close - entry_price) / (sl_dist * 5) * win_size - costs * contracts
            else:
                result = (entry_price - close) / (sl_dist * 5) * win_size - costs * contracts

        results.append((entry_time, result))

    return results

def visualize_candles(candles, t="Candlestick Chart", moving_averages=None, cross_up=None, cross_down=None, volume_profiles=None, entries=None, sl=None, rrr=None):
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
            fig.add_trace(go.Scatter(x=[e["entry_time"] for e in long_entries], y=[e["entry_price"] for e in long_entries], mode="markers", name="Long Entry", marker=dict(symbol="triangle-up", color="lime", size=12, line=dict(width=1, color="black")), ), row=1, col=1)
            if sl and rrr:
                fig.add_trace(go.Scatter( x=[e["entry_time"] for e in long_entries], y=[e["entry_price"] - sl*5 for e in long_entries], mode="markers", name="Stop Loss", marker=dict(symbol="x", color="pink", size=12, line=dict(width=1, color="black")), ), row=1, col=1)
                fig.add_trace(go.Scatter( x=[e["entry_time"] for e in long_entries], y=[e["entry_price"] + rrr*sl*5 for e in long_entries], mode="markers", name="Target", marker=dict(symbol="x", color="cyan", size=12, line=dict(width=1, color="black")), ), row=1, col=1)

        # Shorts (red downward triangles)
        if short_entries:
            fig.add_trace(go.Scatter( x=[e["entry_time"] for e in short_entries], y=[e["entry_price"] for e in short_entries], mode="markers", name="Short Entry", marker=dict(symbol="triangle-down", color="red", size=12, line=dict(width=1, color="black")), ), row=1, col=1)
            if sl and rrr:  
                fig.add_trace(go.Scatter( x=[e["entry_time"] for e in short_entries], y=[e["entry_price"] + sl*5 for e in short_entries], mode="markers", name="Stop Loss", marker=dict(symbol="x", color="pink", size=12, line=dict(width=1, color="black")), ), row=1, col=1)
                fig.add_trace(go.Scatter( x=[e["entry_time"] for e in short_entries], y=[e["entry_price"] - rrr*sl*5 for e in short_entries], mode="markers", name="Target", marker=dict(symbol="x", color="cyan", size=12, line=dict(width=1, color="black")), ), row=1, col=1)
    
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

def calculate_sharpe(returns, risk_free_rate=0.0, trading_days=252):
    """
    Calculate daily and annualized Sharpe ratio and 95% bootstrap CI.
    """
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0, 0.0, (0.0, 0.0)

    # Excess returns
    excess = returns - risk_free_rate

    mean_return = np.mean(excess)
    std_return = np.std(excess, ddof=1)

    if std_return == 0:
        sharpe = 0.0
    else:
        sharpe = mean_return / std_return

    annualized_sharpe = sharpe * np.sqrt(trading_days)

    # Bootstrap 95% confidence interval
    try:
        res = bootstrap(
            (excess,),
            np.mean,
            confidence_level=0.95,
            random_state=42,
            n_resamples=5000,
            method="percentile"
        )
        ci_low, ci_high = res.confidence_interval.low, res.confidence_interval.high
        # Convert to Sharpe scale
        ci_low = (ci_low / std_return) * np.sqrt(trading_days)
        ci_high = (ci_high / std_return) * np.sqrt(trading_days)
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    return sharpe, annualized_sharpe, (ci_low, ci_high)


def calculate_winrate(returns):
    """Percentage of positive returns."""
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    wins = np.sum(returns > 0)
    return (wins / len(returns)) * 100

def calculate_max_drawdown(returns):
    """Compute max drawdown from cumulative equity curve."""
    equity = np.cumsum(returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks
    max_drawdown = drawdowns.min()
    return max_drawdown


def calculate_profit_factor(returns):
    """Sum of positive returns / absolute sum of negative returns."""
    returns = np.array(returns, dtype=float)
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    if gross_loss == 0:
        return np.inf
    return gross_profit / gross_loss


def calculate_expectancy(returns):
    """Average profit per trade (mean return)."""
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    return np.mean(returns)


def compute_data(trade_list, last_date):
    """
    Compute all key strategy metrics.
    """
    # trade_list = [(Timestamp('2024-10-24 06:25:00'), -1), (Timestamp('2024-10-28 12:05:00'), -1), (Timestamp('2024-10-29 11:50:00'), -1), (Timestamp('2024-10-29 08:05:00'), -1), (Timestamp('2024-10-29 10:10:00'), -1), (Timestamp('2024-10-29 20:10:00'), -1), (Timestamp('2024-11-06 02:20:00'), -1), (Timestamp('2024-11-01 11:55:00'), -1), (Timestamp('2024-11-01 13:35:00'), -1), (Timestamp('2024-11-03 22:40:00'), -1), (Timestamp('2024-11-06 00:40:00'), -1), (Timestamp('2024-11-05 10:55:00'), -1), (Timestamp('2024-11-06 00:15:00'), -1), (Timestamp('2024-11-08 16:00:00'), -1), (Timestamp('2024-11-08 11:00:00'), -1), (Timestamp('2024-11-08 12:15:00'), -1), (Timestamp('2024-11-13 13:55:00'), -1), (Timestamp('2024-11-15 08:25:00'), -1), (Timestamp('2024-11-15 19:40:00'), -1), (Timestamp('2024-11-15 18:00:00'), -1), (Timestamp('2024-11-15 17:35:00'), 3), (Timestamp('2024-11-18 10:10:00'), -1), (Timestamp('2024-11-19 12:25:00'), -1), (Timestamp('2024-11-19 16:10:00'), -1), (Timestamp('2024-11-29 07:25:00'), 3), (Timestamp('2024-11-25 01:20:00'), -1), (Timestamp('2024-11-26 04:00:00'), 3), (Timestamp('2024-11-26 15:40:00'), -1), (Timestamp('2024-11-27 07:30:00'), -1), (Timestamp('2024-12-02 14:35:00'), -1), (Timestamp('2024-11-29 00:20:00'), -1), (Timestamp('2024-12-01 23:10:00'), 3), (Timestamp('2024-11-29 14:55:00'), -1), (Timestamp('2024-12-01 22:45:00'), -1), (Timestamp('2024-12-05 13:50:00'), -1), (Timestamp('2024-12-04 03:15:00'), 3), (Timestamp('2024-12-04 05:20:00'), 3), (Timestamp('2024-12-04 08:15:00'), -1), (Timestamp('2024-12-04 11:35:00'), 3), (Timestamp('2024-12-11 07:20:00'), -1), (Timestamp('2024-12-09 09:55:00'), 3), (Timestamp('2024-12-09 12:00:00'), -1), (Timestamp('2024-12-12 08:20:00'), -1), (Timestamp('2024-12-12 14:35:00'), -1), (Timestamp('2024-12-13 14:45:00'), 3), (Timestamp('2024-12-16 15:35:00'), 3), (Timestamp('2024-12-16 20:10:00'), -1), (Timestamp('2024-12-17 14:05:00'), 3), (Timestamp('2024-12-18 05:55:00'), 3), (Timestamp('2024-12-18 09:40:00'), 3), (Timestamp('2024-12-20 13:20:00'), -1), (Timestamp('2024-12-27 12:00:00'), 3), (Timestamp('2024-12-26 06:50:00'), -1), (Timestamp('2024-12-26 07:40:00'), 3), (Timestamp('2024-12-26 11:50:00'), -1), (Timestamp('2024-12-26 12:40:00'), -1), (Timestamp('2024-12-30 13:45:00'), -1), (Timestamp('2024-12-26 14:45:00'), -1), (Timestamp('2024-12-30 13:45:00'), -1), (Timestamp('2024-12-30 09:35:00'), -1), (Timestamp('2024-12-30 13:45:00'), -1), (Timestamp('2025-01-24 06:15:00'), -1), (Timestamp('2025-01-06 08:40:00'), -1), (Timestamp('2025-01-03 16:30:00'), -1), (Timestamp('2025-01-08 11:30:00'), 3), (Timestamp('2025-01-20 13:30:00'), -1), (Timestamp('2025-01-10 13:30:00'), -1), (Timestamp('2025-01-14 20:00:00'), -1), (Timestamp('2025-01-14 01:40:00'), -1), (Timestamp('2025-01-16 15:20:00'), -1), (Timestamp('2025-01-16 17:50:00'), -1), (Timestamp('2025-01-17 08:50:00'), 3), (Timestamp('2025-01-17 10:55:00'), 3), (Timestamp('2025-01-17 14:40:00'), -1), (Timestamp('2025-01-17 20:55:00'), -1), (Timestamp('2025-01-20 03:30:00'), -1), (Timestamp('2025-02-03 15:00:00'), 3), (Timestamp('2025-01-21 15:20:00'), -1), (Timestamp('2025-01-31 21:10:00'), 3), (Timestamp('2025-01-23 10:15:00'), 3), (Timestamp('2025-01-29 08:45:00'), 3), (Timestamp('2025-01-27 08:50:00'), -1), (Timestamp('2025-01-27 23:00:00'), -1), (Timestamp('2025-02-14 13:30:00'), 3), (Timestamp('2025-01-30 03:05:00'), 3), (Timestamp('2025-01-30 07:40:00'), 3), (Timestamp('2025-01-30 20:35:00'), -1), (Timestamp('2025-01-31 17:00:00'), -1), (Timestamp('2025-01-31 18:15:00'), -1), (Timestamp('2025-02-05 10:20:00'), -1), (Timestamp('2025-02-06 07:35:00'), -1), (Timestamp('2025-02-07 13:10:00'), -1), (Timestamp('2025-02-07 01:30:00'), -1), (Timestamp('2025-02-07 03:35:00'), 3), (Timestamp('2025-02-07 07:20:00'), -1), (Timestamp('2025-02-07 09:50:00'), -1), (Timestamp('2025-02-07 13:10:00'), 3), (Timestamp('2025-02-12 13:10:00'), 3), (Timestamp('2025-02-28 02:05:00'), 3), (Timestamp('2025-02-20 14:30:00'), 3), (Timestamp('2025-02-21 13:25:00'), -1), (Timestamp('2025-02-24 14:45:00'), 3), (Timestamp('2025-02-24 16:50:00'), -1), (Timestamp('2025-02-27 02:45:00'), -1), (Timestamp('2025-02-26 15:30:00'), -1), (Timestamp('2025-02-26 17:35:00'), 3), (Timestamp('2025-03-04 08:10:00'), -1), (Timestamp('2025-03-03 06:45:00'), -1), (Timestamp('2025-03-10 10:50:00'), -1), (Timestamp('2025-03-10 17:30:00'), 3), (Timestamp('2025-03-11 05:35:00'), 3), (Timestamp('2025-03-12 15:20:00'), -1), (Timestamp('2025-03-12 18:15:00'), -1), (Timestamp('2025-03-17 08:05:00'), -1), (Timestamp('2025-03-17 09:20:00'), -1), (Timestamp('2025-04-03 05:05:00'), -1), (Timestamp('2025-03-24 12:10:00'), -1), (Timestamp('2025-03-31 02:30:00'), 3), (Timestamp('2025-03-25 15:40:00'), 3), (Timestamp('2025-03-27 15:10:00'), -1), (Timestamp('2025-03-27 11:25:00'), -1), (Timestamp('2025-03-28 12:50:00'), -1), (Timestamp('2025-04-01 00:35:00'), 3), (Timestamp('2025-04-01 06:00:00'), -1), (Timestamp('2025-04-01 08:55:00'), 3), (Timestamp('2025-04-02 12:50:00'), -1), (Timestamp('2025-04-04 12:20:00'), -1), (Timestamp('2025-04-09 04:25:00'), -1), (Timestamp('2025-04-08 07:35:00'), -1), (Timestamp('2025-04-08 19:15:00'), 3), (Timestamp('2025-04-09 17:45:00'), -1), (Timestamp('2025-04-10 08:20:00'), -1), (Timestamp('2025-04-14 17:45:00'), -1), (Timestamp('2025-04-22 22:25:00'), 3), (Timestamp('2025-04-17 07:25:00'), -1), (Timestamp('2025-04-17 09:55:00'), -1), (Timestamp('2025-04-17 12:50:00'), 3), (Timestamp('2025-04-22 22:00:00'), -1), (Timestamp('2025-04-22 11:35:00'), 3), (Timestamp('2025-04-24 15:15:00'), -1), (Timestamp('2025-04-25 14:35:00'), 3), (Timestamp('2025-04-25 19:35:00'), -1), (Timestamp('2025-04-28 04:40:00'), -1), (Timestamp('2025-04-28 06:20:00'), -1), (Timestamp('2025-04-28 12:10:00'), -1), (Timestamp('2025-04-30 01:40:00'), 3), (Timestamp('2025-04-29 10:40:00'), -1), (Timestamp('2025-04-29 14:50:00'), 3), (Timestamp('2025-04-30 06:40:00'), 3), (Timestamp('2025-05-02 13:40:00'), -1), (Timestamp('2025-05-05 00:50:00'), -1), (Timestamp('2025-05-05 15:50:00'), -1), (Timestamp('2025-05-06 07:15:00'), -1), (Timestamp('2025-05-07 19:30:00'), -1), (Timestamp('2025-05-21 10:10:00'), -1), (Timestamp('2025-05-11 21:50:00'), -1), (Timestamp('2025-05-19 09:00:00'), -1), (Timestamp('2025-05-15 00:25:00'), -1), (Timestamp('2025-05-15 07:30:00'), 3), (Timestamp('2025-05-16 00:35:00'), 3), (Timestamp('2025-05-16 13:55:00'), -1), (Timestamp('2025-05-19 06:05:00'), -1), (Timestamp('2025-05-22 17:00:00'), -1), (Timestamp('2025-05-28 05:05:00'), -1), (Timestamp('2025-05-27 07:50:00'), 3), (Timestamp('2025-05-30 12:05:00'), -1), (Timestamp('2025-05-30 16:15:00'), 3), (Timestamp('2025-06-04 13:45:00'), -1), (Timestamp('2025-06-06 12:25:00'), 3), (Timestamp('2025-06-06 07:50:00'), -1), (Timestamp('2025-06-09 12:30:00'), -1), (Timestamp('2025-06-09 15:00:00'), -1), (Timestamp('2025-06-10 01:50:00'), -1), (Timestamp('2025-06-10 11:00:00'), -1), (Timestamp('2025-06-10 15:10:00'), -1), (Timestamp('2025-06-11 05:45:00'), 3), (Timestamp('2025-06-11 12:25:00'), 3), (Timestamp('2025-06-13 19:50:00'), 3), (Timestamp('2025-06-15 22:55:00'), 3), (Timestamp('2025-06-16 04:45:00'), -1), (Timestamp('2025-06-17 14:30:00'), -1), (Timestamp('2025-06-23 21:45:00'), -1), (Timestamp('2025-06-17 12:25:00'), -1), (Timestamp('2025-06-23 18:00:00'), -1), (Timestamp('2025-06-19 23:10:00'), -1), (Timestamp('2025-06-23 05:55:00'), 3), (Timestamp('2025-07-30 12:30:00'), -1), (Timestamp('2025-06-25 10:00:00'), -1), (Timestamp('2025-07-15 14:10:00'), -1), (Timestamp('2025-06-27 06:35:00'), -1), (Timestamp('2025-06-27 17:50:00'), -1), (Timestamp('2025-07-03 12:10:00'), 3), (Timestamp('2025-07-03 12:10:00'), -1), (Timestamp('2025-07-07 06:10:00'), -1), (Timestamp('2025-07-08 10:05:00'), -1), (Timestamp('2025-07-22 16:10:00'), 3), (Timestamp('2025-07-10 11:40:00'), -1), (Timestamp('2025-07-22 15:20:00'), -1), (Timestamp('2025-07-14 12:20:00'), -1), (Timestamp('2025-07-14 14:00:00'), -1), (Timestamp('2025-07-14 14:50:00'), 3), (Timestamp('2025-07-15 05:25:00'), -1), (Timestamp('2025-07-15 11:15:00'), -1), (Timestamp('2025-07-15 07:05:00'), -1), (Timestamp('2025-07-15 10:00:00'), -1), (Timestamp('2025-07-15 12:30:00'), 3), (Timestamp('2025-07-15 12:55:00'), -1), (Timestamp('2025-07-16 15:10:00'), -1), (Timestamp('2025-07-17 00:20:00'), -1), (Timestamp('2025-07-18 08:00:00'), -1), (Timestamp('2025-07-28 17:10:00'), -1), (Timestamp('2025-07-21 00:35:00'), -1), (Timestamp('2025-07-21 00:10:00'), -1), (Timestamp('2025-07-21 04:45:00'), -1), (Timestamp('2025-07-28 13:25:00'), -1), (Timestamp('2025-07-28 07:10:00'), -1), (Timestamp('2025-07-24 20:15:00'), -1), (Timestamp('2025-07-27 22:00:00'), 3), (Timestamp('2025-07-28 06:20:00'), -1), (Timestamp('2025-08-05 06:25:00'), 3), (Timestamp('2025-08-07 15:05:00'), -1), (Timestamp('2025-08-08 05:40:00'), 3), (Timestamp('2025-08-08 14:00:00'), -1), (Timestamp('2025-08-08 18:35:00'), -1), (Timestamp('2025-08-11 01:35:00'), -1), (Timestamp('2025-08-11 09:05:00'), -1), (Timestamp('2025-08-12 12:10:00'), -1), (Timestamp('2025-08-21 14:30:00'), -1), (Timestamp('2025-08-22 14:40:00'), -1), (Timestamp('2025-08-18 14:25:00'), -1), (Timestamp('2025-08-22 14:40:00'), -1), (Timestamp('2025-08-19 12:30:00'), -1), (Timestamp('2025-08-22 13:50:00'), -1), (Timestamp('2025-08-20 23:55:00'), -1), (Timestamp('2025-08-22 13:50:00'), -1), (Timestamp('2025-08-25 18:55:00'), -1), (Timestamp('2025-09-01 05:55:00'), -1), (Timestamp('2025-08-26 15:20:00'), -1), (Timestamp('2025-08-28 10:15:00'), -1), (Timestamp('2025-08-29 13:20:00'), -1), (Timestamp('2025-09-02 07:45:00'), -1), (Timestamp('2025-09-02 06:05:00'), -1), (Timestamp('2025-09-04 04:45:00'), 3), (Timestamp('2025-09-05 08:15:00'), -1), (Timestamp('2025-09-09 13:55:00'), -1), (Timestamp('2025-09-11 14:40:00'), 3), (Timestamp('2025-09-12 16:55:00'), -1), (Timestamp('2025-09-15 04:20:00'), 3), (Timestamp('2025-09-19 12:05:00'), 3), (Timestamp('2025-09-24 09:35:00'), -1), (Timestamp('2025-10-06 07:25:00'), -1), (Timestamp('2025-10-01 04:05:00'), -1), (Timestamp('2025-10-01 09:30:00'), -1), (Timestamp('2025-10-01 12:00:00'), -1), (Timestamp('2025-10-01 14:05:00'), -1), (Timestamp('2025-10-02 06:45:00'), 3), (Timestamp('2025-10-02 12:35:00'), -1), (Timestamp('2025-10-03 14:00:00'), 3), (Timestamp('2025-10-05 21:50:00'), -1), (Timestamp('2025-10-06 19:55:00'), -1), (Timestamp('2025-10-17 05:30:00'), -1), (Timestamp('2025-10-09 05:50:00'), -1), (Timestamp('2025-10-15 15:35:00'), 3), (Timestamp('2025-10-13 11:30:00'), -1), (Timestamp('2025-10-14 16:15:00'), -1), (Timestamp('2025-10-21 05:20:00'), -1), (Timestamp('2025-10-22 18:25:00'), -1)]

    # Ensure trade_list is sorted chronologically

    trade_list = sorted(trade_list, key=lambda x: x[0])
    # print(trade_list)
    # print(trade_list[0][0].normalize())
    # quit()

    # Create list of all weekdays between first_date and last_date (exclude Saturdays)
    first_date = trade_list[0][0].normalize()
    last_date = pd.to_datetime(str(last_date), format="mixed", dayfirst=False)
    days = pd.date_range(start=first_date, end=last_date, freq="D")
    
    days = [d for d in days if d.weekday() != 5]  # weekday(): Monday=0, Saturday=5, Sunday=6

    # Initialize dictionary to store daily returns
    daily_returns = {day.date(): 0 for day in days}

    # Group trades by day and sum results
    trade_dict = {}
    for trade in trade_list:
        day = trade[0].date()
        trade_dict[day] = trade_dict.get(day, 0) + trade[1]

    # Assign the sum to daily_returns (if day not in trade_dict → remains 0)
    for day in daily_returns:
        daily_returns[day] = trade_dict.get(day, 0)

    l = [i for i in daily_returns.values()]
    l = [i for i in daily_returns.values()] if isinstance(daily_returns, dict) else daily_returns    

    sharpe, annualized_sharpe, ci = calculate_sharpe(l)
    winrate = calculate_winrate(l)
    max_dd = calculate_max_drawdown(l)
    profit_factor = calculate_profit_factor(l)
    expectancy = calculate_expectancy(l)

    metrics = {
        'Returns': l,
        "Sharpe Ratio": round(sharpe, 3),
        "Annualized Sharpe": round(annualized_sharpe, 3),
        "Sharpe 95% CI": (round(ci[0], 3), round(ci[1], 3)),
        "Win Rate (%)": round(winrate, 2),
        "Max Drawdown": round(max_dd, 3),
        "Profit Factor": round(profit_factor, 3),
        "Expectancy": round(expectancy, 3),
        "Total Trades": len(l)
    }

    return metrics

class Variant():
    def __init__(self, entries, candles, stop_losses, rrratios, last_date, n, c=1):
        self.entries = entries
        self.candles = candles
        self.stop_losses = stop_losses
        self.rrratios = rrratios
        
        self.name = n
        self.metrics =  compute_data(result_from_entries(entries, candles, stop_losses, rrratios, contracts=c), last_date)

def run_strategy(dataset, all_candles, data_splits, freq):
    print(f"Running strategy on {len(data_splits)} splits.")

    short_ma = 40
    long_ma = 50

    used_symbols = []
    avolume_profiles = []
    aentries = []
    v_data = []

    if data_splits[1] == []:
        data_splits = [data_splits[0]]

    dt = str(data_splits[0][-1].time)[:10]


    for split_index, candles in enumerate(data_splits):
        print(f"\n🔹 Processing split {split_index+1}/{len(data_splits)} "
              f"({candles[0].time} → {candles[-1].time})")

        results = []
        ma_dict = calculate_moving_averages(candles, periods=(short_ma, long_ma))
        cross_up, cross_down = detect_ma_crossovers(ma_dict[short_ma], ma_dict[long_ma])

        # === Build Volume Profiles between crossovers ===
        volume_profiles = []
        all_crosses = sorted(
            [(i, "up") for i in cross_up] + [(i, "down") for i in cross_down],
            key=lambda x: x[0]
        )

        for i in range(len(all_crosses) - 1):
            idx_a, _ = all_crosses[i]
            idx_b, _ = all_crosses[i + 1]

            A = pd.Timestamp(candles[idx_a].time).tz_localize(None)
            B = pd.Timestamp(candles[idx_b].time).tz_localize(None)

            if candles[idx_a].symbol != candles[idx_b].symbol:
                continue

            vp = calculate_volume_profile_from_trades(dataset, A, B, symbol=candles[idx_a].symbol, bins=160)
            if vp is None:
                continue

            poc = calculate_poc(vp)
            volume_profiles.append({"A": A, "B": B, "poc": poc})
            avolume_profiles.append({"A": A, "B": B, "poc": poc})

        entries = detect_entries(candles, volume_profiles)
        print(f"🔸 Found {len(entries)} entries in this split.")

        used_symbols.append(s)
        aentries.extend(entries)

        if not entries:
            print("⚠️ No entries detected, skipping Variant creation for this split.")
            v_data.append([])  # keep structure consistent
            continue

        # === Create Variants ===
        for stop in [10, 20]:
            contracts = 1
            if stop == 10: contracts = 2

            for r in [0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
                name = f"{stop}_{r}"
                stop_losses = [stop / 100000] * len(entries)
                rrratios = [r] * len(entries)
                

                if split_index: dt = last_date

                try:
                    variant = Variant(entries, candles, stop_losses, rrratios, dt, name, c=contracts)
                    if variant.metrics:  # only keep if metrics exist
                        results.append(variant)
                except Exception as e:
                    print(f"⚠️ Error creating Variant {name}: {e}")

        v_data.append(results)

    # === Handle empty splits ===
    if not any(v_data):
        print("❌ No data splits produced valid variants.")
        return

    # === Rank and visualize ===
    fit_data = sorted(v_data[0], key=lambda x: x.metrics.get("Sharpe Ratio", 0), reverse=True)[:] if v_data[0] else []

    ama_dict = calculate_moving_averages(all_candles, periods=(short_ma, long_ma))
    across_up, across_down = detect_ma_crossovers(ama_dict[short_ma], ama_dict[long_ma])

    visualize_candles(
        all_candles,
        t=f"6E {freq} Candlestick Chart",
        moving_averages=ama_dict,
        cross_up=across_up,
        cross_down=across_down,
        volume_profiles=avolume_profiles,
        entries=aentries,
        sl=0.0002,
        rrr=10
    )

    # === Print summary ===
    if fit_data:
        for i, v in enumerate(fit_data):
            print(f"\n✅ Fit Variant {i+1}: {v.name}")
            for m in v.metrics:
                print(f"{m}: {v.metrics[m]}")
    else:
        print("⚠️ No fit data available.")

    if len(v_data) > 1:
        test_data = sorted(v_data[1], key=lambda x: x.metrics.get("Sharpe Ratio", 0), reverse=True)[:] if v_data[1] else []
        if test_data:
            for i, v in enumerate(test_data):
                print(f"\n🧩 Test Variant {i+1}: {v.name}")
                for m in v.metrics:
                    print(f"{m}: {v.metrics[m]}")
        else:
            print("⚠️ No test data available.")

if __name__ == "__main__":

    # === Load data ===
    date = int(sys.argv[1])
    dataset, days, first_date, last_date = load_raw_data(date, max_files=None)

    freq = "25min"
    symbols = ["6EH4", "6EM4", "6EU4", "6EZ4", "6EH5", "6EM5", "6EU5", "6EZ5"]
    roll_schedule = {
        "6EH4": ("2024-12-14", "2024-03-15"),
        "6EM4": ("2024-03-15", "2024-06-14"),
        "6EU4": ("2024-06-14", "2024-09-14"),
        "6EZ4": ("2024-09-14", "2024-12-14"),
        "6EH5": ("2024-12-14", "2025-03-15"),
        "6EM5": ("2025-03-15", "2025-06-14"),
        "6EU5": ("2025-06-14", "2025-09-14"),
        "6EZ5": ("2025-09-14", "2025-12-14"),
    }

    # === Build all candles ===
    all_candles = []
    for s in symbols:
        for t in make_candles(dataset, freq, symbol=s, roll_schedule=roll_schedule):
            all_candles.append(t)

    # === Split data (fit/test) ===
    data_splits = separate_data(all_candles, days, first_date, last_date, 1)
    run_strategy(dataset, all_candles, data_splits, freq)

'''
Todo

Resolver situacao dos simbolos       OK
calcular Moving Average Crossings    OK
Calcular Volume Profile              OK
Plotar POC                           OK
Add entry signals                    OK
Fix decide_entry_direction           OK
Calculare result from each entrie    OK
Fix symbol with duplicate entries    OK
Fix sharpe calculation               OK
Add Costs                            OK
Add Slippage                         OK
Plot Entries stop and tp             OK
Fix POC step (0.0005)                OK
Compute sharpe using % daily returns
Add Metric Average Trade Duration
Control Trade Duration               OK
Optimize load_data function
Optimize volume profile function     OK
Generate p&l graph function
Fit and Test separated               OK
In Sample Permutation Test
'''