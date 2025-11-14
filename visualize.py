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
from collections import defaultdict

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

def calculate_moving_averages_and_crossovers(all_candles, short_ma, long_ma, per_symbol=False):
    """
    Calculates MAs and Crossovers correctly, respecting symbol boundaries
    to avoid roll gap corruption.
    
    This function performs a "split-apply-combine" operation:
    1. SPLIT: Groups all candles by their symbol.
    2. APPLY: Calculates MAs and crossovers for each symbol independently.
    3. COMBINE: Maps the local results back to their global indices.
    
    Returns:
        (dict): The correct, globally-mapped ma_dict.
        (list): The correct, sorted list of global cross_up indices.
        (list): The correct, sorted list of global cross_down indices.
    """
    periods = (short_ma, long_ma)
    
    if not per_symbol:
        print("Calculating MAs and crossovers globally...")
        # Compute global MAs
        ma_dict = calculate_moving_averages(all_candles, periods=periods)

        # Detect crossovers globally
        cross_up, cross_down = detect_ma_crossovers(
            ma_dict[short_ma], ma_dict[long_ma]
        )

        # Return directly
        return ma_dict, cross_up, cross_down

    print("Calculating MAs and crossovers per-symbol to fix roll gaps...")    
    # 1. Map: candle's memory ID -> its global index in 'all_candles'
    global_index_map = {id(c): i for i, c in enumerate(all_candles)}

    # 2. (SPLIT) Group all candles by their symbol
    symbol_candle_groups = defaultdict(list)
    for c in all_candles:
        symbol_candle_groups[c.symbol].append(c)
        
    # 3. Initialize collectors for combined, global data
    correct_cross_up = []
    correct_cross_down = []
    correct_ma_dict = {
        period: [None] * len(all_candles) for period in periods
    }
    
    # 4. (APPLY) Loop over each symbol's group
    for symbol, sym_candles in symbol_candle_groups.items():
        # Need enough candles for the *longest* MA
        if len(sym_candles) < long_ma: 
            continue
            
        # Calculate MAs *only* for this single symbol's candles
        ma_dict_local = calculate_moving_averages(sym_candles, periods=periods)
        
        # Find crossovers. These are *local* indices
        local_cross_up, local_cross_down = detect_ma_crossovers(
            ma_dict_local[short_ma], ma_dict_local[long_ma]
        )

        # 5. (COMBINE) Map local results back to global lists
        
        # Map crossover indices
        for local_idx in local_cross_up:
            correct_cross_up.append(global_index_map[id(sym_candles[local_idx])])
            
        for local_idx in local_cross_down:
            correct_cross_down.append(global_index_map[id(sym_candles[local_idx])])

        # Map MA values
        for period in periods:
            local_ma_list = ma_dict_local[period]
            
            for local_idx, candle_obj in enumerate(sym_candles):
                global_idx = global_index_map[id(candle_obj)]
                correct_ma_dict[period][global_idx] = local_ma_list[local_idx]

    # 6. Return the combined, correct results
    return correct_ma_dict, sorted(correct_cross_up), sorted(correct_cross_down)

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

                    
def detect_entries(candles, volume_profiles, same_symbols=False):
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
        vp_symbol = vp["symbol"]

        # Find the candle that matches B
        b_candle = next((c for c in candles if pd.Timestamp(c.time).tz_localize(None) == B_time), None)
        if b_candle is None:
            print(f"⚠️ Could not find B candle at {B_time}")
            continue

        if same_symbols and b_candle.symbol != vp_symbol:
            print(f"⚠️ B-candle symbol ({b_candle.symbol}) does not match VP symbol ({vp_symbol}) at {B_time}")
            continue

        b_close = b_candle.close

        for i in range(len(candles)):
            c = candles[i]
            if pd.Timestamp(c.time).tz_localize(None) <= B_time:
                continue  # only check after B

            if same_symbols and c.symbol != vp_symbol:
                break  # Stop search, we are on a new contract

            entry_type = decide_entry_direction(b_close, poc, c)

            if entry_type:
                entries.append({ "B": B_time, "entry_time": times[i], "entry_price": poc, "candle_index": i, "entry_type": entry_type, "symbol": vp_symbol})
                break  # stop after first touch

    return entries

def result_from_entries(entries, candles, stop_losses, rrratios,  last_date, win_size=125, costs=0.84*2, contracts=1, slipage_on_losses=12.5):
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
                    print(f"→ STOP HIT at {candle_time}, result={result:.2f}")
                    trade_closed = True
                    break
                elif high >= target_level:
                    result = rr * win_size - costs * contracts
                    print(f"→ TARGET HIT at {candle_time}, result={result:.2f}")
                    trade_closed = True
                    break
            else:
                if high >= stop_level:
                    result = -1 * win_size - costs * contracts - slipage_on_losses
                    print(f"→ STOP HIT at {candle_time}, result={result:.2f}")
                    trade_closed = True
                    break
                elif low <= target_level:
                    result = rr * win_size - costs * contracts
                    print(f"→ TARGET HIT at {candle_time}, result={result:.2f}")
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
            close = candles[-1].close
            if direction == "long":
                result = (close - entry_price) / (sl_dist * 5) * win_size - costs * contracts
            else:
                result = (entry_price - close) / (sl_dist * 5) * win_size - costs * contracts
            print(f"Trade remained open — closing at last candle price ({close:.5f}), result={result:.2f}")

        results.append((entry_time, result))

    # Ensure trade_list is sorted chronologically
    trade_list = sorted(results, key=lambda x: x[0])

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

    return l, trade_list

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

def _calculate_daily_sharpe(excess_returns):
    """Helper function to calculate Sharpe for a bootstrap sample."""
    # ddof=1 for sample standard deviation
    std_dev = np.std(excess_returns, ddof=1) 
    
    if std_dev == 0:
        # Handle cases with zero volatility in a resample (return 0 or np.nan)
        return 0.0
        
    return np.mean(excess_returns) / std_dev

def calculate_sharpe(returns, annual_risk_free_rate=0.0, trading_days=252):
    """
    Calculate daily and annualized Sharpe ratio and 95% bootstrap CI.
    
    Parameters:
    - returns: array-like of daily returns.
    - annual_risk_free_rate: The *annual* risk-free rate (e.g., 0.05 for 5%).
    - trading_days: Number of trading days in a year (e.g., 252).
    """
    returns = np.array(returns, dtype=float)
    if len(returns) < 2: # Need at least 2 returns to calculate std dev
        return 0.0, 0.0, (np.nan, np.nan)

    # Convert annual RF rate to daily RF rate
    daily_risk_free_rate = annual_risk_free_rate / trading_days
    
    # 1. Calculate Excess Returns
    excess = returns - daily_risk_free_rate

    # 2. Calculate Point Estimates (Daily and Annualized Sharpe)
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)

    if std_excess == 0:
        # Handle case of zero volatility in the main sample
        sharpe = 0.0
        annualized_sharpe = 0.0
    else:
        sharpe = mean_excess / std_excess
        annualized_sharpe = sharpe * np.sqrt(trading_days)

    # 3. Bootstrap 95% confidence interval for the *daily* Sharpe
    try:
        # Pass the 'excess' returns as the data
        # Pass our helper function as the statistic to bootstrap
        res = bootstrap(
            (excess,),
            _calculate_daily_sharpe,  # <-- This is the crucial change
            confidence_level=0.95,
            random_state=42,
            n_resamples=5000,
            method="percentile" )
        
        # The result 'res.confidence_interval' is for the DAILY Sharpe
        ci_low_daily = res.confidence_interval.low
        ci_high_daily = res.confidence_interval.high
        
        # 4. Annualize the confidence interval bounds
        annualization_factor = np.sqrt(trading_days)
        ci_low = ci_low_daily * annualization_factor
        ci_high = ci_high_daily * annualization_factor
        
    except Exception:
        ci_low, ci_high = np.nan, np.nan

    return sharpe, annualized_sharpe, (ci_low, ci_high)

def calculate_winrate(returns):
    """Percentage of positive returns."""
    returns = np.array(returns, dtype=float)
    if len(returns) == 0:
        return 0.0
    wins = np.sum(returns > 0)
    return (wins / np.sum(returns != 0)) * 100

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


def compute_data(l, last_date, n, trade_list):
    """
    Compute all key strategy metrics.
    """
    t = len(l)

    # Group the list into n groups the first having t%n more elements
    d = []
    r = t % n
    for i in range(n):
        v = []
        for j in range((t//n)+r):
            v.append(l.pop(0))
        r = 0
        d.append(v)
    
    data = []
    for i in range(len(d)-1):
        l_fit = d[:i+1]
        l_test = d[i+1:]

        l_fit = [item for sublist in l_fit for item in sublist]
        l_test = [item for sublist in l_test for item in sublist]
        data.append([l_fit, l_test])
        
    if n == 1: data = [ [d[0], []] ]


    results = []
    for p in data:
        r = []
        for l in p:
            if l == []: r.append({})
            sharpe, annualized_sharpe, ci = calculate_sharpe(l)
            winrate = calculate_winrate(l)
            max_dd = calculate_max_drawdown(l)
            profit_factor = calculate_profit_factor(l)
            expectancy = calculate_expectancy(l)

            r.append({
                "Trade Returns": [i[1] for i in trade_list],
                'Daily Returns': l,
                "Sharpe Ratio": round(sharpe, 3),
                "Annualized Sharpe": round(annualized_sharpe, 3),
                "Sharpe 95% CI": (round(ci[0], 3), round(ci[1], 3)),
                "Daily Win Rate (%)": round(winrate, 2),
                "Max Drawdown": round(max_dd, 3),
                "Profit Factor": round(profit_factor, 3),
                "Expectancy": round(expectancy, 3),
                "Total Trades (Global)": len(trade_list),
                "Total Days": len(l)
            })
        results.append(r)

    return results

def print_results(results, number, best=5):
    for i in range(number-1):
        k = []
        for variant in results:
            k.append((variant, i ))

        k = sorted(k, key=lambda x: x[0].metrics[x[1]][0].get("Sharpe Ratio", 0), reverse=True)[:best]

        for j, v in enumerate(k):

            print(f"\n\n\nRolling {i} - Variant {j} - {v[0].name} - Fit Metrics")
            for m in v[0].metrics[v[1]][0]:
                print(f"{m}: {v[0].metrics[v[1]][0][m]}")

            print(f"\nRolling {i} - Variant {j} - {v[0].name} - Test Metrics")
            for m in v[0].metrics[v[1]][1]:
                print(f"{m}: {v[0].metrics[v[1]][1][m]}")

class Variant():
    def __init__(self, entries, candles, stop_losses, rrratios, last_date, n, num, c=1):
        self.entries = entries
        self.candles = candles
        self.stop_losses = stop_losses
        self.rrratios = rrratios
        
        self.name = n
        self.results, self.trade_list = result_from_entries(entries, candles, stop_losses, rrratios, last_date, contracts=c)
        #print(len(self.results)) # 312 days, 26 per month
        #quit()
        self.metrics =  compute_data(self.results, last_date, num, self.trade_list)


def run_strategy(dataset, all_candles, num, title=f"6E 60min Candlestick Chart"):
    print(f"Running strategy on {len(all_candles)} candles.")

    short_ma = 40
    long_ma = 50

    avolume_profiles = []
    aentries = []

    results = []


    dt = str(all_candles[-1].time)[:10]


    # === Calculate Moving Averages ===
    ma_dict, cross_up, cross_down = calculate_moving_averages_and_crossovers(all_candles, short_ma, long_ma)
    print(f"Found {len(cross_up)} valid cross-ups and {len(cross_down)} valid cross-downs.")


    # === Build Volume Profiles between crossovers ===
    volume_profiles = []
    all_crosses = sorted(
        [(i, "up") for i in cross_up] + [(i, "down") for i in cross_down],
        key=lambda x: x[0]
    )

    for i in range(len(all_crosses) - 1):
        idx_a, _ = all_crosses[i]
        idx_b, _ = all_crosses[i + 1]

        A = pd.Timestamp(all_candles[idx_a].time).tz_localize(None)
        B = pd.Timestamp(all_candles[idx_b].time).tz_localize(None)

        if all_candles[idx_a].symbol != all_candles[idx_b].symbol:
            continue

        vp = calculate_volume_profile_from_trades(dataset, A, B, symbol=all_candles[idx_a].symbol, bins=160)
        if vp is None:
            continue

        if all_candles[idx_a].symbol != all_candles[idx_b].symbol:
            raise ValueError(f"Symbol mismatch: {all_candles[idx_a].symbol} != {all_candles[idx_b].symbol}")
        
        current_symbol = all_candles[idx_a].symbol
        poc = calculate_poc(vp)
        volume_profiles.append({"A": A, "B": B, "poc": poc, "symbol": current_symbol})
        avolume_profiles.append({"A": A, "B": B, "poc": poc})

    entries = detect_entries(all_candles, volume_profiles)
    print(f"🔸 Found {len(entries)} entries in this split.")

    aentries.extend(entries)

    # === Create Variants ===
    for stop in [10, 20]:
        contracts = 5
        if stop == 10: contracts = 10


        for r in [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
            name = f"{stop}_{r}"
            stop_losses = [stop / 100000] * len(entries)
            rrratios = [r] * len(entries)
            
            try:
                variant = Variant(entries, all_candles, stop_losses, rrratios, dt, name, num, c=contracts)
                results.append(variant)
            except Exception as e:
                print(f"⚠️ Error creating Variant {name}: {e}")

    visualize_candles(
        all_candles,
        t=title,
        moving_averages=ma_dict,
        cross_up=cross_up,
        cross_down=cross_down,
        volume_profiles=avolume_profiles,
        entries=aentries,
        sl=0.0001,
        rrr=7
    )

    print_results(results, num)


if __name__ == "__main__":

    # === Load data ===
    date = int(sys.argv[1])
    p = sys.argv[2]
    dataset, days, first_date, last_date = load_raw_data(date, path=p, max_files=None)


    freq = "60min"

    if p == "data6E":
        symbols = ["6EH4", "6EM4", "6EU4", "6EZ4", "6EH5", "6EM5", "6EU5", "6EZ5"]
        roll_schedule = {
            "6EH4": ("2023-12-14", "2024-03-15"),
            "6EM4": ("2024-03-15", "2024-06-14"),
            "6EU4": ("2024-06-14", "2024-09-14"),
            "6EZ4": ("2024-09-14", "2024-12-14"),
            "6EH5": ("2024-12-14", "2025-03-15"),
            "6EM5": ("2025-03-15", "2025-06-14"),
            "6EU5": ("2025-06-14", "2025-09-14"),
            "6EZ5": ("2025-09-14", "2025-12-14"),
        }
        tit = f"6E {freq} Candlestick Chart"

    if p == "dataM6E":
        symbols = ["M6EH2", "M6EM2", "M6EU2", "M6EZ2","M6EH3", "M6EM3", "M6EU3", "M6EZ3", "M6EH4", "M6EM4", "M6EU4", "M6EZ4", "M6EH5", "M6EM5", "M6EU5", "M6EZ5"]
        roll_schedule = {
            "M6EH2": ("2021-12-11", "2022-03-12"),
            "M6EM2": ("2022-03-12", "2022-06-11"),
            "M6EU2": ("2022-06-11", "2022-09-17"),
            "M6EZ2": ("2022-09-17", "2022-12-17"),
            "M6EH3": ("2022-12-17", "2023-03-11"),
            "M6EM3": ("2023-03-11", "2023-06-17"),
            "M6EU3": ("2023-06-17", "2023-09-16"),
            "M6EZ3": ("2023-09-16", "2023-12-16"),
            "M6EH4": ("2023-12-16", "2024-03-16"),
            "M6EM4": ("2024-03-16", "2024-06-15"),
            "M6EU4": ("2024-06-15", "2024-09-14"),
            "M6EZ4": ("2024-09-14", "2024-12-14"),
            "M6EH5": ("2024-12-14", "2025-03-15"),
            "M6EM5": ("2025-03-15", "2025-06-14"),
            "M6EU5": ("2025-06-14", "2025-09-13"),
            "M6EZ5": ("2025-09-13", "2025-12-13"),
        }
        tit = f"M6E {freq} Candlestick Chart"

    # === Build all candles ===
    all_candles = []
    for s in symbols:
        for t in make_candles(dataset, freq, symbol=s, roll_schedule=roll_schedule):
            all_candles.append(t)

    # === Split data (fit/test) ===
    run_strategy(dataset, all_candles, 12, tit)

'''
Todo

Resolver situacao dos simbolos          OK
calcular Moving Average Crossings       OK
Calcular Volume Profile                 OK
Plotar POC                              OK
Add entry signals                       OK
Fix decide_entry_direction              OK
Calculate result from each entrie       OK
Fix symbol with duplicate entries       OK
Fix sharpe calculation                  OK
Add Costs                               OK
Add Slippage                            OK
Plot Entries stop and tp                OK
Fix POC step (0.0005)                   OK
Fix moving Average  Mistake             OK
Make Moving Average Fix Clean           OK
Fix Sharpe Calculation                  OK
Review Trade Closing
Review for other possible mistakes      OK
Compute sharpe using % daily returns 
Limit trades to same contract as vp     OK
Add Metric Average Trade Duration
Add Metric Avg Win
Add Metric Avg Loss
Add Metric Number of days               OK
Fix metrics Total Trades                OK
Control Trade Duration                  OK
Optimize entry function
Optimize load_data function
Optimize volume profile function        OK
Generate p&l graph function
Fit and Test separated (out of sample)  OK
Fit and Test rolling out of sample      OK
In Sample Permutation Test
'''