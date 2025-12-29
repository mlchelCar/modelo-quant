import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import load_raw_data, make_candles, separate_data
from metrics import *
import numpy as np
import sys
from pandas import Timestamp
from numba import njit
import pandas.api.types as ptypes
from collections import defaultdict
from collections import Counter
import time
import datetime
import math

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

def calculate_volume_profile_from_trades_fast(trades_df, A, B, symbol=None, bins=80):
    print(f"\nCalculating fast volume profile from {len(trades_df)} trades...")
    print(f"  A: {A}, B: {B}, symbol: {symbol}, bins: {bins}")

    df = trades_df

    # --- Time handling ---
    if "ts_event" in df.columns:
        time = df["ts_event"].values
    elif "time" in df.columns:
        time = df["time"].values
    else:
        raise ValueError("No valid time column found")

    A = np.datetime64(A)
    B = np.datetime64(B)

    mask = (time >= A) & (time <= B)

    if symbol is not None and "symbol" in df.columns:
        mask &= (df["symbol"].values == symbol)

    if not mask.any():
        return None

    prices  = df["price"].values[mask].astype(np.float64, copy=False)
    volumes = df["size"].values[mask].astype(np.float64, copy=False)

    # --- Volume profile ---
    pmin, pmax = prices.min(), prices.max()
    bin_edges = np.linspace(pmin, pmax, bins + 1)

    vol, _ = np.histogram(prices, bins=bin_edges, weights=volumes)
    price_bins = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return {
        "price_bins": price_bins,
        "volumes": vol,
        "A": A,
        "B": B,
        "symbol": symbol
    }

def calculate_poc(vp, tick_size, return_volume=False):
    """Return the price of the POC (and optionally the volume)."""
    if vp is None or len(vp["volumes"]) == 0:
        raise ValueError("Invalid or empty volume profile data")

    idx_max = np.argmax(vp["volumes"])
    poc_price = round(vp["price_bins"][idx_max],5)
    poc_price = round(round(poc_price / tick_size) * tick_size, 5)
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
                    
def detect_entries(candles, volume_profiles, same_symbols_A_B=False, same_symbols_A_entry=False):
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
        vp_symbolA = vp["a_candle"].symbol
        vp_symbolB = vp["b_candle"].symbol

        # Find the candle that matches B
        b_candle = vp["b_candle"]

        if same_symbols_A_B and vp_symbolB != vp_symbolA:
            print(f"⚠️ B-candle symbol ({b_candle.symbol}) does not match A-candle symbol ({vp_symbolA}) at {B_time}")
            continue

        b_close = b_candle.close

        for i in range(len(candles)):
            c = candles[i]
            if pd.Timestamp(c.time).tz_localize(None) <= B_time:
                continue  # only check after B

            if same_symbols_A_entry and c.symbol != vp_symbolA:
                break  # Stop search, we are on a new contract

            entry_type = decide_entry_direction(b_close, poc, c)

            if entry_type:
                entries.append({ "B": B_time, "entry_time": times[i], "entry_price": poc, "candle_index": i, "entry_type": entry_type})
                break  # stop after first touch

    return entries

def compute_time_exit(entry_time, exit_mode):
    if exit_mode == "EOD":
        if entry_time.hour < 21:
            return entry_time.replace(hour=21, minute=0, second=0, microsecond=0)
        else:
            return (entry_time + pd.Timedelta(days=1)).replace(
                hour=21, minute=0, second=0, microsecond=0)

    elif exit_mode == "EOW":
        # Friday = 4 (Monday=0)
        weekday = entry_time.weekday()

        # Days until Friday
        days_to_friday = 4 - weekday

        if days_to_friday == 0 and entry_time.hour >= 21:
            days_to_friday = 7
        elif weekday == 6:
            days_to_friday += 5
        elif weekday == 5:
            raise ValueError("Entry on Saturday not supported")
            
        # Candidate Friday 21:00
        friday_close = (entry_time + pd.Timedelta(days=days_to_friday)).replace(
            hour=21, minute=0, second=0, microsecond=0)

        return friday_close

    elif exit_mode == "EOM":
        raise NotImplementedError("EOM not implemented yet")

    elif exit_mode == "NONE":
        return None

    else:
        raise ValueError(f"Unknown exit_mode: {exit_mode}")

def result_from_entries(entries, candles, stop_losses, rrratios, last_date,tick_size_in_price, tick_value, costs, capital, symbol, exit_mode):
    results = []
    contracts = []
    slippage_ticks = {
    "6E": 0.5,
    "M6E": 0.5,
    "MCL": 1.0}
    
    for entry, sl_ticks, rr in zip(entries, stop_losses, rrratios):
        c = determine_contracts_volatility(entry, tick_size_in_price, tick_value, capital, target_vol=0.10)

        if sl_ticks is None or c == 0:
            continue
        contracts.append(c)

        slippage_cost = slippage_ticks[symbol] * tick_value * c
        entry_idx   = entry["candle_index"]
        entry_time = entry["entry_time"]
        entry_price = entry["entry_price"]
        direction = entry["entry_type"]


        if entry_time.tzinfo is not None:
            entry_time = entry_time.tz_convert(None)

        # stop distance in PRICE
        stop_dist_price = sl_ticks * tick_size_in_price

        # stop/target levels
        if direction == "long":
            stop_level   = entry_price - stop_dist_price
            target_level = entry_price + rr * stop_dist_price
        else:
            stop_level   = entry_price + stop_dist_price
            target_level = entry_price - rr * stop_dist_price

        trade_closed = False
        result = 0
        exit_time = None

        # determine closing time
        closing_time = compute_time_exit(entry_time, exit_mode)

        # for candle in candles:
        for candle in candles[entry_idx + 1:]:
            candle_time = candle.time
            if candle_time.tzinfo is not None:
                candle_time = candle_time.tz_convert(None)

            if candle_time <= entry_time:
                continue

            high, low, close = candle.high, candle.low, candle.close

            # stop hit
            if direction == "long" and low <= stop_level:
                result = -sl_ticks * tick_value * c - costs * c - slippage_cost
                trade_closed = True
                exit_time = candle_time
                break
            if direction == "short" and high >= stop_level:
                result = -sl_ticks * tick_value * c - costs * c - slippage_cost
                trade_closed = True
                exit_time = candle_time
                break

            # target hit
            if direction == "long" and high >= target_level:
                result = rr * sl_ticks * tick_value * c - costs * c
                trade_closed = True
                exit_time = candle_time
                break
            if direction == "short" and low <= target_level:
                result = rr * sl_ticks * tick_value * c - costs * c
                trade_closed = True
                exit_time = candle_time
                break

            # time exit
            if closing_time is not None and candle_time >= closing_time:
                ticks_pnl = (close - entry_price) / tick_size_in_price
                if direction == "short":
                    ticks_pnl = -ticks_pnl

                result = ticks_pnl * tick_value * c - costs * c
                trade_closed = True
                exit_time = candle_time
                break

            prev_close = close

        # fallback
        if not trade_closed:
            close = candles[-1].close
            ticks_pnl = (close - entry_price) / tick_size_in_price
            if direction == "short":
                ticks_pnl = -ticks_pnl

            result = ticks_pnl * tick_value * c - costs * c
            exit_time = candles[-1].time
            if exit_time.tzinfo is not None:
                exit_time = exit_time.tz_convert(None)

        results.append((exit_time, result/capital))
        capital += result

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

    # --- NEW: ensure chronological order ---
    dates = sorted(daily_returns.keys())              # oldest → newest
    l = [daily_returns[d] for d in dates]             # aligned returns

    return l, dates, trade_list, contracts

def visualize_candles(candles, stds, atr, t="Candlestick Chart", moving_averages=None, cross_up=None, cross_down=None, volume_profiles=None, entries=None, sl=None, rrr=None):
    times = [pd.Timestamp(c.time).tz_localize(None) for c in candles]
    opens, highs, lows, closes, volumes = zip(*[(c.open, c.high, c.low, c.close, c.volume) for c in candles])
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(candles))]

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.60, 0.20, 0.1, 0.1])
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

    # === STD PANEL (Row 3) ===
    fig.add_trace(go.Scatter(x=times, y=stds, mode="lines", name="STD", line=dict(width=1.5, color="blue")), row=3, col=1)

    # === STD PANEL (Row 4) ===
    fig.add_trace(go.Scatter(x=times, y=atr, mode="lines", name="ATR", line=dict(width=1.5, color="red")), row=4, col=1)

    fig.update_layout(title=t, xaxis_rangeslider_visible=False, height=800, hovermode='x unified', xaxis=dict(type='date'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.show()

OUTPUT_TEXT = []
def save_output(s):
    print(s)
    OUTPUT_TEXT.append(s + "\n")

def write_output_to_file(title):
    filename = f"{title}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.writelines(OUTPUT_TEXT)

def print_results(results, number, best=5):
    # Counter for how many times each variant appears in the top 'best'
    appearance_counter = Counter()
    ranking_results = []   # store (rolling, rank, variant) for later plots

    # --- FIRST PASS: COUNT APPEARANCES ---
    for i in range(number - 1):
        k = []
        for variant in results:
            k.append((variant, i ))

        # sort by Sharpe Ratio in FIT
        k = sorted(
            k,
            key=lambda x: x[0].metrics[x[1]][0].get("Sharpe Ratio", 0),
            reverse=True
        )[:best]

        # Track appearances
        for rank, (variant, r) in enumerate(k):
            appearance_counter[variant.name] += 1
            ranking_results.append((r, rank, variant))

        # Print results (your original code)
        for j, v in enumerate(k):
            save_output(f"\n\n\nRolling {i} - Variant {j} - {v[0].name} - Fit Metrics")
            save_output(f"Contracts: {v[0].contracts}")
            for m in v[0].metrics[v[1]][0]:
                save_output(f"{m}: {v[0].metrics[v[1]][0][m]}")

            save_output(f"\nRolling {i} - Variant {j} - {v[0].name} - Test Metrics")
            save_output(f"Contracts: {v[0].contracts}")
            for m in v[0].metrics[v[1]][1]:
                save_output(f"{m}: {v[0].metrics[v[1]][1][m]}")

    # Print summary BEFORE printing each rolling result
    save_output("\n========== Variant Appearance Count (FIT Rankings) ==========")
    for name, count in sorted(appearance_counter.items(), key=lambda x: -x[1]):
        save_output(f"{name}: {count} times")

    best_variant_name, _ = appearance_counter.most_common(1)[0]
    best_variant = next(v for v in results if v.name == best_variant_name)

    save_output("\n============================================")
    save_output(f"BEST VARIANT:{best_variant.name}")
    save_output(f"Total Fit (top {best}) Appearances: {appearance_counter[best_variant_name]}")
    save_output("============================================")

    print_best_variant_details(best_variant)

    make_graphs(best_variant)

def print_results_2(results, number):
    for variant in results:
        save_output(f"\n========== Variant {variant.name} ==========")

        for i in range(number - 1):
            test_metrics = variant.metrics[i][1]  # TEST metrics

            sharpe = test_metrics.get("Annualized Sharpe", None)
            sharpe_ci = test_metrics.get("Sharpe 95% CI", None)

            save_output(f"\nRolling {i}:")
            save_output(f"  Sharpe Ratio (TEST):   {sharpe}")
            save_output(f"  Sharpe 95% CI (TEST):  {sharpe_ci}")

def print_results_3(results, number, variant_name="atr_0.5_8"):
    for variant in results:
        if variant.name != variant_name: continue
        save_output(f"\n========== Variant {variant.name} ==========")

        for i in range(number - 1):
            test_metrics = variant.metrics[i][1]  # TEST metrics

            dreturns = test_metrics.get("Daily Returns", None)
            sharpe = test_metrics.get("Annualized Sharpe", None)
            sharpe_ci = test_metrics.get("Sharpe 95% CI", None)

            save_output(f"\nRolling {i}:")
            save_output(f"  Daily Returns (TEST): {dreturns}")
            save_output(f"  Sharpe Ratio (TEST):   {sharpe}")
            save_output(f"  Sharpe 95% CI (TEST):  {sharpe_ci}")

def print_best_variant_details(best_variant):
    save_output("\n========== Best Variant Detailed Metrics ==========\n")
    save_output(f"Variant Name: {best_variant.name}")
    save_output("---------------------------------------------------")

    for i, rolling in enumerate(best_variant.metrics):
        fit = rolling[0]   # dictionary of FIT metrics
        test = rolling[1]  # dictionary of TEST metrics

        save_output(f"\n================ Rolling {i+1} ================")

        # =============================
        # FIT METRICS
        # =============================
        # save_output("\n--- FIT Metrics ---")
        # if fit != {}:
            # save_output(f"Start Date: {test.get('Start Date', 'N/A')}")
            # save_output(f"End Date:   {test.get('End Date','N/A')}")

            # save_output(f"Sharpe Ratio:        {test['Sharpe Ratio']}")
            # save_output(f"Annualized Sharpe:   {test['Annualized Sharpe']}")
            # save_output(f"Sortino Ratio:       {test.get('Sortino Ratio', 'N/A')}")
            # save_output(f"Calmar Ratio:        {test.get('Calmar Ratio', 'N/A')}")
            # save_output(f"CAGR:                {test.get('CAGR', 'N/A')}")

            # save_output(f"Sharpe 95% CI:       {test['Sharpe 95% CI']}")
            # save_output(f"Daily Win Rate (%):  {test['Daily Win Rate (%)']}")
            # save_output(f"Max Drawdown:        {test['Max Drawdown']}")

            # save_output(f"Profit Factor:       {test['Profit Factor']}")
            # save_output(f"Expectancy:          {test['Expectancy']}")
            # save_output(f"Average Win:         {test['Average Win']:.2f}")
            # save_output(f"Average Loss:        {test['Average Loss']:.2f}")

            # save_output(f"Total Trades:        {test['Total Trades']}")
            # save_output(f"Total Days:          {test['Total Days']}")
        # else:
        #     save_output("No FIT data (this is the last rolling window).")

        # =============================
        # TEST METRICS
        # =============================
        save_output("\n--- TEST Metrics ---")
        if test != {}:
            save_output(f"Start Date: {test.get('Start Date', 'N/A')}")
            save_output(f"End Date:   {test.get('End Date','N/A')}")

            save_output(f"Sharpe Ratio:        {test['Sharpe Ratio']}")
            save_output(f"Annualized Sharpe:   {test['Annualized Sharpe']}")
            save_output(f"Sortino Ratio:       {test.get('Sortino Ratio', 'N/A')}")
            save_output(f"Calmar Ratio:        {test.get('Calmar Ratio', 'N/A')}")
            save_output(f"CAGR:                {test.get('CAGR', 'N/A')}")

            save_output(f"Sharpe 95% CI:       {test['Sharpe 95% CI']}")
            save_output(f"Daily Win Rate (%):  {test['Daily Win Rate (%)']}")
            save_output(f"Max Drawdown:        {test['Max Drawdown']}")

            save_output(f"Profit Factor:       {test['Profit Factor']}")
            save_output(f"Expectancy:          {test['Expectancy']}")
            save_output(f"Average Win:         {test['Average Win']:.2f}")
            save_output(f"Average Loss:        {test['Average Loss']:.2f}")

            # save_output(f"Total Trades:        {test['Total Trades']}")
            # save_output(f"Total Days:          {test['Total Days']}")

        else:
            save_output("No TEST data in this rolling window.")

def make_graphs(variant):
    pass

class Variant():
    def __init__(self, entries, candles, stop_losses, rrratios, last_date, n, num, sm):
        self.entries = entries
        self.candles = candles
        self.stop_losses = stop_losses
        self.rrratios = rrratios
        self.sm = sm

        self.last_date = last_date
        self.num = num
        self.name = n
    
    def compute(self, tick_size, tick_value, costs, capital, exit_type):
        self.results, dates, self.trade_list, self.contracts = result_from_entries(self.entries, self. candles, self.stop_losses, self.rrratios, self.last_date, tick_size, tick_value, 2*costs, capital, self.sm, exit_type)
        self.metrics =  compute_data(self.results, dates, self.num, self.trade_list)
        print(f"Variant {self.name} created.")

def determine_stop_losses(stop_type, entries, v, tick_size):
    if stop_type == "fixed": return [v] * len(entries)

    stops = []
    if stop_type == "atr":
        for e in entries:
            atr = e.get("atr")
            if atr is None: stops.append(None)
            else: stops.append(round(v * atr / tick_size))
        return stops

def determine_contracts(entries, stop_size):
    if stop_size == 20: return [5]*len(entries)
    if stop_size == 10: return [10]*len(entries)

def determine_contracts_volatility(e, tick_size, tick_value, capital, target_vol=0.10):
    # Convert annual target vol → daily target dollar P&L volatility
    target_daily_vol = (capital * target_vol) / math.sqrt(252)


    std = e.get("std", None)
    price = e.get("entry_price")

    if std is None or std <= 0:
        return 0

    # Convert daily price std → ticks
    std_ticks = std * price/ tick_size

    # Convert ticks → daily $ P&L volatility per contract
    dollar_vol_per_contract = std_ticks * tick_value

    # avoid division by zero
    if dollar_vol_per_contract <= 0:
        return 0

    # Volatility-based sizing
    pos = target_daily_vol / dollar_vol_per_contract

    # round down to whole contracts
    pos = int(max(0, math.floor(pos)))

    return pos

def calculate_atr(candles, period=20, mult=24):
    n = len(candles)
    if n <= period:
        return [None] * n

    tr = [0.0] * n
    atr = [None] * n

    # True Range
    for i in range(1, n):
        high = candles[i].high
        low = candles[i].low
        prev_close = candles[i-1].close

        tr[i] = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )

    # First ATR = SMA(TR)
    first_atr = sum(tr[1:period+1]) / period
    atr[period] = first_atr

    # Wilder smoothing
    alpha = 1 / period
    for i in range(period + 1, n):
        atr[i] = atr[i-1] * (1 - alpha) + tr[i] * alpha * math.sqrt(mult)

    return atr

def calculate_std(candles, period=20, mult=24):

    """
    Calculate rolling standard deviation of returns over 'period'.
    Returns a list aligned with candle indices.
    """

    n = len(candles)
    std_list = [0.0] * n
    returns = [0.0] * n

    # --- 1) Compute returns ---
    for i in range(1, n):
        prev_close = candles[i-1].close
        if prev_close != 0:
            returns[i] = (candles[i].close - prev_close) / prev_close

    # --- 2) Rolling std ---

    for i in range(period):
        std_list[i] = None  # not enough data

    for i in range(period, n):
        window = returns[i-period+1 : i+1]
        mean = sum(window) / period

        # sample variance (ddof=1)
        var = sum((x - mean) ** 2 for x in window) / (period - 1) if period > 1 else 0
        std_list[i] = math.sqrt(var) * math.sqrt(mult)

    return std_list

def run_strategy(dataset, all_candles, num, stop_type, tick_size, tick_value, costs, exit_type, title=f"6E 60min ", sm="6E", starting_capital=100000):
    print(f"Running strategy {title} {stop_type} stop loss on {len(all_candles)} candles.")
    print(f"Using {num} rolling windows.")

    short_ma = 40
    long_ma = 50

    avolume_profiles = []
    aentries = []

    results = []


    dt = str(all_candles[-1].time)[:10]


    # === Calculate Moving Averages ===
    ma_dict, cross_up, cross_down = calculate_moving_averages_and_crossovers(all_candles, short_ma, long_ma)
    print(f"Found {len(cross_up)} valid cross-ups and {len(cross_down)} valid cross-downs.")


    # 2️⃣ Calculate ATR and std once for the full candle series
    atr_series = calculate_atr(all_candles, period=20)
    std_series = calculate_std(all_candles, period=100)

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

        # if all_candles[idx_a].symbol != all_candles[idx_b].symbol: # we check this in detect_entries
        #     continue

        vp = calculate_volume_profile_from_trades_fast(dataset, A, B, symbol=all_candles[idx_a].symbol, bins=160)
        if vp is None:
            continue

        # if all_candles[idx_a].symbol != all_candles[idx_b].symbol:
        #     raise ValueError(f"Symbol mismatch: {all_candles[idx_a].symbol} != {all_candles[idx_b].symbol}")
        
        poc = calculate_poc(vp, tick_size)
        volume_profiles.append({"a_candle": all_candles[idx_a], "b_candle": all_candles[idx_b], "poc": poc, "B": B})
        avolume_profiles.append({"A": A, "B": B, "poc": poc})

    entries = detect_entries(all_candles, volume_profiles)



    aentries.extend(entries)

    for e in entries:
        idx = e["candle_index"]  # where the entry occurs
        e["atr"] = atr_series[idx]
        e["std"] = std_series[idx]

    # === Create Variants ===
    if stop_type == "atr": s = [0.25, 0.5]

    # [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
                                
    elif stop_type == "fixed": s = [10, 20]

    for stop in s:

        stop_losses = determine_stop_losses(stop_type, entries, stop, tick_size)

        for r in [0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20]:
            name = f"{stop_type}_{stop}_{r}"

            rrratios = [r] * len(entries)
            
            try:
                variant = Variant(entries, all_candles, stop_losses, rrratios, dt, name, num, sm)
                results.append(variant)
            except Exception as e:
                print(f"⚠️ Error creating Variant {name}: {e}")

    for v in results: v.compute(tick_size, tick_value, costs, starting_capital, exit_type)

    visualize_candles(
        all_candles,
        std_series,
        atr_series,
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
    # print_results_2(results, num)

def cronometer(func):

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        elapsed = end - start
        print(f"\n⏱️ Cronômetro: {elapsed:.4f} segundos")
        return result
    return wrapper

@cronometer
def main():
    # === Load data ===
    initial_date = int(sys.argv[1])
    last_date = int(sys.argv[2])
    p = sys.argv[3]
    dataset, days, first_date, last_date = load_raw_data(initial_date, last_date, path=p, max_files=None)


    freq = "60min" #If we change this we must change the std calculation
    exit_type = "EOD"

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
        tit = f"6E {freq} {exit_type} Candlestick Chart"
        sb = "6E"
        tick_size, tick_value, costs = 0.00005, 6.25, 3.1

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
        tit = f"M6E {freq} {exit_type} Candlestick Chart"
        sb = "M6E"
        tick_size, tick_value, costs = 0.00005, 0.625, 0.84

    if p == "dataMCL":
        symbols = ["MCLH2", "MCLM2", "MCLU2", "MCLZ2",
                "MCLH3", "MCLM3", "MCLU3", "MCLZ3",
                "MCLH4", "MCLM4", "MCLU4", "MCLZ4",
                "MCLH5", "MCLM5", "MCLU5", "MCLZ5"]

        roll_schedule = {
            "MCLH2": ("2021-12-11", "2022-03-12"),
            "MCLM2": ("2022-03-12", "2022-06-11"),
            "MCLU2": ("2022-06-11", "2022-09-17"),
            "MCLZ2": ("2022-09-17", "2022-12-17"),

            "MCLH3": ("2022-12-17", "2023-03-11"),
            "MCLM3": ("2023-03-11", "2023-06-17"),
            "MCLU3": ("2023-06-17", "2023-09-16"),
            "MCLZ3": ("2023-09-16", "2023-12-16"),

            "MCLH4": ("2023-12-16", "2024-03-16"),
            "MCLM4": ("2024-03-16", "2024-06-15"),
            "MCLU4": ("2024-06-15", "2024-09-14"),
            "MCLZ4": ("2024-09-14", "2024-12-14"),

            "MCLH5": ("2024-12-14", "2025-03-15"),
            "MCLM5": ("2025-03-15", "2025-06-14"),
            "MCLU5": ("2025-06-14", "2025-09-13"),
            "MCLZ5": ("2025-09-13", "2025-12-13"),
        }

        tick_size, tick_value, costs = 0.01, 1, 1.1
        tit = f"MCL {freq} {exit_type} Candlestick Chart"
        sb = "MCL"

    # === Build all candles ===
    all_candles = []
    for s in symbols:
        for t in make_candles(dataset, freq, symbol=s, roll_schedule=roll_schedule):
            all_candles.append(t)

    # stop_type = "fixed"
    stop_type = "atr"

    rollings = 12

    # === Split data (fit/test) ===
    run_strategy(dataset, all_candles, rollings, stop_type, tick_size, tick_value, costs, exit_type ,sm=sb, title=tit)

    # === Save data ===
    write_output_to_file(tit)

if __name__ == "__main__":
    main()
    

'''
Todo

Resolver situacao dos simbolos                                       OK
Calcular Moving Average Crossings                                    OK
Calcular Volume Profile                                              OK
Plotar POC                                                           OK
Add entry signals                                                    OK
Fix decide_entry_direction                                           OK
Calculate result from each entrie                                    OK
Fix symbol with duplicate entries                                    OK
Fix sharpe calculation                                               OK
Add Costs                                                            OK
Add Slippage                                                         OK
Plot Entries stop and tp                                             OK
Fix POC step (0.0005)                                                OK
Fix moving Average  Mistake                                          OK
Make Moving Average Fix Clean                                        OK
Fix Sharpe Calculation                                               OK
Review Trade Closing                                                 OK
Review for other possible mistakes                                   OK
Limit trades to same contract as vp                                  OK
Add Metric Average Trade Duration                                    -
Add Metric Avg Win                                                   OK
Add Metric Avg Loss                                                  OK
Add Metric Number of days                                            OK
Fix metrics Total Trades                                             OK
Control Trade Duration                                               OK
Optimize load_data function
Optimizacao: fazer compute_data e return from trades para todas variantes ao mesmo tempo
Optimizar memoria
Otimizacao: detect entries lopando em candles apartir do B
Optimize volume profile function                                     OK
Save output                                                          OK
Generate p&l graph function                                         
Week profit visualization
Generate CI per rolling graph
Show Rolling Division on visualization
Add 1 year, 3 months, 1 month returns
Fit and Test separated (out of sample)                               OK
Fit and Test rolling out of sample                                   OK
Add final date                                                       OK
Returns in %                                                         OK
Update Capital with trade results                                    OK
In Sample Permutation Test
Permutate_candles function
Position Sizing with volatility standardization                      OK
Ploting Standard Deviation                                           OK
Stop size based on ATR                                               OK
Fix instrument specifics(tick size, tick value, costs)               OK
Fix volality in % not being handled                                  OK
Handle 0 contracts situations                                        OK
Test keeping position open
Fix Contract Switch to match tradingview
Trades bleeding across Contracts Add Switch                          OK
Volatility scaling: you’re treating 1-hour STD as daily STD          OK
Using Daily ATR                                                      OK
Compute std on daily candles
Remove Zero-return days?
Avg Win / Avg Loss are wrong                                         OK
Fix Compute Data                                                     OK
Tralling Stop instead of take profit
Position Sizing with Forecast Value
Plot metrics x profit factor
Combine M6E, MCL, etc daily returns to see the metrics togheter
'''
