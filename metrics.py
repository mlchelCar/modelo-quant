import numpy as np
import pandas as pd
from scipy.stats import bootstrap


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
    d = np.sum(returns != 0)

    if len(returns) == 0 or d == 0: return 0.0
    wins = np.sum(returns > 0)

    return (wins / d) * 100

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

def calculate_sortino(returns, risk_free=0.0):
    """
    Sortino Ratio: reward relative to downside volatility.
    returns: list of daily returns (floats)
    risk_free: daily risk-free rate (default 0)
    """

    if len(returns) == 0:
        return 0.0

    downside = [r - risk_free for r in returns if r < risk_free]

    if len(downside) == 0:
        return float("inf")   # no downside => perfect system

    downside_std = np.std(downside, ddof=1)

    if downside_std == 0:
        return float("inf")

    mean_excess = np.mean(returns) - risk_free

    return mean_excess / downside_std

def calculate_max_drawdown(returns):
    """
    Max Drawdown using compound equity curve.
    returns: list of daily returns (floats)
    """

    if len(returns) == 0:
        return 0.0

    equity = np.cumprod(1 + np.array(returns))      # correct compounding
    highwater = np.maximum.accumulate(equity)
    dd = (highwater - equity) / highwater           # normalized drawdown
    max_dd = float(np.max(dd))

    return max_dd

def calculate_cagr(returns):
    """
    CAGR based on daily returns.
    returns: list of daily returns (floats)
    """

    t = len(returns)
    if t == 0:
        return 0.0

    years = t / 252

    equity = np.cumprod(1 + np.array(returns))
    final_equity = equity[-1]

    if years <= 0 or final_equity <= 0:
        return 0.0

    cagr = final_equity ** (1 / years) - 1
    return float(cagr)

def calculate_calmar(returns):
    """
    Calmar Ratio = CAGR / Max Drawdown.
    returns: list of daily returns (floats)
    """

    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)

    if max_dd == 0:
        return float("inf")

    return float(cagr / max_dd)

def compute_data(l, dates, n, trade_list):
    """
    Compute rolling FIT/TEST metrics for strategy evaluation.

    Parameters
    ----------
    l : list[float]
        daily returns (oldest -> newest)
    dates : list[datetime.date]
        matching dates for each daily return (oldest -> newest)
    n : int
        number of rolling segments
    trade_list : list[(exit_time, pnl)]
        list of PnL with exit timestamps

    Returns
    -------
    results : list of [fit_metrics_dict, test_metrics_dict]
    """

    # ---------------------------------------------------------
    # 1) Immutable copies
    # ---------------------------------------------------------
    daily_returns = list(l)
    day_list = list(dates)
    t = len(daily_returns)

    if t == 0:
        return []

    # ---------------------------------------------------------
    # 2) Split into n contiguous groups using the REAL dates
    # ---------------------------------------------------------
    base = t // n
    extra = t % n

    groups_returns = []
    groups_days = []
    pos = 0

    for i in range(n):
        size = base + (1 if i < extra else 0)
        groups_returns.append(daily_returns[pos:pos+size])
        groups_days.append(day_list[pos:pos+size])
        pos += size

    # ---------------------------------------------------------
    # 3) Build rolling FIT/TEST splits
    # ---------------------------------------------------------
    segments = []
    if n == 1:
        segments = [(daily_returns, day_list, [], [])]
    else:
        for i in range(n-1):
            fit_r = [x for g in groups_returns[:i+1] for x in g]
            fit_d = [x for g in groups_days[:i+1] for x in g]
            test_r = [x for g in groups_returns[i+1:] for x in g]
            test_d = [x for g in groups_days[i+1:] for x in g]
            segments.append((fit_r, fit_d, test_r, test_d))

    # ---------------------------------------------------------
    # 4) Normalize trade timestamps
    # ---------------------------------------------------------
    normalized_trades = []
    for exit_time, pnl in trade_list:
        try:
            ts = pd.to_datetime(exit_time)
            normalized_trades.append((ts, float(pnl)))
        except Exception:
            continue

    # ---------------------------------------------------------
    # 5) Compute metrics per segment
    # ---------------------------------------------------------
    results = []

    for (fit_r, fit_d, test_r, test_d) in segments:
        pair = []

        # FIT then TEST
        for seg_returns, seg_days in ((fit_r, fit_d), (test_r, test_d)):

            if not seg_returns:
                pair.append({})
                continue

            # Real start/end date
            start_day = seg_days[0]
            end_day   = seg_days[-1]

            # Filter trades by REAL date window
            trades_in_segment = [
                pnl for (ts, pnl) in normalized_trades
                if start_day <= ts.date() <= end_day
            ]

            # Sharpe + CI
            sharpe, annualized_sharpe, ci = calculate_sharpe(seg_returns)
            sortino = calculate_sortino(seg_returns)
            max_dd = calculate_max_drawdown(seg_returns)
            cagr = calculate_cagr(seg_returns)
            calmar = calculate_calmar(seg_returns)

            # ---------------------------------------------------------
            # Trade stats
            # ---------------------------------------------------------
            wins = [x for x in trades_in_segment if x > 0]
            losses = [x for x in trades_in_segment if x < 0]

            sum_w = sum(wins)
            sum_l = abs(sum(losses))
            winrate = len(wins) / len(trades_in_segment) if trades_in_segment else 0.0
            avg_w = sum_w / len(wins) if wins else 0.0
            avg_l = sum_l / len(losses) if losses else 0.0

            if sum_l == 0:
                profit_factor = float("inf") if sum_w > 0 else 0.0
            else:
                profit_factor = sum_w / sum_l

            expectancy = np.mean(trades_in_segment) if trades_in_segment else 0.0

            # ---------------------------------------------------------
            # Store metrics
            # ---------------------------------------------------------
            pair.append({
                "Trade Returns": trades_in_segment,
                "Daily Returns": seg_returns,
                "Start Date": start_day,
                "End Date": end_day,

                "Sharpe Ratio": round(sharpe, 3),
                "Annualized Sharpe": round(annualized_sharpe, 3),
                "Sharpe 95% CI": (
                    None if np.isnan(ci[0]) else round(ci[0], 3),
                    None if np.isnan(ci[1]) else round(ci[1], 3)
                ),
                "Sortino Ratio": float("inf") if not np.isfinite(sortino) else round(sortino, 3),
                "CAGR": float("inf") if not np.isfinite(cagr) else round(cagr, 3),
                "Calmar Ratio": float("inf") if not np.isfinite(calmar) else round(calmar, 3),

                "Daily Win Rate (%)": round(winrate, 3),
                "Max Drawdown": round(max_dd, 3),
                "Profit Factor": (
                    float("inf") if not np.isfinite(profit_factor)
                    else round(profit_factor, 3)
                ),
                "Expectancy": round(expectancy, 6),

                "Average Win": round(avg_w*100, 6),
                "Average Loss": round(avg_l*100, 6),

                "Total Trades": len(trades_in_segment),
                "Total Days": len(seg_returns),
            })

        results.append(pair)

    return results