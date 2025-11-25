def determine_stop_losses(atr, tick_size, v= 0.25):
        return round(v * atr / tick_size)

def determine_contracts_volatility(std, tick_size, tick_value, last_price, capital=100000, target_vol=0.10):

    # Convert annual target vol → daily target dollar P&L volatility
    target_daily_vol = (capital * target_vol) / math.sqrt(252)

    if std is None or std <= 0:
        return 0        

    price_std = std * last_price
    # Convert daily price std → ticks
    std_ticks = price_std  / tick_size

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

import sys
import math

k = 0.25
tick_size = 0.01
tick_value = 1
atr = float(sys.argv[1])
std = float(sys.argv[2])
last_price = float(sys.argv[3])

print(f"ATR: {atr}")
print(f"STD: {std}")

print("============")
print(f"Contracts: {determine_contracts_volatility(std, tick_size, tick_value, last_price)}")
print(f"Stop {determine_stop_losses(atr, tick_size, v = k)}")