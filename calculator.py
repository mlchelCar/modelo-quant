import sys
import math

def determine_stop_losses(atr, tick_size, v= 0.25):
        return round(v * math.sqrt(24) * atr / tick_size)

def determine_contracts_volatility(std, tick_size, tick_value, last_price, capital=1000, target_vol=1):

    # Convert annual target vol → daily target dollar P&L volatility
    target_daily_vol = (capital * target_vol) / math.sqrt(252)

    if std is None or std <= 0:
        return 0        

    price_std = std * last_price * math.sqrt(24)
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
    print(pos)
    pos = int(max(0, math.floor(pos)))

    return pos

k = 0.25
tick_size, tick_value = 0.0001, 1.25
atr = float(sys.argv[1])
std = float(sys.argv[2])
last_price = float(sys.argv[3])
capital = float(sys.argv[4])
target_vol = float(sys.argv[5])

print("Usage: python calculator.py <atr> <std> <last_price> <capital> <target_vol>")
print(f"ATR: {atr}")
print(f"STD: {std}")

print("============")
print(f"Contracts: {determine_contracts_volatility(std, tick_size, tick_value, last_price, capital, target_vol)}")
print(f"Stop {determine_stop_losses(atr, tick_size, v = k)}")