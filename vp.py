import pandas as pd
from main import load_raw_data, make_candles, separate_data
import numpy as np
import sys

def main():
    # === Load data ===
    initial_date = int(sys.argv[1])
    last_date = int(sys.argv[2])
    p = sys.argv[3]
    dataset, days, first_date, last_date = load_raw_data(initial_date, last_date, path=p, max_files=None)
    
    print(dataset)

    freq = ""

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
        tit = f"M6E {freq} Candlestick Chart"
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
        tit = f"MCL {freq} Candlestick Chart"

if __name__ == "__main__":
    main()