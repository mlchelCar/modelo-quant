import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import load_raw_data, make_candles


def visualize_candles(candles, t="Candlestick Chart"):
    times = [pd.Timestamp(c.time).tz_localize(None) for c in candles]
    opens, highs, lows, closes, volumes = zip(*[(c.open, c.high, c.low, c.close, c.volume) for c in candles])
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(candles))]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    fig.add_trace(go.Candlestick(x=times, open=opens, high=highs, low=lows, close=closes, name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=times, y=volumes, name='Volume', marker_color=colors), row=2, col=1)
    fig.update_layout(title=t, xaxis_rangeslider_visible=False, height=800, hovermode='x unified', xaxis=dict(type='date'))
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.show()


if __name__ == "__main__":
    # Load last 60 days of data (faster rendering, still plenty of data)
    # Change max_files to load more/less data

    dataset = load_raw_data(20241023, max_files=None)

    symbols = ["6EH4", "6EM4", "6EU4", "6EZ4","6EH5", "6EM5", "6EU5", "6EZ5"]

    freq="25min"
    for s in symbols:
        candles = make_candles(dataset, freq, symbol=s)
        
        if not candles:
            print(f"No candles for {s}")
            continue

        visualize_candles(candles, t=f"{s} {freq} Candlestick Chart")



'''
Todo

Resolver situacao dos simbolos OK
calcular Moving Average Crossings
Calcular Volume Profile
Calcular POC e trade levels
'''