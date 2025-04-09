# Tasks:
#     Implement a data_handler.py module/class to unify data retrieval:
#         Historical data fetch using alpaca.get_bars().
#         Real-time streaming callbacks for incremental updates.
#     Transform raw data into standard formats (e.g. Pandas DataFrames with columns like 
#     open, high, low, close, volume).

# Deliverable: A Pythonic interface (class or set of functions) that provides a consistent 
# data feed for both backtesting (offline) and live trading (online).

import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

class DataHandler:
    def __init__(self, api_key, secret_key):
        # Initialize Alpaca clients
        self.historical_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        self.stream_client = StockDataStream(api_key, secret_key)

    def fetch_historical_data(self, symbol, start_date, end_date=None, timeframe="1Hour", limit=10000):
        """
        Fetch historical data for a given symbol and timeframe.
        Returns a Pandas DataFrame with columns: open, high, low, close, volume.
        """
        # Parse timeframe
        import re
        match = re.match(r"(\d+)([a-zA-Z]+)", timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
        amount, unit = int(match.group(1)), match.group(2)

        unit_enum= {
            "Minute": TimeFrameUnit.Minute,
            "Hour": TimeFrameUnit.Hour,
            "Day": TimeFrameUnit.Day,
            "Week": TimeFrameUnit.Week,
            "Month": TimeFrameUnit.Month
        }

        unit_enum = unit_enum.get(unit, None)
        if unit_enum is None:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
        # unit_enum = getattr(TimeFrameUnit, unit, None)
        # if unit_enum is None:
        #     raise ValueError(f"Unsupported timeframe unit: {unit}")

        # Create request
        req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame(amount=amount, unit=unit_enum),
            start=start_date,
            # end=end_date,
            limit=limit
        )

        # Fetch data
        bars = self.historical_client.get_stock_bars(req).df

        # Transform data into a consistent format
        if symbol in bars.index.get_level_values(0):
            bars = bars.loc[symbol]
            bars.reset_index(inplace=True)  # Reset index to include the timestamp column
            bars = bars[["timestamp", "open", "high", "low", "close", "volume"]]  # Include timestamp
        else:
            bars = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        return bars

    def subscribe_realtime_data(self, symbols, callback):
        """
        Subscribe to real-time data for the given symbols.
        The callback function will handle incoming data.
        """
        async def data_handler(data):
            # Transform data into a consistent format and pass to callback
            transformed_data = {
                "symbol": data.symbol,
                "price": getattr(data, "price", None),
                "bid_price": getattr(data, "bid_price", None),
                "ask_price": getattr(data, "ask_price", None),
                "volume": getattr(data, "volume", None),
                "timestamp": data.timestamp
            }
            callback(transformed_data)

        # Subscribe to quotes and trades
        self.stream_client.subscribe_quotes(data_handler, *symbols)
        self.stream_client.subscribe_trades(data_handler, *symbols)

    def run_stream(self):
        """
        Run the real-time data stream.
        """
        self.stream_client.run()