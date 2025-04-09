#import necessary libraries
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.historical.corporate_actions import CorporateActionsClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.stream import TradingStream
from alpaca.data.live.stock import StockDataStream

from alpaca.data.requests import (
    CorporateActionsRequest,
    StockBarsRequest,
    StockQuotesRequest,
    StockTradesRequest,
)
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetAssetsRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLimitOrderRequest,
    StopLossRequest,
    StopOrderRequest,
    TakeProfitRequest,
    TrailingStopOrderRequest,
)
from alpaca.trading.enums import (
    AssetExchange,
    AssetStatus,
    OrderClass,
    OrderSide,
    OrderType,
    QueryOrderStatus,
    TimeInForce,
)

#Get API keys
load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

#Authenticate and instantiate Trading Client. paper=True enables paper trading
trade_client = TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=True)

#Fetch account info (balance, portfolio value).
# account = trade_client.get_account()

# print(f"Account Balance: {account.buying_power}")
# print(f"Portfolio Value: {account.portfolio_value}")

symbol = "AAPL"

#Retrieve current market data for a test symbol (e.g., AAPL).
# stock_historical_data_client = StockHistoricalDataClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)
# # Get the last 5 minutes of AAPL data
# now = datetime.now(ZoneInfo("America/Los_Angeles"))
# req = StockBarsRequest(
#     symbol_or_symbols = [symbol],
#     timeframe=TimeFrame(amount = 1, unit = TimeFrameUnit.Hour), # specify timeframe
#     start = now - timedelta(days = 5),                          # specify start datetime, default=the beginning of the current day.
#     # end_date=None,                                        # specify end datetime, default=now
#     limit = 2,                                               # specify limit
# )

# bars = stock_historical_data_client.get_stock_bars(req).df
# print(bars)
# print(bars.columns)

# Place a test buy/sell order in paper mode.

# If you have an error of `qty must be integer`, please try to `Reset Account` of your paper account via the Alpaca Trading API dashboard
# Place a market order to buy 5.5 shares of AAPL
# req = MarketOrderRequest(
#     symbol = symbol,
#     qty = 5.5,
#     side = OrderSide.BUY,
#     type = OrderType.MARKET,
#     time_in_force = TimeInForce.DAY,
# )
# res = trade_client.submit_order(req)
# res

# get a list of orders including closed (e.g. filled) orders by specifying symbol
req = GetOrdersRequest(
    status = QueryOrderStatus.ALL,
    symbols = [symbol]
)
orders = trade_client.get_orders(req)
print(orders)

# cancel all open orders
# trade_client.cancel_orders()

# Market data stream example
# Fetch market data stream for a test symbol (e.g., AAPL).
# stock_data_stream_client = StockDataStream(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# async def stock_data_stream_handler(data):
#     print(data)

# symbols = [symbol]

# stock_data_stream_client.subscribe_quotes(stock_data_stream_handler, *symbols)
# stock_data_stream_client.subscribe_trades(stock_data_stream_handler, *symbols)

# stock_data_stream_client.run()