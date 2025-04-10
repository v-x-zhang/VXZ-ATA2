# Using the data_handler.py module
# Download at least several months (or years) of minute-level data for the 
# stocks/ETFs you plan to trade.

# Store them as CSV or in a local database (e.g. SQLite) for backtesting and 
# offline RL training.

# Deliverable: A local dataset of historical bar data you can reference for simulations.

import os
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from data_handler import DataHandler

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Alpaca API keys
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Initialize DataHandler
data_handler = DataHandler(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY)

def fetch_and_store_data(symbols, start_date, end_date, timeframe="1Minute", output_format="csv", limit=10000):
    """
    Fetch historical data for the given symbols and store them locally.
    
    Args:
        symbols (list): List of stock/ETF symbols to fetch data for.
        start_date (datetime): Start date for historical data.
        end_date (datetime): End date for historical data.
        timeframe (str): Timeframe for the data (e.g., "1Minute", "1Hour").
        output_format (str): Output format ("csv" or "sqlite").
    """
    # Adjust the end date to exclude the test period
    adjusted_end_date = end_date - timedelta(days=7)  # Exclude the last week for api (for some reason)

    project_directory = os.path.dirname(os.path.dirname(__file__))  # Get PROJECTDIRECTORY
    data_directory = os.path.join(project_directory, "data")  # Define "PROJECTDIRECTORY/data"

    for symbol in symbols:
        print(f"Fetching data for {symbol}...")

        # Initialize rolling window variables
        current_start_date = start_date
        all_data = pd.DataFrame()

        while current_start_date < adjusted_end_date:
            # Determine the current end date for the fetch
            current_end_date = current_start_date + timedelta(days=7)  # Fetch 7 days at a time
            if current_end_date > adjusted_end_date:
                current_end_date = adjusted_end_date  # Limit to the adjusted end date

            print(f"Fetching data from {current_start_date} to {current_end_date}...")
            data = data_handler.fetch_historical_data(
                symbol=symbol,
                start_date=current_start_date,
                end_date=current_end_date,
                timeframe=timeframe
            )

            if data.empty:
                print(f"No data found for {symbol} from {current_start_date} to {current_end_date}.")
                break

            # Append the fetched data to the main DataFrame
            all_data = pd.concat([all_data, data], ignore_index=True)

            # Update the rolling window start date
            current_start_date = pd.to_datetime(data['timestamp'].iloc[-1]) + timedelta(minutes=1)

        # Save the combined data to a CSV file
        output_file = os.path.join(data_directory, f"{symbol}.csv")
        all_data.to_csv(output_file, index=False)
        print(f"Data for {symbol} saved to {output_file}.")

if __name__ == "__main__":
    # Define the symbols and date range
    symbols = ["AAPL"]  # Add the stocks/ETFs you plan to trade
    start_date = datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(days=548)  # Last 18 months  Last two weeks for testing models
    end_date = datetime.now(ZoneInfo("America/Los_Angeles"))
    timeframe = "1Minute"  # Minute-level data
    output_format = "csv"

    # Alpaca API limits the number of data points per call (e.g., 10000 rows)
    max_data_points = 10000

    # Initialize the data directory
    project_directory = os.path.dirname(os.path.dirname(__file__))
    data_directory = os.path.join(project_directory, "data")
    os.makedirs(data_directory, exist_ok=True)

    fetch_and_store_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        output_format=output_format,
        limit=max_data_points
    )

    


# if __name__ == "__main__":
#     # Define the symbols and date range
#     symbols = ["AAPL"]  # Add the stocks/ETFs you plan to trade
#     start_date = datetime.now(ZoneInfo("America/Los_Angeles")) - timedelta(days=180)  # Last 6 months
#     end_date = datetime.now(ZoneInfo("America/Los_Angeles"))

#     # Fetch and store data
#     fetch_and_store_data(
#         symbols=symbols,
#         start_date=start_date,
#         end_date=end_date,
#         timeframe="1Minute",  # Minute-level data
#         output_format="csv",   # Change to "sqlite" for database storage
#         limit=10000
#     )