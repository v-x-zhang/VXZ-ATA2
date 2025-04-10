import praw
import os
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Twitter API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_SECRET_KEY = os.getenv("REDDIT_SECRET_KEY")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_SECRET_KEY,
    user_agent=REDDIT_SECRET_KEY
)

def fetch_reddit_posts_with_timestamps(subreddit, query, start_date, end_date, chunk_size_days=30, limit=100):
    """
    Fetch Reddit posts and their timestamps based on a query and timeframe.

    Args:
        subreddit (str): Subreddit to search in (e.g., "stocks").
        query (str): Search query (e.g., "AAPL").
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        chunk_size_days (int): Number of days per query chunk.
        limit (int): Number of posts to fetch per query.

    Returns:
        list: List of dictionaries with 'text' and 'timestamp' keys.
    """
    from datetime import datetime, timedelta

    # Convert start and end dates to datetime objects
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    posts = []
    current_start = start_date

    while current_start < end_date:
        # Define the current chunk's end date
        current_end = current_start + timedelta(days=chunk_size_days)
        if current_end > end_date:
            current_end = end_date

        # Convert to Unix timestamps
        after = int(current_start.timestamp())
        before = int(current_end.timestamp())

        print(f"Fetching posts from {current_start} to {current_end}...")

        # Query Reddit
        subreddit_obj = reddit.subreddit(subreddit)
        for submission in subreddit_obj.search(query, sort="new", limit=limit, params={"after": after, "before": before}):
            posts.append({"text": submission.title, "timestamp": submission.created_utc})
            submission.comments.replace_more(limit=100)  # Fetch all comments
            for comment in submission.comments.list():
                posts.append({"text": comment.body, "timestamp": comment.created_utc})

        # Move to the next chunk
        current_start = current_end

    return posts

# Load FinBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

def analyze_sentiment_with_timestamps(posts, batch_size=32):
    """
    Analyze sentiment using FinBERT in batches and include timestamps.

    Args:
        posts (list): List of dictionaries with 'text' and 'timestamp' keys.
        batch_size (int): Number of posts to process in each batch.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' and 'sentiment_score' columns.
    """
    texts = [post["text"] for post in posts]
    timestamps = [post["timestamp"] for post in posts]

    sentiment_scores = []

    # Process in batches with a progress bar
    print("Analyzing sentiment...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis Progress"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            batch_scores = predictions[:, 2] - predictions[:, 0]  # Positive - Negative
            sentiment_scores.extend(batch_scores.cpu().numpy())  # Move to CPU and store

    # Create a DataFrame with timestamps and sentiment scores
    sentiment_df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s"),
        "sentiment_score": sentiment_scores
    })

    return sentiment_df

def aggregate_sentiment_to_trading_data(sentiment_df, trading_df, batch_size=1000):
    """
    Align sentiment data with trading data by aggregating sentiment scores into daily intervals.

    Args:
        sentiment_df (pd.DataFrame): DataFrame with 'timestamp' and 'sentiment_score'.
        trading_df (pd.DataFrame): Trading data with 'timestamp'.
        batch_size (int): Number of rows to process in each batch.

    Returns:
        pd.DataFrame: Trading data with an additional 'sentiment_score' column.
    """
    # Set the timestamp as the index for both DataFrames
    sentiment_df.set_index("timestamp", inplace=True)
    trading_df.set_index("timestamp", inplace=True)

    # Resample sentiment data to daily intervals and calculate the mean sentiment score
    sentiment_resampled = sentiment_df.resample('D').mean()  # Daily intervals

    # Initialize an empty DataFrame for the result
    result_df = pd.DataFrame()

    # Process trading data in batches with a progress bar
    print("Aggregating sentiment data...")
    for i in tqdm(range(0, len(trading_df), batch_size), desc="Aggregation Progress"):
        batch = trading_df.iloc[i:i + batch_size]
        batch["sentiment_score"] = sentiment_resampled["sentiment_score"]
        batch["sentiment_score"].fillna(method="ffill", inplace=True)  # Forward-fill missing sentiment scores
        result_df = pd.concat([result_df, batch])

    # Reset the index
    result_df.reset_index(inplace=True)

    return result_df

def preprocess_sentiment_data(subreddit, query, trading_df, output_file="DEFAULT_with_sentiment.csv", batch_size=32):
    """
    Preprocess sentiment data by fetching, analyzing, and aligning it with trading data.

    Args:
        subreddit (str): Subreddit to fetch posts from (e.g., "stocks").
        query (str): Search query (e.g., "AAPL").
        trading_df (pd.DataFrame): Trading data with 'timestamp' column.
        output_file (str): File to save the processed sentiment data.
        batch_size (int): Number of posts to process in each batch.

    Returns:
        pd.DataFrame: Trading data with an additional 'sentiment_score' column.
    """
    # Fetch Reddit posts for the entire time range
    start_date = trading_df['timestamp'].min().strftime('%Y-%m-%d')
    end_date = trading_df['timestamp'].max().strftime('%Y-%m-%d')
    print(f"Fetching Reddit posts from {start_date} to {end_date}...")

    reddit_posts = fetch_reddit_posts_with_timestamps(
        subreddit=subreddit,
        query=query,
        start_date=start_date,
        end_date=end_date,
        chunk_size_days=30,  # Fetch posts in 30-day chunks
        limit=100  # Adjust limit as needed
    )

    # Analyze sentiment in batches
    sentiment_df = analyze_sentiment_with_timestamps(reddit_posts, batch_size=batch_size)

    # Align sentiment with trading data (aggregate by day)
    trading_df = aggregate_sentiment_to_trading_data(sentiment_df, trading_df)

    # Save the processed data
    print(f"Saving processed sentiment data to {output_file}...")
    trading_df.to_csv(output_file, index=False)

    return trading_df

# if __name__ == "__main__":
#     # Load trading data
#     trading_data_file = "data/AAPL.csv"
#     trading_df = pd.read_csv(trading_data_file)
#     trading_df['timestamp'] = pd.to_datetime(trading_df['timestamp'])

#     # Preprocess sentiment data
#     processed_data = preprocess_sentiment_data(
#         subreddit="stocks",
#         query="AAPL",
#         trading_df=trading_df,
#         output_file="data/AAPL_with_sentiment.csv"
#     )

def preprocess_sentiment_data_for_month(subreddit, query, trading_df, start_date, end_date, output_file="DEFAULT_with_sentiment.csv", batch_size=32):
    """
    Preprocess sentiment data for a specific month by fetching, analyzing, and aligning it with trading data.

    Args:
        subreddit (str): Subreddit to fetch posts from (e.g., "stocks").
        query (str): Search query (e.g., "AAPL").
        trading_df (pd.DataFrame): Trading data with 'timestamp' column.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        output_file (str): File to save the processed sentiment data.
        batch_size (int): Number of posts to process in each batch.

    Returns:
        pd.DataFrame: Trading data with an additional 'sentiment_score' column.
    """
    print(f"Fetching Reddit posts from {start_date} to {end_date}...")

    # Fetch Reddit posts for the specified time range
    reddit_posts = fetch_reddit_posts_with_timestamps(
        subreddit=subreddit,
        query=query,
        start_date=start_date,
        end_date=end_date,
        chunk_size_days=7,  # Fetch posts in 7-day chunks
        limit=10  # Adjust limit as needed
    )

    # Analyze sentiment in batches
    sentiment_df = analyze_sentiment_with_timestamps(reddit_posts, batch_size=batch_size)

    # Filter trading data for the specified month
    trading_df = trading_df[(trading_df['timestamp'] >= start_date) & (trading_df['timestamp'] <= end_date)]

    # Align sentiment with trading data (aggregate by day)
    trading_df = aggregate_sentiment_to_trading_data(sentiment_df, trading_df)

    # Save the processed data
    print(f"Saving processed sentiment data to {output_file}...")
    trading_df.to_csv(output_file, index=False)

    return trading_df

if __name__ == "__main__":
    # Load trading data
    trading_data_file = "data/AAPL.csv"
    trading_df = pd.read_csv(trading_data_file)
    trading_df['timestamp'] = pd.to_datetime(trading_df['timestamp'])

    # Define the one-month timeframe
    start_date = "2023-09-01"  # Start of the month
    end_date = "2023-09-30"    # End of the month

    # Preprocess sentiment data for one month
    processed_data = preprocess_sentiment_data_for_month(
        subreddit="stocks",
        query="AAPL",
        trading_df=trading_df,
        start_date=start_date,
        end_date=end_date,
        output_file="data/AAPL_with_sentiment_1_month.csv"
    )