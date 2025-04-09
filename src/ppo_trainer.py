from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from custom_trading_env import CustomTradingEnv
import pandas as pd


def compute_rsi(close_prices, window=14):
    """
    Compute the Relative Strength Index (RSI) for a given series of close prices.

    Args:
        close_prices (pd.Series): Series of close prices.
        window (int): Lookback window for RSI calculation (default is 14).

    Returns:
        pd.Series: RSI values.
    """
    # Calculate price changes
    delta = close_prices.diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)  # Positive changes (gains)
    loss = -delta.where(delta < 0, 0)  # Negative changes (losses)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / (avg_loss + 1e-8)  # Add a small value to avoid division by zero

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi



# Load historical data
symbol = "AAPL"
df = pd.read_csv(f"data/{symbol}.csv")

# Preprocess the data
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime
# df = df.set_index('timestamp').resample('min').ffill().reset_index()  # Fill missing timestamps
# df = df.sort_values(by='timestamp')  # Sort data by timestamp
df = df[['open', 'high', 'low', 'close', 'volume']]  # Ensure correct column order
df = df.astype(float)  # Ensure all remaining columns are numeric

# Add technical indicators
df['ma_10'] = df['close'].rolling(window=10).mean()
df['rsi'] = compute_rsi(df['close'])  # Implement compute_rsi function

# Handle NaN values
df.loc[:, 'rsi'] = test_df['rsi'].fillna(50)  # Fill NaN RSI values with 50
df.dropna(inplace=True)  # Drop rows with any remaining NaN values

# Initialize the environment
env = CustomTradingEnv(df=df)

# Wrap the environment for vectorized training
vec_env = make_vec_env(lambda: env, n_envs=1)

# Normalize observations and rewards
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Train an RL agent
model = PPO(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=0.0001,  # Increased learning rate
    gamma=0.95,  # Discount factor
    n_steps=2048,  # Smaller batch of steps
    batch_size=512,  # Batch size for updates
    ent_coef=0.01,  # Encourage exploration
    clip_range=0.2  # Increased clipping range
)

# Train the model
model.learn(total_timesteps=500_000)

# Save the trained model
model.save(f"models/ppo_model_{symbol}")