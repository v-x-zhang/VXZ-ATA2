from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from custom_trading_env import CustomTradingEnv
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import os

class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging additional metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log custom metrics
        infos = self.locals.get("infos", [])
        if infos:
            # Log metrics from the last step
            last_info = infos[-1]  # Get the most recent info dictionary
            if "sharpe_ratio" in last_info:
                self.logger.record("custom/sharpe_ratio", last_info["sharpe_ratio"])
            if "step_return" in last_info:
                self.logger.record("custom/step_return", last_info["step_return"])
            if "drawdown_penalty" in last_info:
                self.logger.record("custom/drawdown_penalty", last_info["drawdown_penalty"])
            if "trade_penalty" in last_info:
                self.logger.record("custom/trade_penalty", last_info["trade_penalty"])

        # Example: Log the total reward
        total_reward = self.locals.get("rewards", 0)
        self.logger.record("custom/total_reward", total_reward)

        return True

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
df = pd.read_csv(f"data/{symbol}_with_sentiment.csv")

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
df.loc[:, 'rsi'] = df['rsi'].fillna(50)  # Fill NaN RSI values with 50
df.dropna(inplace=True)  # Drop rows with any remaining NaN values

env_count = 4  # Number of environments to run in parallel

# Initialize the environment
env = CustomTradingEnv(df=df)

# # Wrap the environment for vectorized training
vec_env = make_vec_env(lambda: env, n_envs=1)

"""
4 Environments in parallel
"""
# # Define a function to create a new instance of the environment
# def make_env(env_id, df):
#     def _init():
#         return CustomTradingEnv(df=df)
#     return _init
# # Create a list of environment creation functions
# env_count = 4  # Number of environments to run in parallel
# env_fns = [make_env(i, df) for i in range(env_count)]
# # Use SubprocVecEnv for parallel environments
# vec_env = SubprocVecEnv(env_fns)


# Normalize observations and rewards
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# Define a directory for TensorBoard logs
tensorboard_log_dir = "tensorboard_logs/"

# Initialize the PPO model with TensorBoard logging enabled
# model = PPO(
#     "MlpPolicy",
#     vec_env,
#     verbose=1,
#     learning_rate=1e-4,  # Decreased learning rate
#     ent_coef=0.4,  # Encourage exploration
#     gamma=0.99,  # Discount factor
#     n_steps=4096 // env_count,  # Smaller batch of steps
#     batch_size=128,  # Batch size for updates
#     clip_range=0.2,  # Increased clipping range
#     tensorboard_log=tensorboard_log_dir  # Enable TensorBoard logging
# )

# SAC Model
model = SAC(
    "MlpPolicy",
    vec_env,
    verbose=1,
    learning_rate=1e-4,        # Adjust as needed
    ent_coef='auto',           # Commonly used in SAC for automatic entropy tuning
    gamma=0.99,                # Discount factor
    buffer_size=100000,        # Replay buffer size
    learning_starts=1000,      # Steps before training begins
    batch_size=128,            # Mini-batch size
    tau=0.02,                  # Soft update coefficient
    tensorboard_log=tensorboard_log_dir,
    device='cuda',            # Use GPU if available 
)

# Train the model
model.learn(total_timesteps=1_000_000, callback=TensorboardCallback())

# Save the trained model
# model.save(f"models/ppo_model_{symbol}")

model.save(f"models/SAC_model_{symbol}")