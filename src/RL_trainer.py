from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from custom_trading_env import CustomTradingEnv
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np

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
            if "episode_return" in last_info:
                self.logger.record("custom/episode_return", last_info["episode_return"])
            if "drawdown_penalty" in last_info:
                self.logger.record("custom/drawdown_penalty", last_info["drawdown_penalty"])
            if "trade_penalty" in last_info:
                self.logger.record("custom/trade_penalty", last_info["trade_penalty"])
            if "unrealized_pnl" in last_info:
                self.logger.record("custom/unrealized_pnl", last_info["unrealized_pnl"])
            if "transaction_volume" in last_info:
                self.logger.record("custom/transaction_volume", last_info["transaction_volume"])
            if "trade_count" in last_info:
                self.logger.record("custom/trade_count", last_info["trade_count"])
            if "momentum" in last_info:
                self.logger.record("custom/momentum", last_info["momentum"])
            
            # Log final balance and overall return at episode end
            dones = self.locals.get("dones", [])
            if any(dones) and "total_value" in last_info:
                # Calculate overall return compared to initial balance
                initial_balance = 100000  # Using default from environment
                final_balance = last_info["total_value"]
                overall_return = (final_balance - initial_balance) / initial_balance * 100  # percentage
                
                self.logger.record("episode/final_balance", final_balance)
                self.logger.record("episode/overall_return_pct", overall_return)
                self.logger.record("episode/initial_balance", initial_balance)
            
            if any(dones) and "total_realized_profit" in last_info:
                # Log as episode metric instead of continuous metric
                self.logger.record("episode/total_realized_profit", last_info["total_realized_profit"])
            else:
                # Log as regular step metric
                if "total_realized_profit" in last_info:
                    self.logger.record("custom/total_realized_profit", last_info["total_realized_profit"])


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

def compute_average_true_range(df, window=14):
    """
    Compute the Average True Range (ATR) for volatility measurement.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
        window (int): Lookback window for ATR calculation (default is 14).
        
    Returns:
        pd.Series: ATR values.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate the true range
    tr1 = high - low  # Current high - current low
    tr2 = abs(high - close.shift())  # Current high - previous close
    tr3 = abs(low - close.shift())  # Current low - previous close
    
    # Take the maximum of the three true ranges
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate the Average True Range
    atr = true_range.rolling(window=window, min_periods=1).mean()
    
    return atr

def compute_bollinger_band_width(df, window=20, num_std=2):
    """
    Compute the Bollinger Band Width as a volatility indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with 'close' price column.
        window (int): Lookback window for calculation (default is 20).
        num_std (int): Number of standard deviations for bands (default is 2).
        
    Returns:
        pd.Series: Bollinger Band Width values.
    """
    # Calculate the middle band (simple moving average)
    middle_band = df['close'].rolling(window=window, min_periods=1).mean()
    
    # Calculate the standard deviation
    std_dev = df['close'].rolling(window=window, min_periods=1).std()
    
    # Calculate the upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    # Calculate the bandwidth
    bandwidth = (upper_band - lower_band) / middle_band
    
    return bandwidth

def compute_macd(close_prices, fast_period=12, slow_period=26, signal_period=9):
    """
    Compute the Moving Average Convergence Divergence (MACD).
    
    Args:
        close_prices (pd.Series): Series of close prices.
        fast_period (int): Period for the fast EMA (default is 12).
        slow_period (int): Period for the slow EMA (default is 26).
        signal_period (int): Period for the signal line (default is 9).
        
    Returns:
        pd.Series: MACD line (difference between fast and slow EMAs).
    """
    # Calculate the fast and slow EMAs
    fast_ema = close_prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close_prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate the MACD line
    macd_line = fast_ema - slow_ema
    
    # Return the MACD line
    return macd_line

def compute_adx(df, window=14):
    """
    Compute the Average Directional Index (ADX) for trend strength measurement.
    
    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', and 'close' columns.
        window (int): Lookback window for calculation (default is 14).
        
    Returns:
        pd.Series: ADX values.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate the up move and down move
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    # Calculate the positive directional movement (+DM) and negative directional movement (-DM)
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Convert to Series for easier manipulation
    pos_dm = pd.Series(pos_dm, index=df.index)
    neg_dm = pd.Series(neg_dm, index=df.index)
    
    # Calculate the True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth the +DM, -DM, and TR using the specified window
    smoothed_pos_dm = pos_dm.rolling(window=window, min_periods=1).sum()
    smoothed_neg_dm = neg_dm.rolling(window=window, min_periods=1).sum()
    smoothed_tr = true_range.rolling(window=window, min_periods=1).sum()
    
    # Calculate the positive directional indicator (+DI) and negative directional indicator (-DI)
    pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
    neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
    
    # Calculate the directional movement index (DX)
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di + 1e-8)  # Adding small value to avoid division by zero
    
    # Calculate the ADX as the smoothed average of DX
    adx = dx.rolling(window=window, min_periods=1).mean()
    
    return adx

# Load historical data
symbol = "AAPL"
df = pd.read_csv(f"data/{symbol}.csv")

# Preprocess the data
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp to datetime
# df = df.set_index('timestamp').resample('min').ffill().reset_index()  # Fill missing timestamps
# df = df.sort_values(by='timestamp')  # Sort data by timestamp
# Add time features to help the agent learn intraday patterns
hour = df['timestamp'].dt.hour
minute = df['timestamp'].dt.minute
time_of_day = (hour * 60 + minute) / (24 * 60)  # Normalized time of day
df = df[['open', 'high', 'low', 'close', 'volume']]  # Ensure correct column order
df = df.astype(float)  # Ensure all remaining columns are numeric
df['time_of_day'] = time_of_day

# Add technical indicators
df['ma_10'] = df['close'].rolling(window=10).mean()
df['rsi'] = compute_rsi(df['close'])  # Implement compute_rsi function
df['momentum'] = df['close'] - df['close'].shift(10)  # Momentum: Difference between current and 10 steps ago
df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100  # Rate of Change (ROC)

# Add volatility indicators
df['atr'] = compute_average_true_range(df, window=14)
df['bbands_width'] = compute_bollinger_band_width(df, window=20)

# Add trend indicators
df['macd'] = compute_macd(df['close'])
df['adx'] = compute_adx(df, window=14)  # Average Directional Index for trend strength



# Handle NaN values
df.loc[:, 'rsi'] = df['rsi'].fillna(50)  # Fill NaN RSI values with 50
df.dropna(inplace=True)  # Drop rows with any remaining NaN values

# Initialize the environment
env = CustomTradingEnv(df=df)

# # Wrap the environment for vectorized training
vec_env = make_vec_env(lambda: env, n_envs=1)

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
    learning_rate=3e-4,        # Slightly higher learning rate
    ent_coef='auto_0.1',       # Start with higher entropy coefficient
    gamma=0.99,
    buffer_size=200000,        # Larger buffer for better experience replay
    learning_starts=2000,      # More initial exploration
    batch_size=256,            # Larger batch size
    tau=0.01,                  # Slower target update
    tensorboard_log=tensorboard_log_dir,
    device='cuda',
)

# Train the model
model.learn(total_timesteps=1_000_000, callback=TensorboardCallback())

# Save the trained model
# model.save(f"models/ppo_model_{symbol}")

model.save(f"models/SAC_model_{symbol}")