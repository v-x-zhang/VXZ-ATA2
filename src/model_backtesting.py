import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from custom_trading_env import CustomTradingEnv
# Import the feature calculation functions if they are not in custom_trading_env
from RL_trainer import (
    compute_rsi, compute_average_true_range, compute_bollinger_band_width,
    compute_macd, compute_adx
)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
symbol = "AAPL"
model_path = f"../models/SAC_model_{symbol}_final.zip"  # Path to your saved final model
stats_path = f"../models/SAC_vecnormalize_{symbol}_final.pkl" # Path to saved normalization stats
data_path = f"../data/{symbol}.csv"
initial_balance = 100000 # Should match the training initial balance

# --- Load and Preprocess Data ---
print("Loading and preprocessing data...")
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Add time features
hour = df['timestamp'].dt.hour
minute = df['timestamp'].dt.minute
df['time_of_day'] = (hour * 60 + minute) / (24 * 60)
df = df[['open', 'high', 'low', 'close', 'volume', 'time_of_day']] # Keep time_of_day
df = df.astype(float)

# Add technical indicators (ensure these functions are accessible)
df['ma_10'] = df['close'].rolling(window=10).mean()
df['rsi'] = compute_rsi(df['close'])
df['momentum'] = df['close'] - df['close'].shift(10)
df['roc'] = (df['close'] / df['close'].shift(10) - 1) * 100
df['atr'] = compute_average_true_range(df, window=14)
df['bbands_width'] = compute_bollinger_band_width(df, window=20)
df['macd'] = compute_macd(df['close'])
df['adx'] = compute_adx(df, window=14)

# Handle NaN values
df.loc[:, 'rsi'] = df['rsi'].fillna(50)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# --- Data Splitting (Isolate Test Set) ---
total_len = len(df)
train_size = int(total_len * 0.75)
eval_size = int(total_len * 0.125)
test_df = df[train_size + eval_size:].reset_index(drop=True) # Use the last 12.5%

print(f"Using test data from index {train_size + eval_size} onwards ({len(test_df)} rows)")

# --- Load Model and Environment ---
print("Loading trained model and normalization stats...")

# Create the test environment function
# Use lookback_window and other params consistent with training
lookback_window = 30 # Make sure this matches the training env
max_episode_length_test = len(test_df) - lookback_window - 1 # Run through the whole test set

def make_test_env():
    return CustomTradingEnv(
        df=test_df,
        initial_balance=initial_balance,
        lookback_window=lookback_window,
        max_episode_length=max_episode_length_test, # Adjust for test set length
        transaction_cost=0.000, # Match training cost
        term_train=False # Assuming no termination action needed for backtesting
    )

# Wrap in DummyVecEnv for VecNormalize
dummy_env = DummyVecEnv([make_test_env])

# Load the normalization statistics
# Set training=False to prevent updates to the running mean/std
# Set norm_reward=False as we don't need normalized rewards for evaluation
vec_test_env = VecNormalize.load(stats_path, dummy_env)
vec_test_env.training = False
vec_test_env.norm_reward = False

# Load the trained agent
model = SAC.load(model_path, env=vec_test_env) # Provide the env for device placement if needed
print("Model and environment loaded successfully.")

# --- Run Backtest ---
print("Starting backtest...")
obs = vec_test_env.reset()
done = False
total_steps = 0

# Store results
portfolio_values = [initial_balance]
balances = [initial_balance]
holdings_list = [0]
actions_list = []
rewards_list = []
trade_steps = []
trade_prices = []
trade_types = [] # 'buy' or 'sell'

# Access the underlying environment to get detailed info
underlying_env = vec_test_env.envs[0]

while not done:
    # Use deterministic=True for evaluation
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_test_env.step(action)

    # Extract detailed info from the underlying environment's info dictionary
    # Note: info from VecNormalize might be slightly different
    raw_info = underlying_env.unwrapped.get_attr('info')[0] if hasattr(underlying_env.unwrapped, 'get_attr') else info[0]


    portfolio_values.append(raw_info.get('total_value', portfolio_values[-1]))
    balances.append(raw_info.get('balance', balances[-1]))
    holdings_list.append(raw_info.get('holdings', holdings_list[-1]))
    actions_list.append(action[0]) # Store the action taken
    rewards_list.append(reward[0]) # Store the reward received

    # Check for trades in the underlying env's trades list (if implemented)
    if hasattr(underlying_env, 'trades') and underlying_env.trades:
         # Assuming trades are appended like: (step, type, price, allocation, shares)
         last_trade = underlying_env.trades[-1]
         if last_trade[0] == underlying_env.current_step -1: # Check if trade happened in the last step
             trade_steps.append(underlying_env.current_step -1)
             trade_prices.append(last_trade[2])
             trade_types.append(last_trade[1])


    total_steps += 1
    if total_steps % 1000 == 0:
        print(f"Step: {total_steps}, Portfolio Value: {portfolio_values[-1]:.2f}")

    # Handle potential manual termination if 'done' isn't triggered correctly
    if total_steps >= max_episode_length_test:
        print("Reached max steps for backtest.")
        break

print(f"Backtest finished after {total_steps} steps.")

# --- Calculate Metrics ---
print("Calculating performance metrics...")
portfolio_values = np.array(portfolio_values)
returns = pd.Series(portfolio_values).pct_change().dropna()

# Total Return
total_return_pct = (portfolio_values[-1] - initial_balance) / initial_balance * 100

# Sharpe Ratio (assuming daily data and risk-free rate of 0)
# Adjust frequency (252 for daily, 52 for weekly, etc.) if needed
risk_free_rate_annual = 0.0
trading_days_per_year = 252 # Assuming daily data
daily_returns = returns
mean_daily_return = daily_returns.mean()
std_daily_return = daily_returns.std()
sharpe_ratio = (mean_daily_return * trading_days_per_year - risk_free_rate_annual) / (std_daily_return * np.sqrt(trading_days_per_year)) if std_daily_return > 0 else 0

# Max Drawdown
cumulative_returns = (1 + daily_returns).cumprod()
peak = cumulative_returns.expanding(min_periods=1).max()
drawdown = (cumulative_returns - peak) / peak
max_drawdown_pct = drawdown.min() * 100

# --- Print Results ---
print("\n--- Backtest Results ---")
print(f"Final Portfolio Value: ${portfolio_values[-1]:,.2f}")
print(f"Total Return: {total_return_pct:.2f}%")
print(f"Sharpe Ratio (Annualized): {sharpe_ratio:.4f}")
print(f"Max Drawdown: {max_drawdown_pct:.2f}%")
print(f"Total Trades: {len(trade_steps)}") # Count trades recorded
print("------------------------\n")

# --- Plot Results ---
print("Generating plot...")
plt.figure(figsize=(14, 7))
plt.plot(portfolio_values, label='Portfolio Value')

# Add markers for trades
buy_indices = [i for i, t in enumerate(trade_types) if t == 'buy']
sell_indices = [i for i, t in enumerate(trade_types) if t == 'sell']
buy_steps = [trade_steps[i] for i in buy_indices]
sell_steps = [trade_steps[i] for i in sell_indices]

# Adjust indices if portfolio_values includes initial balance
plot_offset = 1 # If portfolio_values[0] is initial_balance

plt.scatter([s + plot_offset for s in buy_steps], portfolio_values[[s + plot_offset for s in buy_steps]], marker='^', color='g', label='Buy', s=100, zorder=5)
plt.scatter([s + plot_offset for s in sell_steps], portfolio_values[[s + plot_offset for s in sell_steps]], marker='v', color='r', label='Sell', s=100, zorder=5)


plt.title(f'{symbol} Backtest Performance')
plt.xlabel('Steps')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../results/{symbol}_backtest_performance.png")
print(f"Plot saved to results/{symbol}_backtest_performance.png")
plt.show()