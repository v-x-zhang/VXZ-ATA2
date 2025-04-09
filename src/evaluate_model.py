import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from custom_trading_env import CustomTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv

def evaluate_model(model, env, df):
    """
    Evaluate the trained model on a test dataset.

    Args:
        model: Trained PPO model.
        env: CustomTradingEnv environment.
        df: Test dataset (Pandas DataFrame).

    Returns:
        dict: A dictionary of performance metrics.
    """
    obs = env.reset()
    done = False
    total_rewards = 0
    total_trades = 0
    profit_list = []
    portfolio_values = []
    actions = []

    # Access the underlying environment
    underlying_env = env.envs[0]

    for _ in range(len(df)):
        action, _ = model.predict(obs)
        step_result = env.step(action)
        
        if len(step_result) == 5:  # Newer Gym API
            obs, reward, done, truncated, info = step_result
        else:  # Older Gym API
            obs, reward, done, info = step_result
            truncated = False  # Default value for compatibility

        done = done or truncated  # Combine `done` and `truncated` for compatibility

        # Track performance
        total_rewards += reward
        profit_list.append(reward)  # Track step-to-step profits
        portfolio_values.append(underlying_env.total_value)  # Access total_value from the underlying environment
        actions.append(action)

        # Count trades
        if action == 1 or action == 2:  # Buy or Sell
            total_trades += 1

    # Calculate performance metrics
    final_portfolio_value = underlying_env.total_value
    profit = final_portfolio_value - underlying_env.initial_balance
    returns = np.array(portfolio_values) / underlying_env.initial_balance - 1
    sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Assuming 252 trading days
    max_drawdown = (np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) /
                    np.max(portfolio_values) if len(portfolio_values) > 0 else 0)
    win_rate = len([r for r in profit_list if r > 0]) / len(profit_list) if profit_list else 0

    # Ensure metrics are scalar values
    return {
        "Final Portfolio Value": float(final_portfolio_value),
        "Total Profit": float(profit),
        "Total Trades": int(total_trades),
        "Sharpe Ratio": float(sharpe_ratio),
        "Max Drawdown": float(max_drawdown),
        "Win Rate": float(win_rate),
        "Total Rewards": float(total_rewards),
    }


if __name__ == "__main__":
    # Load the test dataset
    test_df = pd.read_csv("data/test_AAPL.csv")  # Replace with your test dataset
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
    test_df = test_df.set_index('timestamp').resample('min').ffill().reset_index()
    test_df = test_df[['open', 'high', 'low', 'close', 'volume']]
    test_df = test_df.astype(float)

    # Add technical indicators
    test_df['ma_10'] = test_df['close'].rolling(window=10).mean()
    test_df['rsi'] = (test_df['close'].diff().apply(lambda x: max(x, 0)).rolling(window=14).mean() /
                      test_df['close'].diff().abs().rolling(window=14).mean()) * 100
    test_df.loc[:, 'rsi'] = test_df['rsi'].fillna(50)  # Fill NaN RSI values with 50
    test_df.dropna(inplace=True)

    # Initialize the environment
    test_env = CustomTradingEnv(df=test_df)

    # Wrap the environment for evaluation
    test_env = DummyVecEnv([lambda: test_env])

    # Load the trained model
    model = PPO.load("models/ppo_model_AAPL")

    # Evaluate the model
    metrics = evaluate_model(model, test_env, test_df)

    # Display performance metrics
    print("Model Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.5f}")
        else:
            print(f"{key}: {value}")