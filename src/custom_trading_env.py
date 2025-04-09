# Tasks:
#     Create a gym.Env-compatible environment or a custom class that simulates trading over historical data:
#         Observation Space: Latest price bars, technical indicators, optional sentiment signals, current portfolio holdings.
#         Action Space: Discrete (buy, sell, hold) or continuous (allocation weights).
#         Reward: Day-to-day or step-to-step PnL, risk-adjusted metric, etc.
#     Use frameworks like Gym, Stable-Baselines3, or FinRL to accelerate RL dev.

# Deliverable: A fully defined RL environment for single or multiple assets with the standard step(), reset(), render() methods

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class CustomTradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    """
    metadata = {'render_modes': ['human']}
    render_mode = 'human'

    def __init__(self, df, initial_balance=100000, lookback_window=30, transaction_cost=0.001):
        """
        Initialize the environment.

        Args:
            df (pd.DataFrame): Historical data with columns ['open', 'high', 'low', 'close', 'volume'].
            initial_balance (float): Starting portfolio balance.
            lookback_window (int): Number of past time steps to include in the observation.
            transaction_cost (float): Cost of each transaction as a fraction of trade value.
        """
        super(CustomTradingEnv, self).__init__()

        # Store parameters
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost

        # Ensure dataset is large enough
        assert len(self.df) > self.lookback_window, \
            "Dataset must have more rows than the lookback window."

        # Define action space: Discrete (0 = hold, 1 = buy, 2 = sell)
        self.action_space = spaces.Discrete(3)

        # Define observation space: Price bars, portfolio balance, and holdings
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, len(df.columns) + 2),  # +2 for balance and holdings
            dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.holdings = 0
        self.total_value = self.initial_balance
        self.trades = []

        # Return the initial observation and an empty info dictionary
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (int): Action to take (0 = hold, 1 = buy, 2 = sell).

        Returns:
            observation (np.array): Next state.
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        # Get current price
        current_price = self.df.loc[self.current_step, 'close']

        # Handle edge case: Division by zero
        if current_price == 0:
            reward = -1  # Penalize the agent for invalid data
            done = True
            return self._get_observation(), reward, done, False, {}

        # Execute action
        if action == 1:  # Buy
            if self.balance > 0:
                self.holdings += self.balance / current_price
                self.balance = 0
                self.trades.append((self.current_step, 'buy', current_price))
        elif action == 2:  # Sell
            if self.holdings > 0:
                self.balance += self.holdings * current_price * (1 - self.transaction_cost)
                self.holdings = 0
                self.trades.append((self.current_step, 'sell', current_price))

        # Update portfolio value
        previous_total_value = self.total_value
        self.total_value = self.balance + self.holdings * current_price
        
        # Calculate normalized reward (OLD)
        # reward = (self.total_value - self.initial_balance) / self.initial_balance
        
         # Calculate step-to-step profit/loss
        step_return = self.total_value - previous_total_value

        # Track returns for Sharpe Ratio calculation
        if not hasattr(self, 'returns'):
            self.returns = []
        self.returns.append(step_return)

        # Calculate Sharpe Ratio as reward
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) + 1e-8  # Avoid division by zero
            risk_free_rate = 0.01 / 252  # Daily risk-free rate
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
        else:
            sharpe_ratio = 0  # Not enough data to calculate Sharpe Ratio

        reward = sharpe_ratio

        trade_penalty = -0.001 if action in [1, 2] else 0
        reward += trade_penalty

        # Advance to the next step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1
        truncated = False  # Explicitly set truncated to False for compatibility

        # Get next observation
        observation = self._get_observation()

        # print(f"Step: {self.current_step}, Action: {action}, Balance: {self.balance}, Holdings: {self.holdings}, Total Value: {self.total_value}")
        # Return step information
        return observation, reward, done, truncated, {}

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            np.array: Observation array.
        """
        # Get price bars and technical indicators
        frame = self.df.iloc[self.current_step - self.lookback_window:self.current_step].copy()


        # Normalize price and volume data
        price_columns = ['open', 'high', 'low', 'close']
        volume_column = 'volume'
        technical_columns = ['ma_10', 'rsi']  # Add technical indicators here

        # Normalize prices using min-max scaling
        frame[price_columns] = (frame[price_columns] - frame[price_columns].min()) / (frame[price_columns].max() - frame[price_columns].min() + 1e-8)

        # Normalize volume using z-score normalization
        frame[volume_column] = (frame[volume_column] - frame[volume_column].mean()) / (frame[volume_column].std() + 1e-8)

        # Normalize technical indicators using z-score normalization
        frame[technical_columns] = (frame[technical_columns] - frame[technical_columns].mean()) / (frame[technical_columns].std() + 1e-8)

        # Add portfolio balance and holdings as features
        frame['balance'] = self.balance / self.initial_balance  # Normalize balance
        if frame['close'].iloc[-1] == 0:
            frame['holdings'] = 0  # Set holdings to zero if the closing price is zero
        else:
            frame['holdings'] = self.holdings / (self.initial_balance / frame['close'].iloc[-1])     # Normalize holdings   
        
        # # Add portfolio balance and holdings as features
        # frame['balance'] = self.balance
        # frame['holdings'] = self.holdings

        # Ensure the observation matches the expected shape
        observation = frame.values
        assert observation.shape == self.observation_space.shape, \
            f"Observation shape mismatch: expected {self.observation_space.shape}, got {observation.shape}"

        # Ensure the observation does not contain NaN or inf values
        assert not np.any(np.isnan(observation)), "Observation contains NaN values."
        assert not np.any(np.isinf(observation)), "Observation contains inf values."


        return observation

    def render(self, render_mode='human'):
        """
        Render the environment (optional).
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance}")
        print(f"Holdings: {self.holdings}")
        print(f"Total Value: {self.total_value}")

    def close(self):
        """
        Close the environment (optional).
        """
        pass