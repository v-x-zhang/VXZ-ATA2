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
import random


class CustomTradingEnv(gym.Env):
    """
    Custom Trading Environment for Reinforcement Learning.
    """
    metadata = {'render_modes': ['human']}
    render_mode = 'human'

    def __init__(self, df, initial_balance=100000, lookback_window=30, transaction_cost=0.002, transition_steps=8000, max_episode_length=8000, term_train=True):
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
        self.term_train = term_train
        self.df = df.reset_index(drop=True)
        self.transition_steps = transition_steps  # Number of steps before switching reward functions
        self.total_steps = 0  # Track total steps taken
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        # self.max_episode_length = max_episode_length  # Maximum episode length
        # OR
        self.max_episode_length = random.randint(2000, 8000)  # Randomize episode length

        self.current_episode_steps = 0  # Track steps within the current episode
        self.transaction_cost = transaction_cost

        # Ensure dataset is large enough
        assert len(self.df) > self.lookback_window, \
            "Dataset must have more rows than the lookback window."

        # Define action space: Discrete (0 = hold, 1 = buy, 2 = sell)
        # self.action_space = spaces.Discrete(3)

        if(self.term_train):
            # Define action space: Continuous (-1 = sell all, 1 = buy all)
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

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

        # Randomize the initial step and balance
        max_start = len(self.df) - self.lookback_window - 1
        self.current_step = random.randint(self.lookback_window, max_start)
        self.balance = self.initial_balance * random.uniform(0.8, 1.2)

        # Deterministic initial step and balance
        # self.current_step = self.lookback_window
        # self.balance = self.initial_balance

        self.holdings = 0
        self.total_value = self.initial_balance
        self.trades = []
        self.current_episode_steps = 0  # Reset episode step counter

        # Return the initial observation and an empty info dictionary
        return self._get_observation(), {}

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (float): Continuous action in the range [-1, 1].
                        -1 = sell all holdings, 1 = buy with all balance.

        Returns:
            observation (np.array): Next state.
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """

        trading_action = action
        if(self.term_train):
            # Extract trading and termination actions
            trading_action = action[0]
            terminate_action = action[1]

        # Get current price
        current_price = self.df.loc[self.current_step, 'close']

        # Handle edge case: Division by zero
        if current_price == 0:
            reward = -1  # Penalize the agent for invalid data
            done = True
            return self._get_observation(), reward, done, False, {}

        # Execute action
        if trading_action > 0:  # Buy
            allocation = min(trading_action, 1)  # Cap allocation to 1 (100%)
            amount_to_buy = allocation * self.balance
            self.holdings += amount_to_buy / current_price
            self.balance -= amount_to_buy
            self.trades.append((self.current_step, 'buy', current_price, allocation))
        elif trading_action < 0:  # Sell
            allocation = min(abs(trading_action), 1)  # Cap allocation to 1 (100%)
            amount_to_sell = allocation * self.holdings * current_price
            self.balance += amount_to_sell * (1 - self.transaction_cost)
            self.holdings -= allocation * self.holdings
            self.trades.append((self.current_step, 'sell', current_price, allocation))

        # Update portfolio value
        previous_total_value = self.total_value
        self.total_value = self.balance + self.holdings * current_price

        sharpe_ratio = 0  # Initialize Sharpe Ratio

        # Calculate step-to-step profit/loss
        step_return = (self.total_value - previous_total_value) / self.initial_balance

        # Track returns for Sharpe Ratio calculation
        if not hasattr(self, 'returns'):
            self.returns = []
        self.returns.append(step_return)

        # Calculate Sharpe Ratio
        if len(self.returns) > 1:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns) + 1e-8  # Avoid division by zero
            risk_free_rate = 0.01 / 252  # Daily risk-free rate
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
        else:
            sharpe_ratio = 0

        # Calculate drawdown penalty
        max_portfolio_value = max(self.total_value, getattr(self, 'max_portfolio_value', self.total_value))
        self.max_portfolio_value = max_portfolio_value
        drawdown = (max_portfolio_value - self.total_value) / max_portfolio_value if max_portfolio_value > 0 else 0
        drawdown_penalty = 0

        # Include sentiment score
        sentiment_score = 0
        # Calculate sentiment score
        if 'sentiment' in self.df.columns:
            sentiment_score = self.df.loc[self.current_step, 'sentiment']
        else:
            sentiment_score = 0

        if self.term_train:
            # Check if the agent chooses to terminate the episode
            if terminate_action > 0.5:  # Threshold for termination
                reward = -0.1  # Small penalty for terminating early
                if drawdown > 0.2:
                    reward += 0.3  # Reward for terminating early
                done = True
                truncated = True
                return self._get_observation(), reward, done, truncated, {}

        # Penalize large trades
        # trade_penalty = -0.001 * abs(action)
        trade_penalty = 0

        # Adjust weights dynamically based on training phase
        # if self.total_steps < self.transition_steps:
        #     reward = step_return  # Early phase: Focus on immediate profit
        # else:
        #     reward = (
        #         0.4 * step_return +
        #         0.5 * sharpe_ratio +
        #         0.1 * drawdown_penalty
        #     )

        reward = (
            0.6 * step_return +
            0.3 * sharpe_ratio +
            0.1 * sentiment_score
            # 0.2 * drawdown_penalty 
        )

        if self.term_train:
            # Add a penalty for high drawdowns or volatility
            if drawdown > 0.2:  # Example threshold for high drawdown
                reward -= 0.5  # Penalize for staying in the market



        # Advance to the next step
        self.current_step += 1
        self.current_episode_steps += 1

        # Check if the episode is done
        done = self.current_step >= len(self.df) - 1 or self.current_episode_steps >= self.max_episode_length

        truncated = self.current_episode_steps >= self.max_episode_length  # Explicitly set truncated to False for compatibility

        # Get next observation
        observation = self._get_observation()

        # Add metrics to the info dictionary
        info = {
            "sharpe_ratio": sharpe_ratio,
            "step_return": step_return,
            "drawdown_penalty": drawdown_penalty,
            "trade_penalty": trade_penalty,
        }

        # print(f"Step: {self.current_step}, Action: {action}, Total Value: {self.total_value}") #  Balance: {self.balance}, Holdings: {self.holdings},
        # Return step information
        return observation, reward, done, truncated, info

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

        # Add portfolio balance and holdings as features (broadcast scalar values to match the frame length)
        frame['balance'] = (self.balance / self.initial_balance) * np.ones(len(frame))  # Normalize balance
        if frame['close'].iloc[-1] == 0:
            frame['holdings'] = np.zeros(len(frame))  # Set holdings to zero if the closing price is zero
        else:
            frame['holdings'] = (self.holdings / (self.initial_balance / frame['close'].iloc[-1])) * np.ones(len(frame))  # Normalize holdings

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