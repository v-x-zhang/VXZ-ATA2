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

    def __init__(self, df, initial_balance=100000, lookback_window=30, transaction_cost=0.002, transition_steps=8000, max_episode_length=4000, term_train=True):
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
        self.max_episode_length = max_episode_length  # Maximum episode length
        self.episode_return = 0  # Track cumulative return for the episode
        # OR
        # self.max_episode_length = random.randint(2000, 8000)  # Randomize episode length

        self.returns = []  # Track returns for Sharpe Ratio calculation

        self.current_episode_steps = 0  # Track steps within the current episode
        self.transaction_cost = transaction_cost

        self.low_holdings_counter = 0  # Counter for consecutive low holdings
        self.low_holdings_threshold = 0.05  # Threshold for low holdings (e.g., 1% of initial balance)

        self.total_realized_profit = 0  # Track total realized profit

        # Initialize cost basis tracking
        self.cost_basis_per_share = 0
        self.total_shares_bought = 0
        self.total_cost = 0

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
        momentum_columns = ['momentum', 'roc']
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
        # self.balance = self.initial_balance * random.uniform(0.8, 1.2)
        self.balance = self.initial_balance

        self.cost_basis = 0  # Initialize cost basis

        self.trade_volume = 0
        self.trade_count = 0

        self.episode_return = 0  # Reset cumulative return for the episode

        self.total_realized_profit = 0  
        # Deterministic initial step and balance
        # self.current_step = self.lookback_window
        # self.balance = self.initial_balance

        self.holdings = 0
        self.total_value = self.initial_balance
        self.returns = []  # Track returns for Sharpe Ratio calculation
        self.trades = []
        self.current_episode_steps = 0  # Reset episode step counter

        # Reset cost basis tracking
        self.cost_basis_per_share = 0
        self.total_shares_bought = 0
        self.total_cost = 0

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

        #initial values for balance and holdings
        last_balance = self.balance
        last_holdings = self.holdings


        realized_profit = 0

        # Execute action
        if trading_action > 0:  # Buy
            allocation = min(trading_action, 1)  # Cap allocation to 1 (100%)
            amount_to_buy = allocation * self.balance
            shares_bought = amount_to_buy / current_price
            
            # Update cost basis using weighted average
            self.total_shares_bought += shares_bought
            self.total_cost += amount_to_buy
            self.cost_basis_per_share = self.total_cost / self.total_shares_bought if self.total_shares_bought > 0 else 0
            
            self.holdings += shares_bought
            self.balance -= amount_to_buy
            self.trades.append((self.current_step, 'buy', current_price, allocation))
        elif trading_action < 0:  # Sell
            allocation = min(abs(trading_action), 1)  # Cap allocation to 1 (100%)
            shares_sold = allocation * self.holdings
            amount_to_sell = shares_sold * current_price
            
            # Calculate realized profit directly
            realized_profit = shares_sold * (current_price - self.cost_basis_per_share)
            
            # Update holdings and balance
            self.holdings -= shares_sold
            self.balance += amount_to_sell * (1 - self.transaction_cost)
            
            # No need to adjust cost_basis_per_share when selling
            self.total_realized_profit += realized_profit
            self.trades.append((self.current_step, 'sell', current_price, allocation))


        # Update portfolio value
        previous_total_value = self.total_value
        self.total_value = self.balance + self.holdings * current_price

        sharpe_ratio = 0  # Initialize Sharpe Ratio

        # Calculate step-to-step profit/loss
        step_return = (self.total_value - previous_total_value) / self.initial_balance

        self.episode_return += step_return  # Update cumulative return for the episode

        # Calculate unrealized profit/loss
        unrealized_pnl = self.holdings * (current_price - self.cost_basis_per_share)
        unrealized_pnl = np.clip(unrealized_pnl, -self.initial_balance, self.initial_balance)  # Cap to [-initial_balance, initial_balance]

        # Reward for holding a profitable position
        if unrealized_pnl > 0:
            pnl_reward = 0.1 * (unrealized_pnl / self.initial_balance)  # Scale reward
        else:
            pnl_reward = -0.1 * (abs(unrealized_pnl) / self.initial_balance)  # Penalize unrealized losses

        
        # Check if holdings are below the threshold
        if self.holdings * current_price < self.low_holdings_threshold * self.initial_balance:
            self.low_holdings_counter += 1  # Increment the counter
        else:
            self.low_holdings_counter = 0  # Reset the counter if holdings exceed the threshold

        inactivity_penalty = -0.01 * self.low_holdings_counter  # Increase penalty with consecutive low holdings


        # Track transaction volume
        if not hasattr(self, 'transaction_volume'):
            self.transaction_volume = 0
            self.trade_count = 0

        # Update transaction volume and trade count
        if self.holdings != last_holdings or self.balance != last_balance:
            # Correct transaction volume calculation
            trade_volume = abs(self.holdings - last_holdings) * current_price  # Use holdings difference
            self.transaction_volume += trade_volume
            self.trade_count += 1


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
        drawdown_penalty = -drawdown

        # Include sentiment score
        sentiment_score = 0
        # Calculate sentiment score
        if 'sentiment' in self.df.columns:
            sentiment_score = self.df.loc[self.current_step, 'sentiment']
        else:
            sentiment_score = 0

        
        # Normalize reward components
        step_return = np.clip(step_return, -1, 1)
        pnl_reward = np.clip(pnl_reward, -1, 1)
        drawdown_penalty = np.clip(drawdown_penalty, -1, 1)

        term_penalty = 0

        # Penalize large trades
        trade_penalty = -0.001 * abs(action)
        trade_penalty = 0

        # Calculate momentum-based reward
        momentum = self.df.loc[self.current_step, 'momentum']
        momentum_reward = 0
        if momentum > 0 and trading_action > 0:  # Reward buying in positive momentum
            momentum_reward = 0.1
        elif momentum < 0 and trading_action < 0:  # Reward selling in negative momentum
            momentum_reward = 0.1
        elif momentum > 0 and trading_action < 0:  # Penalize selling in positive momentum
            momentum_reward = -0.1
        elif momentum < 0 and trading_action > 0:  # Penalize buying in negative momentum
            momentum_reward = -0.1

        realized_profit_reward = np.clip(realized_profit / self.initial_balance, -1, 1)  # Scale reward based on initial balance

        # Add momentum reward to the total reward
        reward = (
            0.9 * realized_profit_reward +
            0.1 * pnl_reward +
            # 0.3 * sharpe_ratio +
            # 0.1 * sentiment_score
            # 0.2 * drawdown_penalty +
            trade_penalty + inactivity_penalty + term_penalty + momentum_reward
        )
        
        if self.term_train:
            # Check if the agent chooses to terminate the episode
            if terminate_action > 0.5:  # Threshold for termination
                term_penalty = -0.1  # Small penalty for terminating early
                if drawdown > 0.2:
                    term_penalty = 0.2  # Reward for terminating early
                done = True
                truncated = True
                return self._get_observation(), reward, done, truncated, {}


        if self.term_train:
            # Add a penalty for high drawdowns or volatility
            if drawdown > 0.5:  # threshold for extremely high drawdown
                term_penalty -= 0.5  # Penalize for staying in the market


        # Adjust weights dynamically based on training phase
        # if self.total_steps < self.transition_steps:
        #     reward = step_return  # Early phase: Focus on immediate profit
        # else:
        #     reward = (
        #         0.4 * step_return +
        #         0.5 * sharpe_ratio +
        #         0.1 * drawdown_penalty
        #     )
    

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
            "episode_return": self.episode_return,
            "drawdown_penalty": drawdown_penalty,
            "trade_penalty": trade_penalty,
            "inactivity_penalty": inactivity_penalty,
            "transaction_volume": self.transaction_volume,
            "trade_count": self.trade_count,
            "unrealized_pnl": unrealized_pnl,
            "pnl_reward": pnl_reward,
            "momentum": momentum,
            "momentum_reward": momentum_reward,
            "total_realized_profit": self.total_realized_profit
        }

        # Ensure no NaN or inf values in rewards
        assert not np.isnan(reward), "Reward is NaN."
        assert not np.isinf(reward), "Reward is inf."


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

        # Normalize momentum and ROC using z-score normalization
        momentum_columns = ['momentum', 'roc']
        for col in momentum_columns:
            if frame[col].std() == 0:  # Handle constant momentum
                frame[col] = 0
            else:
                frame[col] = (frame[col] - frame[col].mean()) / (frame[col].std() + 1e-8)

        # Normalize prices using min-max scaling
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            col_min = frame[col].min()
            col_max = frame[col].max()
            if col_max - col_min == 0:  # Handle constant values
                frame[col] = 0  # Set to 0 if no variation
            else:
                frame[col] = (frame[col] - col_min) / (col_max - col_min + 1e-8)

        # Normalize volume using z-score normalization
        volume_column = 'volume'
        if frame[volume_column].std() == 0:  # Handle constant volume
            frame[volume_column] = 0
        else:
            frame[volume_column] = (frame[volume_column] - frame[volume_column].mean()) / (frame[volume_column].std() + 1e-8)

        # Normalize technical indicators using z-score normalization
        technical_columns = ['ma_10', 'rsi']
        for col in technical_columns:
            if frame[col].std() == 0:  # Handle constant technical indicators
                frame[col] = 0
            else:
                frame[col] = (frame[col] - frame[col].mean()) / (frame[col].std() + 1e-8)

        # # Normalize prices using min-max scaling
        # frame[price_columns] = (frame[price_columns] - frame[price_columns].min()) / (frame[price_columns].max() - frame[price_columns].min() + 1e-8)

        # # Normalize volume using z-score normalization
        # frame[volume_column] = (frame[volume_column] - frame[volume_column].mean()) / (frame[volume_column].std() + 1e-8)

        # # Normalize technical indicators using z-score normalization
        # frame[technical_columns] = (frame[technical_columns] - frame[technical_columns].mean()) / (frame[technical_columns].std() + 1e-8)

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