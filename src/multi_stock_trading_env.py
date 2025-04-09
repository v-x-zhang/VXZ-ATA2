import gym
from gym import spaces
import numpy as np
import pandas as pd

class MultiStockTradingEnv(gym.Env):
    """
    A custom trading environment for multiple stocks.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, stock_dim, initial_balance=100000, lookback_window=30, transaction_cost=0.001):
        """
        Initialize the environment.

        Args:
            df (pd.DataFrame): Historical data for multiple stocks. Must include 'tic' (ticker) and 'date' columns.
            stock_dim (int): Number of stocks in the portfolio.
            initial_balance (float): Starting portfolio balance.
            lookback_window (int): Number of past time steps to include in the observation.
            transaction_cost (float): Cost of each transaction as a fraction of trade value.
        """
        super(MultiStockTradingEnv, self).__init__()

        # Store parameters
        self.df = df
        self.stock_dim = stock_dim
        self.initial_balance = initial_balance
        self.lookback_window = lookback_window
        self.transaction_cost = transaction_cost

        # Define action space: Continuous allocation weights for each stock (-1 to 1)
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim,), dtype=np.float32)

        # Define observation space: Price bars and portfolio holdings for each stock
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, stock_dim * (len(df.columns) - 2 + 1)),  # -2 for 'tic' and 'date', +1 for holdings
            dtype=np.float32
        )

        # Initialize environment state
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.
        """
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.stock_dim)
        self.total_value = self.initial_balance
        self.trades = []

        # Filter data for the current episode
        self.episode_data = self.df.copy()

        # Return the initial observation
        return self._get_observation()

    def step(self, actions):
        """
        Execute one time step within the environment.

        Args:
            actions (np.array): Allocation weights for each stock (-1 to 1).

        Returns:
            observation (np.array): Next state.
            reward (float): Reward for the action.
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        # Get current prices for all stocks
        current_prices = self.episode_data.loc[self.current_step, 'close'].values

        # Execute trades based on actions
        for i, action in enumerate(actions):
            if action > 0:  # Buy
                allocation = self.balance * action
                self.holdings[i] += allocation / current_prices[i]
                self.balance -= allocation
            elif action < 0:  # Sell
                allocation = self.holdings[i] * current_prices[i] * abs(action)
                self.holdings[i] -= allocation / current_prices[i]
                self.balance += allocation * (1 - self.transaction_cost)

        # Update portfolio value
        self.total_value = self.balance + np.sum(self.holdings * current_prices)

        # Calculate reward (PnL)
        reward = self.total_value - self.initial_balance

        # Advance to the next step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= len(self.episode_data) - 1

        # Get next observation
        observation = self._get_observation()

        # Return step information
        return observation, reward, done, {}

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            np.array: Observation array.
        """
        # Get price bars for the lookback window
        frame = self.episode_data.iloc[self.current_step - self.lookback_window:self.current_step]

        # Add portfolio holdings as features
        holdings = np.tile(self.holdings, (self.lookback_window, 1))
        frame = frame.drop(columns=['tic', 'date']).values.reshape(self.lookback_window, -1)
        observation = np.hstack([frame, holdings])

        return observation

    def render(self, mode='human'):
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