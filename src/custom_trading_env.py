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

    def __init__(self, df, initial_balance=100000, lookback_window=30, transaction_cost=0.000, transition_steps=8000, max_episode_length=8000, term_train=True, DEBUG=False):
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
        self.debug = DEBUG
        
        self.returns = []  # Track returns for Sharpe Ratio calculation
        self.current_episode_steps = 0  # Track steps within the current episode
        self.transaction_cost = transaction_cost
        self.low_holdings_counter = 0  # Counter for consecutive low holdings
        self.low_holdings_threshold = 0.05  # Threshold for low holdings (e.g., 5% of initial balance)
        self.total_realized_profit = 0  # Track total realized profit

        self.debug_counter = 0

        # Initialize cost basis tracking
        self.cost_basis_per_share = 0
        self.total_shares_bought = 0
        self.total_cost = 0
        
        # Initialize metrics tracking
        self.transaction_volume = 0
        self.trade_count = 0
        self.max_portfolio_value = self.initial_balance

        # Ensure dataset is large enough
        assert len(self.df) > self.lookback_window, \
            "Dataset must have more rows than the lookback window."

        # Define action space
        if(self.term_train):
            # Define action space: Continuous (-1 = sell all, 1 = buy all) with termination
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
        max_start = len(self.df) - self.lookback_window - self.max_episode_length - 1
        self.current_step = random.randint(self.lookback_window, max(self.lookback_window, max_start))
        self.balance = self.initial_balance
        
        # Reset portfolio metrics
        self.holdings = 0
        self.total_value = self.initial_balance
        self.max_portfolio_value = self.initial_balance
        self.episode_return = 0  # Reset cumulative return for the episode
        self.current_episode_steps = 0  # Reset episode step counter
        self.low_holdings_counter = 0  # Reset low holdings counter
        self.total_realized_profit = 0
        
        # Reset transaction metrics
        self.transaction_volume = 0
        self.trade_count = 0
        
        # Reset cost basis tracking
        self.cost_basis_per_share = 0
        self.total_shares_bought = 0
        self.total_cost = 0
        
        # Reset returns and trades history
        self.returns = []
        self.trades = []

        # Return the initial observation and an empty info dictionary
        return self._get_observation(), {}

    def _execute_trade(self, trading_action, current_price):
        """
        Execute a trading action and update portfolio state.
        
        Args:
            trading_action (float): Value between -1 (sell all) and 1 (buy all)
            current_price (float): Current asset price
            
        Returns:
            realized_profit (float): Profit realized from this trade
        """
        # Store initial values for balance and holdings
        last_balance = self.balance
        last_holdings = self.holdings
        
        realized_profit = 0
        
        # Execute trading action
        if trading_action > 0:  # Buy
            allocation = min(trading_action, 1)  # Cap allocation to 1 (100%)
            max_amount_to_buy = self.balance / (1 + self.transaction_cost)  # Account for transaction costs
            amount_to_buy = allocation * max_amount_to_buy
            
            if amount_to_buy > 0:
                # Calculate transaction cost
                # transaction_fee = amount_to_buy * self.transaction_cost
                # actual_amount = amount_to_buy - transaction_fee
                actual_amount = amount_to_buy 
                
                # Calculate shares and update holdings
                shares_bought = actual_amount / current_price if current_price > 0 else 0
                
                # Update cost basis using weighted average
                if shares_bought > 0:
                    self.total_shares_bought += shares_bought
                    self.total_cost += actual_amount
                    if self.total_shares_bought > 0:
                        self.cost_basis_per_share = self.total_cost / self.total_shares_bought
                    
                    # Update holdings and balance
                    self.holdings += shares_bought
                    self.balance -= actual_amount
                    self.trades.append((self.current_step, 'buy', current_price, allocation, shares_bought))
        elif trading_action < 0:  # Sell
            allocation = min(abs(trading_action), 1)  # Cap allocation to 1 (100%)
            shares_to_sell = allocation * self.holdings
            
            if shares_to_sell > 0 and self.holdings > 0:
                # Calculate sale amount and transaction fee
                # sale_amount = shares_to_sell * current_price
                # transaction_fee = sale_amount * self.transaction_cost
                # actual_amount = sale_amount - transaction_fee
                actual_amount = shares_to_sell * current_price

                # Calculate realized profit or loss
                realized_profit = shares_to_sell * (current_price - self.cost_basis_per_share)
                

                # Update cost basis tracking when selling shares
                if self.total_shares_bought > 0:
                    # Calculate the proportion of total shares being sold
                    proportion_sold = shares_to_sell / self.total_shares_bought
                    
                    # Reduce total cost proportionally
                    self.total_cost *= (1 - proportion_sold)
                    self.total_shares_bought -= shares_to_sell
                    
                    # Recalculate cost basis
                    if self.total_shares_bought > 0:
                        self.cost_basis_per_share = self.total_cost / self.total_shares_bought
                    else:
                        self.cost_basis_per_share = 0

                # Update holdings and balance
                self.holdings -= shares_to_sell
                self.balance += actual_amount
                self.total_realized_profit += realized_profit
                

                if(self.debug):
                    if self.debug_counter < 20:
                        self.debug_counter += 1

                        print(f"Step {self.current_step}: Sold {shares_to_sell} shares at {current_price:.2f}, realized profit: {realized_profit:.2f}, total realized profit: {self.total_realized_profit:.2f}, total_moneys: {(self.balance + self.holdings * self.cost_basis_per_share):.2f}")
                        print(f"Cost basis per share: {self.cost_basis_per_share:.2f}, total shares bought: {self.total_shares_bought:.2f}, total cost: {self.total_cost:.2f}")
                    # if self.current_step % 100 == 0:
                    
                
                self.trades.append((self.current_step, 'sell', current_price, allocation, shares_to_sell))
        
        # Update transaction metrics if there was a trade
        if self.holdings != last_holdings:
            trade_volume = abs(self.holdings - last_holdings) * current_price
            self.transaction_volume += trade_volume
            self.trade_count += 1
            
        return realized_profit

    def _calculate_reward(self, realized_profit, current_price, trading_action, terminate_action=None):
        """
        Calculate the reward based on multiple factors.
        
        Args:
            realized_profit (float): Profit realized from the current trade
            current_price (float): Current asset price
            trading_action (float): The trading action taken
            terminate_action (float, optional): Termination action if applicable
            
        Returns:
            reward (float): The calculated reward
        """
        # Calculate unrealized profit/loss
        unrealized_pnl = self.holdings * (current_price - self.cost_basis_per_share)
        unrealized_pnl = np.clip(unrealized_pnl, -self.initial_balance, self.initial_balance)
        
        # Inactivity penalty
        if self.holdings * current_price < self.low_holdings_threshold * self.initial_balance:
            self.low_holdings_counter += 1
        else:
            self.low_holdings_counter = 0
        inactivity_penalty = -0.01 * min(self.low_holdings_counter, 10)  # Cap penalty
        
        portfolio_value = self.balance + self.holdings * current_price
        pnl_scale = min(self.initial_balance * 0.01, 1000)  # 5% of initial balance or 5000, whichever is smaller
        
        # Realized profit - more adaptive scaling
        if realized_profit > 0:
            realized_profit_reward = realized_profit / pnl_scale * 1.5  # Bonus for profits
        else:
            realized_profit_reward = realized_profit / pnl_scale  # Regular scaling for losses
        
        # Unrealized PnL component
        if unrealized_pnl > 0:
            pnl_reward = (unrealized_pnl / pnl_scale)
        else:
            pnl_reward = -2 * (abs(unrealized_pnl) / pnl_scale)  # Penalize losses more
        
        # Sharpe ratio component
        sharpe_ratio = 0
        if len(self.returns) > 10:
            mean_return = np.mean(self.returns[-20:])  # Look at recent returns
            std_return = np.std(self.returns[-20:]) + 1e-8
            risk_free_rate = 0.0001  # Very small for short time periods
            sharpe_ratio = (mean_return - risk_free_rate) / std_return
            sharpe_reward = 0.2 * np.clip(sharpe_ratio, -1, 1)
        else:
            sharpe_reward = 0
        
        # Drawdown component
        drawdown = (self.max_portfolio_value - self.total_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0
        if drawdown > 0.1:  # Only penalize significant drawdowns
            drawdown_penalty = -0.3 * drawdown
        else:
            drawdown_penalty = 0
        
        # Volatility penalty
        volatility_penalty = 0
        if len(self.returns) > 10:
            recent_vol = np.std(self.returns[-10:]) 
            volatility_penalty = -0.1 * recent_vol
        
        # Momentum alignment reward
        momentum = self.df.loc[self.current_step, 'momentum']
        momentum_reward = 0
        # if momentum > 0 and trading_action > 0:  # Reward buying in positive momentum
        #     momentum_reward = 0.1
        # elif momentum < 0 and trading_action < 0:  # Reward selling in negative momentum
        #     momentum_reward = 0.1
        # elif momentum > 0 and trading_action < 0:  # Penalize selling in positive momentum
        #     momentum_reward = -0.1
        # elif momentum < 0 and trading_action > 0:  # Penalize buying in negative momentum
        #     momentum_reward = -0.1
        
        # Sentiment component
        sentiment_score = 0
        if 'sentiment' in self.df.columns:
            sentiment_score = self.df.loc[self.current_step, 'sentiment']
        

        # Position sizing reward - encourage appropriate sizing
        holdings_ratio = (self.holdings * current_price) / portfolio_value if portfolio_value > 0 else 0
        
        # Reward being appropriately invested based on market conditions
        # adx = self.df.loc[self.current_step, 'adx'] if 'adx' in self.df.columns else 20
        # if adx > 25:  # Strong trend detected
        #     # In strong trends, reward taking larger positions
        #     optimal_ratio = 0.7
        #     position_reward = 0.05 * (1 - abs(holdings_ratio - optimal_ratio))
        # else:
        #     # In weaker trends, reward smaller positions
        #     optimal_ratio = 0.3
        #     position_reward = 0.03 * (1 - abs(holdings_ratio - optimal_ratio))
        

        # Combine reward components with appropriate weights
        reward = (
            1 * realized_profit_reward +  # Prioritize realized profits
            0.2 * pnl_reward +  # Consider unrealized PnL
            0.2 * sharpe_reward +  # Risk adjustment
            # 0.2 * position_reward +  # Position sizing reward
            0.1 * sentiment_score +  # Incorporate sentiment
            # 0.1 * drawdown_penalty +  # Penalize drawdowns
            inactivity_penalty + 
            momentum_reward +
            volatility_penalty
        )

        # print(f"Realized Profit Reward={realized_profit_reward:.4f}")      
  
        # Add termination incentives if applicable
        term_penalty = 0
        if self.term_train and terminate_action is not None:
            if drawdown > 0.1:  # 20% drawdown
                term_reward = 0.5  # Reward for terminating during high drawdown
                if terminate_action > 0.3:
                    reward += term_reward
            
            # Penalize for staying in market during extreme conditions
            if drawdown > 0.25:
                term_penalty -= 0.5
                reward += term_penalty
        
        return np.clip(reward, -10, 10), {  # Clip reward to reasonable range and return components
            "sharpe_ratio": sharpe_ratio,
            "drawdown_penalty": drawdown_penalty,
            "inactivity_penalty": inactivity_penalty,
            "unrealized_pnl": unrealized_pnl,
            "pnl_reward": pnl_reward,
            # "position_reward": position_reward,
            "sharpe_reward": sharpe_reward,
            "realized_profit_reward": realized_profit_reward,
            "volatility_penalty": volatility_penalty,
            "term_penalty": term_penalty
        }

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action: Trading action (and possibly termination signal)

        Returns:
            observation, reward, done, truncated, info
        """
        # Extract trading and termination actions
        if self.term_train:
            trading_action = action[0]
            terminate_action = action[1]
        else:
            trading_action = action[0] if isinstance(action, np.ndarray) else action
            terminate_action = None
            
        # Track steps
        self.total_steps += 1
        self.current_episode_steps += 1

        # Get current price
        current_price = self.df.loc[self.current_step, 'close']

        # Handle edge case: Invalid price
        if current_price <= 0:
            return self._get_observation(), -1, True, False, {"error": "Invalid price"}

        # Store previous portfolio value
        previous_total_value = self.total_value
        
        # Execute trading action
        realized_profit = self._execute_trade(trading_action, current_price)
        
        # Update portfolio value
        self.total_value = self.balance + self.holdings * current_price
        
        # Update max portfolio value for drawdown calculation
        self.max_portfolio_value = max(self.max_portfolio_value, self.total_value)
        
        # Calculate return for this step
        step_return = (self.total_value - previous_total_value) / previous_total_value if previous_total_value > 0 else 0
        self.returns.append(step_return)
        self.episode_return += step_return
        
        # Calculate reward
        reward, reward_components = self._calculate_reward(
            realized_profit, 
            current_price, 
            trading_action, 
            terminate_action
        )
        
        if (self.debug):
            # Print reward components every 100 steps
            if self.current_episode_steps % 100 == 0:
                print(f"Step {self.current_episode_steps}: Reward of {reward}:")
                for key, value in reward_components.items():
                    print(f"  {key}: {value:.4f}")

        # Check for early termination from terminate_action
        done = False
        truncated = False
        if self.term_train and terminate_action is not None and terminate_action > 0.3:
            if reward_components.get("drawdown_penalty", 0) < -0.2:  # Significant drawdown
                done = True
                truncated = True
        
        # Check standard termination conditions if not already terminated
        if not done:
            # End episode conditions
            done = (
                self.current_step >= len(self.df) - 1 or
                self.current_episode_steps >= self.max_episode_length or
                self.total_value <= 0  # Bankrupt
            )
            truncated = self.current_episode_steps >= self.max_episode_length
        
        # Move to next step if not done
        if not done:
            self.current_step += 1
            
        # Get next observation
        observation = self._get_observation()
        
        # Compile information dictionary
        info = {
            "episode_return": self.episode_return,
            "total_value": self.total_value,
            "balance": self.balance,
            "holdings": self.holdings,
            "transaction_volume": self.transaction_volume,
            "trade_count": self.trade_count,
            "total_realized_profit": self.total_realized_profit,
            **reward_components  # Include all reward components
        }
        
        return observation, reward, done, truncated, info

    def _get_observation(self):
        """
        Get the current observation.

        Returns:
            np.array: Observation array.
        """
        # Get price bars and technical indicators
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step
        
        # Ensure we have exactly lookback_window rows
        if end_idx - start_idx < self.lookback_window:
            start_idx = max(0, end_idx - self.lookback_window)
            
        frame = self.df.iloc[start_idx:end_idx].copy()
        
        # Handle edge case where frame is shorter than lookback_window
        if len(frame) < self.lookback_window:
            # Pad with the first row repeated
            padding = pd.DataFrame([frame.iloc[0]] * (self.lookback_window - len(frame)))
            frame = pd.concat([padding, frame], ignore_index=True)

        # Normalize prices using min-max scaling with safety checks
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in frame.columns:
                col_min = frame[col].min()
                col_max = frame[col].max()
                if col_max - col_min > 1e-8:  # Avoid division by near-zero
                    frame[col] = (frame[col] - col_min) / (col_max - col_min)
                else:
                    frame[col] = 0.5  # Set to mid-range if no variation

        # Normalize volume using z-score with safety checks
        volume_column = 'volume'
        if volume_column in frame.columns:
            vol_std = frame[volume_column].std()
            if vol_std > 1e-8:  # Avoid division by near-zero
                frame[volume_column] = (frame[volume_column] - frame[volume_column].mean()) / vol_std
            else:
                frame[volume_column] = 0  # Set to zero if no variation

        # Normalize technical indicators and momentum using z-score
        technical_columns = ['ma_10', 'rsi']
        momentum_columns = ['momentum', 'roc']
        
        for col_list in [technical_columns, momentum_columns]:
            for col in col_list:
                if col in frame.columns:
                    col_std = frame[col].std()
                    if col_std > 1e-8:  # Avoid division by near-zero
                        frame[col] = (frame[col] - frame[col].mean()) / col_std
                    else:
                        frame[col] = 0  # Set to zero if no variation

        # Add normalized portfolio metrics
        frame['balance'] = (self.balance / self.initial_balance) * np.ones(len(frame))
        
        # Calculate normalized holdings based on price (avoid division by zero)
        if frame['close'].iloc[-1] > 0:
            normalized_holdings = (self.holdings * frame['close'].iloc[-1]) / self.initial_balance
        else:
            normalized_holdings = 0
        frame['holdings'] = normalized_holdings * np.ones(len(frame))

        # Convert to numpy array
        observation = frame.values.astype(np.float32)
        
        # Safety checks: replace any NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation

    def render(self, render_mode='human'):
        """
        Render the environment (optional).
        """
        print(f"Step: {self.current_step}")
        print(f"Balance: {self.balance:.2f}")
        print(f"Holdings: {self.holdings:.6f}")
        print(f"Total Value: {self.total_value:.2f}")
        print(f"Episode Return: {self.episode_return:.4f}")
        if self.total_shares_bought > 0:
            print(f"Cost Basis: {self.cost_basis_per_share:.2f}")
        print(f"Current Price: {self.df.loc[self.current_step, 'close']:.2f}")

    def close(self):
        """
        Close the environment (optional).
        """
        pass