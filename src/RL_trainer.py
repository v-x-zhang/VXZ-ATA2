import pandas as pd
import numpy as np
import optuna
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env
from custom_trading_env import CustomTradingEnv
from stable_baselines3.common.callbacks import BaseCallback
import json
from tqdm import tqdm

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

# --- Global Variables for Objective Function ---
train_df = None
eval_df = None
initial_balance = 100000
lookback_window = 30
hpo_train_timesteps = 100_000
final_train_timesteps = 1_000_000

# --- Callback for HPO Training Progress ---
class HpoTrainProgressCallback(BaseCallback):
    """
    Updates a TQDM progress bar during HPO training steps.
    """
    def __init__(self, pbar, verbose=0):
        super(HpoTrainProgressCallback, self).__init__(verbose)
        self.pbar = pbar
        self.last_logged_timestep = 0

    def _on_step(self) -> bool:
        steps_taken = self.num_timesteps - self.last_logged_timestep
        self.pbar.update(steps_taken)
        self.last_logged_timestep = self.num_timesteps
        return True

def objective(trial):
    global train_df, eval_df, initial_balance, lookback_window, hpo_train_timesteps

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.02)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    ent_coef_val = trial.suggest_float("ent_coef_val", 0.01, 0.2)
    ent_coef = f'auto_{ent_coef_val:.3f}'
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layer_size = trial.suggest_categorical("layer_size", [128, 256, 512])
    net_arch = [layer_size] * n_layers

    vec_train_env = None # Initialize to None
    vec_eval_env = None  # Initialize to None
    hpo_train_pbar = None # Initialize pbar to None

    try:
        vec_train_env = DummyVecEnv([lambda: CustomTradingEnv(df=train_df,
                                                              initial_balance=initial_balance,
                                                              lookback_window=lookback_window,
                                                              max_episode_length=len(train_df)-lookback_window-1,
                                                              DEBUG=False)])
        vec_train_env = VecNormalize(vec_train_env, norm_obs=True, norm_reward=False, gamma=gamma)

        vec_eval_env = DummyVecEnv([lambda: CustomTradingEnv(df=eval_df,
                                                             initial_balance=initial_balance,
                                                             lookback_window=lookback_window,
                                                             max_episode_length=len(eval_df)-lookback_window-1,
                                                             DEBUG=False)])
        vec_eval_env = VecNormalize(vec_eval_env, training=False, norm_obs=True, norm_reward=False)
        vec_eval_env.obs_rms = vec_train_env.obs_rms
        vec_eval_env.ret_rms = vec_train_env.ret_rms

        model = SAC(
            "MlpPolicy",
            vec_train_env,
            verbose=0,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            batch_size=batch_size,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": net_arch},
            learning_starts=10000,
            gradient_steps=trial.suggest_int("gradient_steps", 1, 4),
            device="cuda",
        )

        # Create and manage the progress bar for this trial's training
        hpo_train_pbar = tqdm(total=hpo_train_timesteps, desc=f"Trial {trial.number} Training", leave=False, position=1)
        hpo_train_callback = HpoTrainProgressCallback(hpo_train_pbar)

        model.learn(total_timesteps=hpo_train_timesteps, callback=hpo_train_callback)
        hpo_train_pbar.close()

        obs = vec_eval_env.reset()
        portfolio_values = [initial_balance]
        done = False
        steps = 0
        max_eval_steps = len(eval_df) - lookback_window - 1

        while not done and steps < max_eval_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_eval_env.step(action)
            actual_info = info[0] if isinstance(info, list) else info
            portfolio_values.append(actual_info.get("total_value", portfolio_values[-1]))
            steps += 1
            if done:
                 break

        final_portfolio_value = portfolio_values[-1]
        total_return = (final_portfolio_value - initial_balance) / initial_balance

        vec_train_env.close()
        vec_eval_env.close()
        del model, vec_train_env, vec_eval_env

        return total_return

    except Exception as e:
        print(f"!!! Trial {trial.number} failed with exception: {e}")
        import traceback
        print(traceback.format_exc())
        if vec_train_env:
            try: vec_train_env.close()
            except: pass
        if vec_eval_env:
            try: vec_eval_env.close()
            except: pass
        if hpo_train_pbar:
            try: hpo_train_pbar.close()
            except: pass
        return -np.inf
    finally:
        if vec_train_env:
            try: vec_train_env.close()
            except: pass
        if vec_eval_env:
            try: vec_eval_env.close()
            except: pass
        if hpo_train_pbar and not hpo_train_pbar.disable:
             try: hpo_train_pbar.close()
             except: pass

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    symbol = "AAPL"
    df = pd.read_csv(f"data/{symbol}.csv")
    n_hpo_trials = 50
    n_envs_final = 4

    print("Loading and preprocessing data...")
    df_raw = pd.read_csv(f"data/{symbol}.csv")
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

    hour = df_raw['timestamp'].dt.hour
    minute = df_raw['timestamp'].dt.minute
    df_raw['time_of_day'] = (hour * 60 + minute) / (24 * 60)
    df_features = df_raw[[ 'open', 'high', 'low', 'close', 'volume', 'time_of_day']].copy()
    df_features = df_features.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float, 'time_of_day': float})

    df_features['ma_10'] = df_features['close'].rolling(window=10).mean()
    df_features['rsi'] = compute_rsi(df_features['close'])
    df_features['momentum'] = df_features['close'] - df_features['close'].shift(10)
    df_features['roc'] = (df_features['close'] / df_features['close'].shift(10) - 1) * 100
    df_features['atr'] = compute_average_true_range(df_features, window=14)
    df_features['bbands_width'] = compute_bollinger_band_width(df_features, window=20)
    df_features['macd'] = compute_macd(df_features['close'])
    df_features['adx'] = compute_adx(df_features, window=14)

    df_features.loc[:, 'rsi'] = df_features['rsi'].fillna(50)
    df_clean = df_features.dropna().reset_index(drop=True)
    print(f"Data shape after cleaning NaNs: {df_clean.shape}")

    total_len = len(df_clean)
    train_size = int(total_len * 0.75)
    eval_size = int(total_len * 0.125)
    test_size = total_len - train_size - eval_size

    train_df = df_clean[:train_size]
    eval_df = df_clean[train_size:train_size + eval_size]
    test_df = df_clean[train_size + eval_size:]

    print(f"Data split: Train={len(train_df)}, Eval={len(eval_df)}, Test={len(test_df)}")

    print("\n--- Starting Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize", study_name=f"SAC_{symbol}_HPO")

    hpo_pbar = tqdm(total=n_hpo_trials, desc="HPO Trials", position=0)

    def hpo_callback(study, trial):
        hpo_pbar.update(1)
        hpo_pbar.set_description(f"HPO Trials (Best Return: {study.best_value:.4f})")

    try:
        study.optimize(objective, n_trials=n_hpo_trials, n_jobs=1, callbacks=[hpo_callback])
    finally:
        hpo_pbar.close()

    print("\n--- Hyperparameter Optimization Finished ---")
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    best_trial = study.best_trial
    print("  Value (Total Return): ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    best_params = best_trial.params
    best_params['policy_kwargs'] = {"net_arch": [best_params.pop('layer_size')] * best_params.pop('n_layers')}
    best_params['ent_coef'] = f'auto_{best_params.pop("ent_coef_val"):.3f}'
    best_params.pop('net_arch_idx', None)

    best_params_filepath = f"models/best_params_SAC_{symbol}.json"
    with open(best_params_filepath, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_filepath}")

    print("\n--- Starting Final Model Training with Best Hyperparameters ---")
    final_train_df = pd.concat([train_df, eval_df]).reset_index(drop=True)
    print(f"Using combined data for final training: {len(final_train_df)} rows")

    def make_final_env():
         return CustomTradingEnv(df=final_train_df,
                                 initial_balance=initial_balance,
                                 lookback_window=lookback_window,
                                 max_episode_length=4000,
                                 DEBUG=False)

    if n_envs_final > 1:
        vec_final_train_env = SubprocVecEnv([make_final_env for _ in range(n_envs_final)])
    else:
        vec_final_train_env = DummyVecEnv([make_final_env])

    vec_final_train_env = VecNormalize(vec_final_train_env,
                                       norm_obs=True,
                                       norm_reward=False,
                                       gamma=best_params['gamma'])

    final_model = SAC(
        "MlpPolicy",
        vec_final_train_env,
        verbose=1,
        device="cuda",
        tensorboard_log="tensorboard_logs/",
        **best_params
    )

    print(f"Training final model for {final_train_timesteps} timesteps...")
    with tqdm(total=final_train_timesteps, desc="Final Training", position=0) as training_pbar:
        class TqdmUpdateCallback(BaseCallback):
            def __init__(self, pbar):
                super().__init__(verbose=0)
                self.pbar = pbar
                self.last_logged_timestep = 0

            def _on_step(self) -> bool:
                steps_taken = self.num_timesteps - self.last_logged_timestep
                self.pbar.update(steps_taken)
                self.last_logged_timestep = self.num_timesteps
                return True

        combined_callbacks = [TensorboardCallback(), TqdmUpdateCallback(training_pbar)]

        final_model.learn(total_timesteps=final_train_timesteps,
                          callback=combined_callbacks,
                          tb_log_name=f"SAC_{symbol}_final_run",
                          log_interval=100)

    model_save_path = f"models/SAC_model_{symbol}_final.zip"
    stats_save_path = f"models/SAC_vecnormalize_{symbol}_final.pkl"
    final_model.save(model_save_path)
    vec_final_train_env.save(stats_save_path)
    print(f"Final model saved to {model_save_path}")
    print(f"Normalization stats saved to {stats_save_path}")

    vec_final_train_env.close()

    print("\n--- Training Complete ---")
