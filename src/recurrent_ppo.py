from src.gridworld import POMDPGridworld

import os
import shutil
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor 

def train_oracle(size, max_steps, total_timesteps=50000, verbose_level=1, lr=3e-4, SEED=42):
    '''
    Trains a RecurrentPPO agent on the POMDPGridworld environment with early stopping based on 
    evaluation performance.

    Inputs:
    - size: Size of the gridworld
    - max_steps: Maximum steps per episode
    - total_timesteps: Total timesteps for training
    - verbose_level: 0 (no logs), 1 (callback logs), 2 (full logs)
    - lr: Learning rate for the PPO agent
    - SEED: Random seed for reproducibility

    Output:
    - model: The trained RecurrentPPO model
    '''

    # 0: Set verbose levels
    if verbose_level == 0:
        verbose_learn = 0
        verbose_callback = 0
    elif verbose_level == 1:
        verbose_learn = 0
        verbose_callback = 1
    elif verbose_level == 2:
        verbose_learn = 1
        verbose_callback = 1
    else:
        raise ValueError(f"Invalid verbose level: {verbose_level}. Choose 0 (no logs), 1 (callback logs), or 2 (full logs).")

    # 1. Initialize the environment and wrap it with Monitor
    env = Monitor(POMDPGridworld(size=size, max_steps=max_steps))
    eval_env = Monitor(POMDPGridworld(size=size, max_steps=max_steps))

    # 2. Automatic check to ensure the environment adheres to standards
    check_env(env)
    check_env(eval_env)

    print("Environment verified! Proceeding with agent creation...")

    # 3. Create the RecurrentPPO model (Use MlpLstmPolicy to include LSTM memory)
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        env, 
        verbose=verbose_learn,         
        learning_rate=lr, 
        n_steps=256,        
        seed=SEED,           
        tensorboard_log="./tensorboard_logs/"
    )

    # 4. Set up the evaluation callback with early stopping based on model improvement
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5, # Stop if the reward does not improve for 5 consecutive evaluations
        min_evals=5,                # Perform at least 5 evaluations before stopping
        verbose=verbose_callback
    )

    if not os.path.exists(f'./models/'):
        os.makedirs('./models', exist_ok=True)

    # Callback to evaluate the agent every 1000 steps. If it's the "best model", it saves it automatically!
    eval_callback = EvalCallback(
        eval_env,
        eval_freq=1000, 
        callback_after_eval=stop_train_callback, # Attach the early stop control
        best_model_save_path=f'./models/RecurrentPPO_{size}_{max_steps}_{lr}/',
        verbose=verbose_callback,
        deterministic=False # Disable deterministic choice to avoid loops at the beginning
    )

    # 5. Train the model
    print("Starting oracle training with Early Stop... \n")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=False)

    try:
        if os.path.isdir(f'./tensorboard_logs/RecurrentPPO_{size}_{max_steps}_{lr}'):
            shutil.rmtree(f'./tensorboard_logs/RecurrentPPO_{size}_{max_steps}_{lr}')
        os.rename('./tensorboard_logs/RecurrentPPO_1', f'./tensorboard_logs/RecurrentPPO_{size}_{max_steps}_{lr}')
    except FileNotFoundError:
        print("File not found.")
    except FileExistsError:
        print("Destination file already exists.")
    except PermissionError:
        print("Permission denied.")

    return model

def load_rppo(path):
    '''
    Loads the best model from the specified path.

    Input:
    -path: The file path to the saved model (e.g., './models/RecurrentPPO_6_500_0.0003/best_model.zip')

    Output:
    - model: The loaded RecurrentPPO model, or None if loading failed
    '''
    
    if path is not None:
        try:
            model = RecurrentPPO.load(path)
            return model
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            raise
    else:
        print("No path provided for loading the model.")
        raise