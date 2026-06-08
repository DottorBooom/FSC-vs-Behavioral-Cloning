from src.gridworld import POMDPGridworld
from src.recurrent_ppo import load_rppo

import numpy as np
import random
import pickle
import os

def collect_trajectories(model_path, grid_size, max_steps, n_episodes_to_collect):
    '''
    Collect and save trajectories from a trained RecurrentPPO model in the POMDPGridworld environment.

    Inputs:
    - model_path: Path to the trained RecurrentPPO model (should be a .zip file)
    - grid_size: Size of the gridworld (e.g., 3, 6, 9)
    - max_steps: Maximum steps per episode in the environment
    - n_episodes_to_collect: Number of episodes (trajectories) to collect

    Output:
    - True if trajectories were successfully collected and saved, False otherwise.
    - Trajectories are saved in a local file named 'oracle_dataset_{grid_size}x{grid_size}.pkl' in the './data/' directory.
    '''
    # 1. Load the model

    if model_path is not None:
        try:
            env = POMDPGridworld(size=grid_size, max_steps=max_steps)
            model = load_rppo(path=model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model from {model_path}: {e}")
            
    else:
        raise ValueError("No path provided for loading the model.")

    # 2. Play and save trajectories

    trajectories = []
    collected = 0

    while collected < n_episodes_to_collect:

        obs, _ = env.reset()

        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool) 

        ep_obs = []
        ep_actions = []
        ep_rewards = []

        done = False
        truncated = False

        while not (done or truncated):
            ep_obs.append(obs)

            # 2. Request action from RecurrentPPO passing LSTM_states and Dones
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            
            if random.random() < 0.25: 
                action = env.action_space.sample() 
            obs, reward, done, truncated, _ = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool) # Immediately after the first action, set to False
            
            ep_actions.append(action)
            ep_rewards.append(reward)

        # 3. Save the data
        if done:
            trajectories.append({
                'observations': np.array(ep_obs),
                'actions': np.array(ep_actions),
                'rewards': np.array(ep_rewards),
                'length': len(ep_obs)
            })
            collected += 1
            if collected % 100 == 0:
                print(f"Collected {collected}/{n_episodes_to_collect} trajectories...")

    print(f"\nCollection completed for {grid_size}x{grid_size}. Average trajectory length: {np.mean([t['length'] for t in trajectories]):.2f} steps.")
    
    # Save the dataset in local file
    if trajectories:
        if not os.path.exists(f'./data/'):
            os.makedirs('./data', exist_ok=True)
        with open(f'./data/Trajectories_{grid_size}x{grid_size}.pkl', 'wb') as f:
            pickle.dump(trajectories, f)
    else:
        raise RuntimeError(f"No trajectories collected for size={grid_size}x{grid_size}. Dataset not saved.")