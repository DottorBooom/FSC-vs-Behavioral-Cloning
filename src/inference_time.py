import time
import torch
import torch.nn.functional as F
import numpy as np

from src.fsc import obs_to_index
from src.gridworld import POMDPGridworld

def inference_time_fsc(fsc, grid_size, max_steps=100, n_episodes=100):
    '''
    Compute the average inference time per action for the given FSC model in the POMDPGridworld environment.

    Inputs:
    - fsc: An instance of the FSC model
    - grid_size: Size of the gridworld (e.g., 3, 6 or 9)
    - max_steps: Maximum number of steps to run in each episode (default: 100)
    - n_episodes: Number of episodes to average over (default: 100)

    Output:
    - Average inference time per action in seconds (float)
    '''
    
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    fsc.eval()
    with torch.no_grad():
        rho = F.softmax(fsc.theta_pi,  dim=0).numpy()  # (M,)
        pie = F.softmax(fsc.theta_pie, dim=0).numpy()  # (A, M)
        g   = F.softmax(fsc.theta_g,   dim=0).numpy()  # (M_next, M_prev, A, Y)

    belief = rho.copy()
    done = False
    truncated = False
    tot_time = 0.0

    while not (done or truncated):
        start = time.time()
        y_t = obs_to_index(obs)

        action_probs = pie @ belief            
        action = int(np.argmax(action_probs))
        tot_time += time.time() - start

        obs, _, done, truncated, _ = env.step(action)

        g_slice          = g[:, :, action, y_t]       
        new_belief_raw   = g_slice @ belief             
        z                = new_belief_raw.sum()
        new_belief       = new_belief_raw / z if z > 1e-12 else np.ones(fsc.M) / fsc.M

        belief = new_belief

    return tot_time

def inference_time_lstm(lstm, grid_size, max_steps=100, n_episodes=100):
    '''
    Compute the average inference time per action for the given LSTM model in the POMDPGridworld environment.

    Inputs:
    - lstm: An instance of the LSTM model
    - grid_size: Size of the gridworld (e.g., 3, 6 or 9)
    - max_steps: Maximum number of steps to run in each episode (default: 100)
    - n_episodes: Number of episodes to average over (default: 100)

    Output:
    - Average inference time per action in seconds (float)
    '''
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    lstm.eval()

    tot_time = 0.0
    done = False
    truncated = False

    while not (done or truncated):

        start = time.time()
        action = lstm.predict_action(obs)[0]
        tot_time += time.time() - start

        obs, _, done, truncated, _ = env.step(action)

    return tot_time