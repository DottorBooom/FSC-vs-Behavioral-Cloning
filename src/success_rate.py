from src.gridworld import POMDPGridworld
import torch
import torch.nn.functional as F
import numpy as np
from src.fsc import obs_to_index

def run_rppo_episode(rppo, grid_size, max_steps=100):
    """
    Run a single episode in the POMDPGridworld environment using the provided RPPO model to select actions.

    Inputs:
    - rppo: An instance of the RPPO model
    - grid_size: Size of the gridworld (e.g., 3, 6, 9)
    - max_steps: Maximum number of steps to run in the episode (default: 100)

    Outputs:
    - total_reward: Total reward accumulated during the episode
    - steps: Number of steps taken in the episode
    - success: Boolean indicating whether the episode ended in success (reaching the goal)
    """
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool) 

    total_reward = 0.0
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        step += 1

        action, lstm_states = rppo.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)

        obs, reward, done, truncated, _ = env.step(action)
        episode_starts = np.zeros((1,), dtype=bool) # Immediately after the first action, set to False
        total_reward += reward

    return total_reward, step, done

def run_fsc_episode(fsc, grid_size, max_steps=100, verbose=False):
    """
    Run a complete episode with the loaded FSC and print, step by step:
    - the current maze
    - the belief over internal states before the action (with ASCII bars)
    - the chosen action and its probabilities
    - the belief over internal states after the transition
    - the most probable transition (m_prev → m_next with probability)
    Soft execution (belief-based):
    - a_t  = argmax_a  Σ_m  b(m) · π(a|m)
    - b_{t+1}(m') ∝  Σ_m  g(m'|m, a_t, y_t) · b(m)

    Inputs:
    - fsc: An instance of the FSC model
    - grid_size: Size of the gridworld (e.g., 3, 6, 9)
    - max_steps: Maximum number of steps to run in the episode (default: 100)
    - verbose: If True, prints detailed step-by-step information (default: False)

    Outputs:
    - total_reward: Total reward accumulated during the episode
    - steps: Number of steps taken in the episode
    - success: Boolean indicating whether the episode ended in success (reaching the goal)
    """
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    fsc.eval()
    with torch.no_grad():
        rho = F.softmax(fsc.theta_pi,  dim=0).numpy()  # (M,)
        pie = F.softmax(fsc.theta_pie, dim=0).numpy()  # (A, M)
        g   = F.softmax(fsc.theta_g,   dim=0).numpy()  # (M_next, M_prev, A, Y)

    def print_belief(belief, label):
        print(f"\n  {label}:")
        for m in range(fsc.M):
            bar = '█' * int(belief[m] * 24)
            print(f"    m{m}: {belief[m]:.3f}  {bar}")
        print(f"    → Most probable state: m{int(np.argmax(belief))}")

    belief = rho.copy()
    total_reward = 0.0
    done = False
    truncated = False
    step = 0
    ACTION_NAMES = ['Up', 'Right', 'Down', 'Left']

    while not (done or truncated):
        step += 1
        y_t = obs_to_index(obs)

        action_probs = pie @ belief            # P(a) = Σ_m π(a|m)·b(m)  shape (A,)
        action = int(np.argmax(action_probs))

        if verbose:
            print(f"\n{'━'*46}")
            print(f"  STEP {step:3d}  |  obs={obs.tolist()} (y={y_t:>2d})")
            print(f"{'━'*46}")
            env.visualize_grid()
            print_belief(belief, "Belief BEFORE action")
            print(f"\n  Chosen action : [{action}] {ACTION_NAMES[action]}")
            print(f"  Action probabilities : "
                + "  ".join(f"{ACTION_NAMES[a]}={action_probs[a]:.2f}" for a in range(fsc.A)))

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

        # ── Update the belief: b'(m') ∝ Σ_m g(m'|m, a, y) · b(m) ───────────
        g_slice          = g[:, :, action, y_t]       # (M_next, M_prev)
        new_belief_raw   = g_slice @ belief             # (M_next,)
        z                = new_belief_raw.sum()
        new_belief       = new_belief_raw / z if z > 1e-12 else np.ones(fsc.M) / fsc.M

        if verbose:
            print_belief(new_belief, "Belief AFTER transition")

            m_prev = int(np.argmax(belief))
            m_next = int(np.argmax(new_belief))
            p_transition = float(g_slice[m_next, m_prev])
            print(f"\n  Most probable transition : m{m_prev} → m{m_next}  "
                f"(g={p_transition:.3f})")
            print(f"  Reward: {reward:+.3f}  |  "
                f"{'GOAL REACHED!' if done else 'Truncated' if truncated else 'Continuing...'}")

        belief = new_belief

    if verbose:
        print(f"\n{'━'*46}")
        print(f"  End of episode  |  Step: {step}  |  Total reward: {total_reward:.3f}")
        print(f"  Outcome: {'✓ Success' if done else '✗ Truncated (goal not reached)'}")
        print(f"{'━'*46}\n")

    return total_reward, step, done

def run_lstm_episode(lstm, grid_size, max_steps=100):
    """
    Run a single episode in the POMDPGridworld environment using the provided LSTM model to select actions.

    Inputs:
    - lstm: An instance of the LSTM model
    - grid_size: Size of the gridworld (e.g., 3, 6, 9)
    - max_steps: Maximum number of steps to run in the episode (default: 100)

    Outputs:
    - total_reward: Total reward accumulated during the episode
    - steps: Number of steps taken in the episode
    - success: Boolean indicating whether the episode ended in success (reaching the goal)
    """
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    lstm.eval()

    total_reward = 0.0
    done = False
    truncated = False
    step = 0

    while not (done or truncated):
        step += 1

        action = lstm.predict_action(obs)[0]

        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward

    return total_reward, step, done