import io
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch
import torch.nn.functional as F

from src.gridworld import GRID_WORD, POMDPGridworld
from src.fsc import obs_to_index

def make_trajectories_grid_gif(
    trajectories,
    gif_path="./plots/trajectories_grid.gif",
    model_order=None,
    size_order=None,
    fps=6,
    cell_scale=3.2,
    wall_map=None
):
    """
    Create a GIF visualizing trajectories of different models and grid sizes in a 3x3 grid format.
    Each cell of the grid corresponds to a specific model (FSC, LSTM, RPPO) and grid size (3x3, 6x6, 9x9).

    Inputs:
    - trajectories: List of 9 trajectory dictionaries, each containing:
        - "model": string, one of "FSC", "LSTM", "RPPO"
        - "size": int, one of 3, 6, 9
        - "positions": list of (x, y) tuples representing the agent's path
        - "goal": optional (x, y) tuple for the goal position (if not provided, defaults to bottom-right corner)
    - gif_path: string, output path for the generated GIF
    - model_order: optional list of model names in the order they should appear in columns (default: sorted order)
    - size_order: optional list of grid sizes in the order they should appear in rows (default: sorted order)
    - fps: frames per second for the GIF
    - cell_scale: scaling factor for the size of each cell in inches (default: 3.2)
    - wall_map: optional dict mapping size to 2D list of 0/1 for free/wall cells (if not provided, uses a default pattern)
    """

    if len(trajectories) != 9:
        raise ValueError(f"Expected exactly 9 trajectories (3 models x 3 sizes), but got {len(trajectories)}.")

    # If model_order or size_order are not provided, infer them from the trajectories
    if model_order is None:
        model_order = sorted(list({t["model"] for t in trajectories}))
    if size_order is None:
        size_order = sorted(list({int(t["size"]) for t in trajectories}))

    if len(model_order) != 3 or len(size_order) != 3:
        raise ValueError("Expected exactly 3 models and 3 sizes.")
    
    # Index trajectories for quick lookup
    lookup = {}
    for t in trajectories:
        key = (int(t["size"]), t["model"])
        if key in lookup:
            raise ValueError(f"Found duplicate trajectories for {key}.")
        if "positions" not in t or len(t["positions"]) == 0:
            raise ValueError(f"Empty or missing positions for {key}.")
        lookup[key] = t

    # Verify complete 3x3 coverage
    for s in size_order:
        for m in model_order:
            if (s, m) not in lookup:
                raise ValueError(f"Missing trajectory for size={s}, model={m}.")

    # GIF duration: maximum number of steps across all trajectories
    max_len = max(len(t["positions"]) for t in trajectories)

    fig, axes = plt.subplots(
        nrows=3, ncols=3, figsize=(cell_scale * 3.2, cell_scale * 3.2), dpi=120
    )

    # Color map: free, wall, goal
    cmap = ListedColormap(["#f7f7f7", "#1f2937", "#22c55e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    frames = []

    for frame_idx in range(max_len):
        for r, size in enumerate(size_order):
            for c, model_name in enumerate(model_order):
                ax = axes[r, c]
                ax.clear()

                tr = lookup[(size, model_name)]
                pos = tr["positions"]
                goal = tr.get("goal", (size - 1, size - 1))

                # Freeze frame if trajectory is shorter than max_len
                idx = min(frame_idx, len(pos) - 1)
                cur_x, cur_y = pos[idx]

                # Base grid: 0 free, 1 wall, 2 goal
                if wall_map is None:
                    # Use GRID_WORD already present in the notebook
                    base = np.array(GRID_WORD[:size, :size], dtype=int)
                else:
                    base = np.array(wall_map[size], dtype=int).copy()

                gx, gy = goal
                if 0 <= gx < size and 0 <= gy < size:
                    base[gy, gx] = 2

                ax.imshow(base, cmap=cmap, norm=norm, origin="upper")

                # Path up to the current frame
                path_xy = np.array(pos[: idx + 1], dtype=float)
                if len(path_xy) > 1:
                    ax.plot(
                        path_xy[:, 0], path_xy[:, 1],
                        color="#2563eb", linewidth=2.0, alpha=0.9
                    )

                # Current agent
                ax.scatter(
                    [cur_x], [cur_y],
                    s=70, c="#ef4444", edgecolors="white", linewidths=0.8, zorder=3
                )

                # If finished early, show "freeze" flag
                if frame_idx >= len(pos) - 1:
                    ax.text(
                        0.98, 0.04, "freeze",
                        transform=ax.transAxes,
                        ha="right", va="bottom",
                        fontsize=8, color="#111827",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5)
                    )

                # Ticks and limits
                ax.set_xlim(-0.5, size - 0.5)
                ax.set_ylim(size - 0.5, -0.5)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])

                # Titles
                ax.set_title(f"{model_name} | {size}x{size}", fontsize=9)
        fig.suptitle(f"Step {frame_idx + 1}/{max_len}", fontsize=12)
        fig.tight_layout()

        # Render frame RGB (backend-agnostic)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.dpi)
        buf.seek(0)
        rgb = imageio.imread(buf)[:, :, :3].copy()
        buf.close()
        frames.append(rgb)

    plt.close(fig)
    plt.close(fig)

    imageio.mimsave(gif_path, frames, fps=fps)
    print(f"GIF saved in: {gif_path}\n")

def _to_xy_tuple(agent_pos):
    # agent_pos in your env is [x, y]
    return (int(agent_pos[0]), int(agent_pos[1]))


def collect_single_fsc_trajectory(fsc_model, grid_size, max_steps):
    """
        Collects a single trajectory from the FSC model interacting with the POMDPGridworld environment.

        Inputs:
        - fsc_model: an instance of the FSC model with parameters theta_pi, theta_p
        - grid_size: int, size of the gridworld (3, 6, or 9)
        - max_steps: int, maximum number of steps to run the episode

        Output:
        - A dictionary containing:
            - "size": int, the grid size
            - "positions": list of (x, y) tuples representing the agent's path
            - "actions": list of int, the actions taken at each step
            - "rewards": list of float, the rewards received at each step
            - "done": bool, whether the episode ended by reaching the goal
            - "truncated": bool, whether the episode ended by reaching max_steps
            - "length": int, the number of steps taken
            - "goal": (x, y) tuple for the goal position
    """
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    fsc_model.eval()
    with torch.no_grad():
        rho = F.softmax(fsc_model.theta_pi, dim=0).cpu().numpy()    # (M,)
        pie = F.softmax(fsc_model.theta_pie, dim=0).cpu().numpy()   # (A, M)
        g = F.softmax(fsc_model.theta_g, dim=0).cpu().numpy()       # (M_next, M_prev, A, Y)

    belief = rho.copy()

    positions = [_to_xy_tuple(env.agent_pos)]
    actions = []
    rewards = []

    done = False
    truncated = False

    while not (done or truncated):
        y_t = obs_to_index(obs)

        # FSC action selection: argmax_a sum_m pi(a|m) * b(m)
        action_probs = pie @ belief
        action = int(np.argmax(action_probs))

        obs, reward, done, truncated, _ = env.step(action)

        actions.append(action)
        rewards.append(float(reward))
        positions.append(_to_xy_tuple(env.agent_pos))

        # Belief update: b'(m') prop sum_m g(m'|m,a,y) * b(m)
        g_slice = g[:, :, action, y_t]  # (M_next, M_prev)
        new_belief_raw = g_slice @ belief
        z = new_belief_raw.sum()
        if z > 1e-12:
            belief = new_belief_raw / z
        else:
            belief = np.ones_like(belief) / len(belief)

    return {
        "size": int(grid_size),
        "positions": positions,
        "actions": actions,
        "rewards": rewards,
        "done": bool(done),
        "truncated": bool(truncated),
        "length": len(actions),
        "goal": (int(grid_size - 1), int(grid_size - 1)),
    }


def collect_single_recurrent_trajectory(model, grid_size, max_steps, kind):
    """
    Collects a single trajectory from a recurrent model (either LSTM or RecurrentPPO) interacting with the POMDPGridworld environment.

    Inputs:
    - model: an instance of the recurrent model (LSTM or RecurrentPPO)
    - grid_size: int, size of the gridworld (3, 6, or 9)
    - max_steps: int, maximum number of steps to run the episode
    - kind: string, either "lstm" or "rppo" to specify the type of model (used for action prediction)

    Output:
    - A dictionary containing:
        - "size": int, the grid size
        - "positions": list of (x, y) tuples representing the agent's path
        - "actions": list of int, the actions taken at each step
        - "rewards": list of float, the rewards received at each step
        - "done": bool, whether the episode ended by reaching the goal
        - "truncated": bool, whether the episode ended by reaching max_steps
        - "length": int, the number of steps taken
        - "goal": (x, y) tuple for the goal position
    """
    env = POMDPGridworld(size=grid_size, max_steps=max_steps)
    obs, _ = env.reset()

    positions = [_to_xy_tuple(env.agent_pos)]
    actions = []
    rewards = []

    done = False
    truncated = False

    kind = kind.lower()

    if kind == "rppo":
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not (done or truncated):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True
            )
            action = int(action)

            obs, reward, done, truncated, _ = env.step(action)
            episode_starts = np.zeros((1,), dtype=bool)

            actions.append(action)
            rewards.append(float(reward))
            positions.append(_to_xy_tuple(env.agent_pos))

    elif kind == "lstm":
        model.eval()
        hidden = None

        while not (done or truncated):
            action, hidden = model.predict_action(obs, hidden=hidden)
            action = int(action)

            obs, reward, done, truncated, _ = env.step(action)

            actions.append(action)
            rewards.append(float(reward))
            positions.append(_to_xy_tuple(env.agent_pos))
    else:
        raise ValueError("kind must be 'rppo' or 'lstm'.")

    return {
        "size": int(grid_size),
        "positions": positions,
        "actions": actions,
        "rewards": rewards,
        "done": bool(done),
        "truncated": bool(truncated),
        "length": len(actions),
        "goal": (int(grid_size - 1), int(grid_size - 1)),
    }


def _get_model_for_size(model_map, size):
    """
    Supports both integer keys (3,6,9) and string keys ('3x3','6x6','9x9').
    """
    if size in model_map:
        return model_map[size]

    key_str = f"{size}x{size}"
    if key_str in model_map:
        return model_map[key_str]

    raise KeyError(f"Model missing for size={size} (tried keys: {size}, '{key_str}').")


def collect_9_trajectories_for_grid_gif(
    fsc_models_by_size,
    lstm_models_by_size,
    rppo_models_by_size,
    sizes=(3, 6, 9),
    max_steps_by_size=None,
    model_order=("FSC", "LSTM", "RPPO"),
):
    """
    Returns a list of 9 trajectories in the format suitable for make_trajectories_grid_gif:
    columns = model_order, rows = sizes.
    """
    if len(sizes) != 3:
        raise ValueError("sizes must contain exactly 3 dimensions.")
    if len(model_order) != 3:
        raise ValueError("model_order must contain exactly 3 models.")

    if max_steps_by_size is None:
        # defaults consistent with your notebook
        max_steps_by_size = {3: 100, 6: 500, 9: 1000}

    output = []

    for size in sizes:
        max_steps = int(max_steps_by_size[size])

        for model_name in model_order:
            if model_name == "FSC":
                model = _get_model_for_size(fsc_models_by_size, size)
                traj = collect_single_fsc_trajectory(model, size, max_steps)

            elif model_name == "LSTM":
                model = _get_model_for_size(lstm_models_by_size, size)
                traj = collect_single_recurrent_trajectory(model, size, max_steps, kind="lstm")

            elif model_name == "RPPO":
                model = _get_model_for_size(rppo_models_by_size, size)
                traj = collect_single_recurrent_trajectory(model, size, max_steps, kind="rppo")

            else:
                raise ValueError(f"Unsupported model name: {model_name}")

            traj["model"] = model_name
            output.append(traj)

    return output