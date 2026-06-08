import os
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import no_grad
from PIL import Image
import torch.nn.functional as F

def plot_tensorboard_logs(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No file found in {log_dir}")
        return

    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    def get_data(tag):
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            return [e.step for e in events], [e.value for e in events]
        return [], []

    metrics = [
        ('eval/mean_reward', 'Eval: Mean Reward', '#ff7f0e'),
        ('rollout/ep_rew_mean', 'Rollout: Mean Reward', '#1f77b4'),
        ('eval/mean_ep_length', 'Eval: Mean Episode Length', '#ff7f0e'),
        ('rollout/ep_len_mean', 'Rollout: Mean Episode Length', '#1f77b4'),
        ('train/approx_kl', 'Train: Approx KL (Stability)', '#2ca02c'),
        ('train/entropy_loss', 'Train: Entropy Loss (Exploration)', '#9467bd'),
        ('train/value_loss', 'Train: Value Loss (Reward Estimation Error)', '#d62728'),
    ]

    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    axs = axs.flatten()

    for idx, (tag, title, color) in enumerate(metrics):
        steps, values = get_data(tag)
        if steps:
            axs[idx].plot(steps, values, color=color, linewidth=1.5, alpha=0.4, label='Raw Data')
            
            if len(values) > 1:
                weight = 0.7  
                smoothed = []
                last = values[0]
                for v in values:
                    smoothed_val = last * weight + (1 - weight) * v
                    smoothed.append(smoothed_val)
                    last = smoothed_val
                
                axs[idx].plot(steps, smoothed, color=color, linewidth=2.5, label='Trend (EMA)')
            
            axs[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
            axs[idx].set_xlabel('Timesteps', fontsize=12)
            axs[idx].set_ylabel('Value', fontsize=12)
            axs[idx].grid(True, linestyle='--', alpha=0.7)
            axs[idx].legend()
        else:
            axs[idx].text(0.5, 0.5, f"Data not found:\n{tag}", ha='center', va='center', color='red')
            axs[idx].set_title(title)

    fig.delaxes(axs[7])

    plt.tight_layout()
    
    if not os.path.exists(f'./plots/RecurrentPPO/'):
        os.makedirs('./plots/RecurrentPPO', exist_ok=True)
    if os.path.exists(f'./plots/RecurrentPPO/{os.path.basename(log_dir)}.png'):
        os.remove(f'./plots/RecurrentPPO/{os.path.basename(log_dir)}.png')
    plt.savefig(f'./plots/RecurrentPPO/{os.path.basename(log_dir)}.png')
    plt.close()

def plot_fsc_training_history(size, Ms, losses):

    best_r = int(np.argmin(losses))
    plt.figure(figsize=(10, 4))

    plt.plot(Ms, losses, marker='o', linewidth=2, color='#1f77b4', label='Best NLL')
    plt.scatter([Ms[best_r]], [losses[best_r]], color='red', zorder=5, s=100, label=f'Best: M={Ms[best_r]}')

    plt.xlabel("Number of internal states M")
    plt.ylabel("Average NLL")
    plt.title(f"FSC {size}x{size} - Best final loss for each M parameter")
    plt.xticks(Ms)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if not os.path.exists(f'./plots/FSC/'):
        os.makedirs('./plots/FSC', exist_ok=True)
    if os.path.exists(f'./plots/FSC/FSC_curve_{size}x{size}.png'):
        os.remove(f'./plots/FSC/FSC_curve_{size}x{size}.png')
    plt.savefig(f'./plots/FSC/FSC_curve_{size}x{size}.png')
    plt.close()

import networkx as nx

def plot_fsc_graph(fsc, size, action_names=None, threshold=0.05, figsize=(16, 7)):
    """
    Visualize the trained FSC as a directed graph + policy heatmap.
    Left panel - transition graph:
    - Nodes : internal states m_0..m_{M-1}
            size ∝ rho(m)  (initial distribution)
            color     = dominant action argmax_a π(a|m)
    - Edges : g(m'|m) marginalized uniformly over A and Y
            width ∝ transition probability
            only edges with p > threshold

    Right panel - heatmap π(a|m):
    shows the probability distribution of each action for each state.

    Input:
    - fsc: The trained FSC model to visualize
    - size: Size of the gridworld (used for naming the saved model)
    - action_names: Optional list of action names for labeling the heatmap (default: ['Up', 'Right', 'Down', 'Left'])
    - threshold: Minimum transition probability to display an edge in the graph
    - figsize: Size of the overall figure (width, height)
    - title: Title for the graph panel
    """
    if action_names is None:
        action_names = ['Up', 'Right', 'Down', 'Left']

    action_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

    fsc.eval()
    with no_grad():
        rho = F.softmax(fsc.theta_pi,  dim=0).numpy()   
        pie = F.softmax(fsc.theta_pie, dim=0).numpy()   
        g   = F.softmax(fsc.theta_g,   dim=0).numpy()   

    M = fsc.M
    dominant_action = np.argmax(pie, axis=0)   
    g_marginal = g.mean(axis=(2, 3))

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                            gridspec_kw={'width_ratios': [2.5, 1]})

    # ── Left panel: transition graph ─────────────────────────────
    ax_graph = axes[0]

    G = nx.DiGraph()
    G.add_nodes_from(range(M))
    for m_prev in range(M):
        for m_next in range(M):
            w = float(g_marginal[m_next, m_prev])
            if w > threshold:
                G.add_edge(m_prev, m_next, weight=w)

    pos         = nx.spring_layout(G, seed=42, k=2.5)
    node_sizes  = [5000 * rho[m] + 500 for m in range(M)]
    node_colors = [action_colors[dominant_action[m]] for m in range(M)]
    edge_data   = list(G.edges(data=True))
    edge_widths = [d['weight'] * 14 for _, _, d in edge_data]

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors,
                        alpha=0.92, ax=ax_graph)
    nx.draw_networkx_labels(G, pos,
                            labels={m: f"m{m}" for m in range(M)},
                            font_color='white', font_weight='bold',
                            font_size=10, ax=ax_graph)
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.55,
                        edge_color='#444444', arrows=True, arrowsize=20,
                        connectionstyle='arc3,rad=0.15', ax=ax_graph)

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in edge_data}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                font_size=7, alpha=0.8, ax=ax_graph)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
                markerfacecolor=action_colors[a], markersize=13,
                label=f"{action_names[a]}")
        for a in range(fsc.A)
    ]
    ax_graph.legend(handles=legend_elements, loc='upper left',
                    fontsize=9, title='Dominant action π(a|m)')
    ax_graph.set_title(
        f"FSC {size}x{size} - Graph of Internal States for M={M}\n"
        r"(edges = $\bar{g}(m'|m)$ over A,Y  ·  node ∝ $\rho(m_0)$  ·  color = dominant action)",
        fontsize=11, fontweight='bold'
    )
    ax_graph.axis('off')
 
    # ── Right panel: heatmap π(a|m) ───────────────────────────────────────
    ax_heat = axes[1]
    im = ax_heat.imshow(pie, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    ax_heat.set_xticks(range(M))
    ax_heat.set_xticklabels([f"m{m}" for m in range(M)], fontsize=8)
    ax_heat.set_yticks(range(fsc.A))
    ax_heat.set_yticklabels(action_names, fontsize=9)
    ax_heat.set_xlabel("Internal state m", fontsize=9)
    ax_heat.set_title("π(a | m)", fontsize=12, fontweight='bold')

    for a in range(fsc.A):
        for m in range(M):
            ax_heat.text(m, a, f"{pie[a, m]:.2f}",
                        ha='center', va='center', fontsize=7,
                        color='white' if pie[a, m] > 0.5 else 'black')

    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label='Probability')
    plt.tight_layout()
    
    if not os.path.exists(f'./plots/FSC/'):
        os.makedirs('./plots/FSC', exist_ok=True)
    if os.path.exists(f'./plots/FSC/FSC_graph_{size}x{size}_M{M}.png'):
        os.remove(f'./plots/FSC/FSC_graph_{size}x{size}_M{M}.png')
    plt.savefig(f'./plots/FSC/FSC_graph_{size}x{size}_M{M}.png')

    plt.close()

def plot_lstm_training_history(size, history_train, history_val):
    
    plt.figure(figsize=(10, 5))

    plt.plot(history_train, label='Train Loss', alpha=0.6, linewidth=2)
    plt.plot(history_val, label='Val Loss', alpha=0.6, linewidth=2, linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"LSTM {size}x{size} - Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if not os.path.exists(f'./plots/LSTM/'):
        os.makedirs('./plots/LSTM', exist_ok=True)
    if os.path.exists(f'./plots/LSTM/LSTM_curve_{size}x{size}.png'):
        os.remove(f'./plots/LSTM/LSTM_curve_{size}x{size}.png')
    plt.savefig(f'./plots/LSTM/LSTM_curve_{size}x{size}.png')
    plt.close()

def plot_model_comparison_bars(
    results_rppo,
    results_fsc,
    results_lstm,
    sizes=(3, 6, 9),
    save_dir="./plots/Comparison",
):
    """
    Plot a grouped bar chart comparing RPPO, FSC, and LSTM across different grid sizes for:
    - Success Rate (%)
    - Average Reward
    - Average Steps

    Inputs:
    - results_rppo: Dictionary with RPPO results for each size (keys: size, values: dict with 'success_rate', 'average_reward', 'average_steps')
    - results_fsc: Dictionary with FSC results for each size (same structure as results_rppo)
    - results_lstm: Dictionary with LSTM results for each size (same structure as results_rppo)
    - sizes: Tuple of grid sizes to include in the plot (default: (3, 6, 9))
    - save_dir: Directory to save the resulting plot (default: "./plots/Comparison")

    Outputs:
    - A saved PNG file with the comparison bar chart in the specified directory.
    """

    def _get_entry(results, size):
        for key in (size, str(size), f"{size}x{size}"):
            if key in results:
                return results[key]
        raise KeyError(f"Missing results for size={size}. Available keys: {list(results.keys())}")

    metrics = [
        ("success_rate", "Success rate (%)", "Success Rate"),
        ("average_reward", "Average reward", "Average Reward"),
        ("average_steps", "Average steps", "Average Steps"),
    ]

    model_specs = [
        ("RPPO", results_rppo, "#056875"), 
        ("FSC", results_fsc, "#50C2E5"),
        ("LSTM", results_lstm, "#C9495E"),
    ]

    x = np.arange(len(sizes))
    
    width = 0.20 

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        for offset, (model_name, results, color) in zip(
            (-width, 0.0, width),
            model_specs
        ):
            values = [float(_get_entry(results, size)[metric_key]) for size in sizes]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=color,
                alpha=0.9,
                label=model_name,
                edgecolor="white", 
                linewidth=0.5
            )

            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        
        ax.margins(y=0.15) 
        
        if metric_key == "success_rate":
            ax.set_ylim(0, 115)

    axes[0].legend(loc="best")

    plt.suptitle("RPPO vs FSC vs LSTM across grid sizes", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, "model_comparison_success_reward_steps.png")
    if os.path.exists(output_path):
        os.remove(output_path)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to: {output_path}")

def plot_inference_time_comparison(
    times_fsc,
    times_lstm,
    sizes=(3, 6, 9),
    save_dir="./plots/Comparison",
):
    """
    Plot a line graph comparing inference times of FSC and LSTM across different grid sizes.

    Inputs:
    - times_fsc: Dictionary with FSC inference times for each size (keys: size, values: time in seconds)
    - times_lstm: Dictionary with LSTM inference times for each size (same structure as times_fsc)
    - sizes: Tuple of grid sizes to include in the plot (default: (3, 6, 9))
    - save_dir: Directory to save the resulting plot (default: "./plots/Comparison")

    Outputs:
    - A saved PNG file with the inference time comparison line graph in the specified directory.
    """

    def _get_entry(results, size):
        for key in (size, str(size), f"{size}x{size}"):
            if key in results:
                return results[key]
        raise KeyError(f"Missing results for size={size}. Available keys: {list(results.keys())}")

    fsc_values = [float(_get_entry(times_fsc, size)) for size in sizes]
    lstm_values = [float(_get_entry(times_lstm, size)) for size in sizes]
    
    x_labels = [f"{size}x{size}" for size in sizes]
    
    color_fsc = "#50C2E5"
    color_lstm = "#C9495E"

    plt.figure(figsize=(8, 6))
    
    plt.plot(x_labels, fsc_values, marker='o', markersize=8, color=color_fsc, linewidth=2.5, label='FSC')
    plt.plot(x_labels, lstm_values, marker='s', markersize=8, color=color_lstm, linewidth=2.5, label='LSTM')
    
    plt.title("Inference Time per Action across Grid Sizes", fontsize=14, fontweight="bold")
    plt.xlabel("Maze Size", fontsize=12)
    plt.ylabel("Time (seconds)", fontsize=12)
    
    plt.yscale('log')
    
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    
    plt.legend(loc="best", fontsize=11)
    
    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, "inference_time_comparison.png")
    if os.path.exists(output_path):
        os.remove(output_path)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved inference time plot to: {output_path}")

def plot_robustness_bars(
    results_fsc,
    results_lstm,
    randomness,
    sizes=(3, 6, 9),
    save_dir="./plots/Comparison",
    ):
    """
    Plot a grouped bar chart comparing FSC and LSTM across different grid sizes for:
    - Success Rate (%)
    - Average Reward
    - Average Steps

    Inputs:
    - results_fsc: Dictionary with FSC results for each size (keys: size, values: dict with 'success_rate', 'average_reward', 'average_steps')
    - results_lstm: Dictionary with LSTM results for each size (same structure as results_fsc)
    - sizes: Tuple of grid sizes to include in the plot (default: (3, 6, 9))
    - save_dir: Directory to save the resulting plot (default: "./plots/Comparison")

    Outputs:
    - A saved PNG file with the comparison bar chart in the specified directory.
    """

    def _get_entry(results, size):
        for key in (size, str(size), f"{size}x{size}"):
            if key in results:
                return results[key]
        raise KeyError(f"Missing results for size={size}. Available keys: {list(results.keys())}")

    metrics = [
        ("success_rate", "Success rate (%)", "Success Rate"),
        ("average_reward", "Average reward", "Average Reward"),
        ("average_steps", "Average steps", "Average Steps"),
    ]

    model_specs = [
        ("FSC", results_fsc, "#50C2E5"),
        ("LSTM", results_lstm, "#C9495E"),
    ]

    x = np.arange(len(sizes))
    
    width = 0.20 

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    for ax, (metric_key, ylabel, title) in zip(axes, metrics):
        for offset, (model_name, results, color) in zip(
            (-width, 0.0, width),
            model_specs
        ):
            values = [float(_get_entry(results, size)[metric_key]) for size in sizes]
            bars = ax.bar(
                x + offset,
                values,
                width=width,
                color=color,
                alpha=0.9,
                label=model_name,
                edgecolor="white", 
                linewidth=0.5
            )

            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(size) for size in sizes])
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        
        ax.margins(y=0.15) 
        
        if metric_key == "success_rate":
            ax.set_ylim(0, 115)

    axes[0].legend(loc="best")

    plt.suptitle(f"FSC vs LSTM Robustness across grid sizes (Randomness: {randomness})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    output_path = os.path.join(save_dir, f"model_robustness_randomness_{randomness}.png")
    if os.path.exists(output_path):
        os.remove(output_path)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot to: {output_path}")
