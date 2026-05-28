import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def plot_tensorboard_logs(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        print(f"No file found in {log_dir}")
        return

    # Initialize the accumulator to read the logs
    ea = EventAccumulator(event_files[0])
    ea.Reload()
    
    # Helper function to extract steps and values for a given tag
    def get_data(tag):
        if tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            return [e.step for e in events], [e.value for e in events]
        return [], []

    # Requested metrics and plot configuration (tag, plot_title, color)
    metrics = [
        ('eval/mean_reward', 'Eval: Mean Reward', '#ff7f0e'),
        ('rollout/ep_rew_mean', 'Rollout: Mean Reward', '#1f77b4'),
        ('eval/mean_ep_length', 'Eval: Mean Episode Length', '#ff7f0e'),
        ('rollout/ep_len_mean', 'Rollout: Mean Episode Length', '#1f77b4'),
        ('train/approx_kl', 'Train: Approx KL (Stability)', '#2ca02c'),
        ('train/entropy_loss', 'Train: Entropy Loss (Exploration)', '#9467bd'),
        ('train/value_loss', 'Train: Value Loss (Reward Estimation Error)', '#d62728'),
    ]

    # Initialize a 4x2 grid to host the 7 plots (the last one will remain empty)
    fig, axs = plt.subplots(4, 2, figsize=(16, 20))
    axs = axs.flatten()

    for idx, (tag, title, color) in enumerate(metrics):
        steps, values = get_data(tag)
        if steps:
            # Plot the raw data
            axs[idx].plot(steps, values, color=color, linewidth=1.5, alpha=0.4, label='Raw Data')
            
            # Exponential Moving Average (EMA) to align the X-axis and replicate TensorBoard
            if len(values) > 1:
                weight = 0.7  # Smoothing factor (0 = raw, 0.9 = very flat)
                smoothed = []
                last = values[0]
                for v in values:
                    smoothed_val = last * weight + (1 - weight) * v
                    smoothed.append(smoothed_val)
                    last = smoothed_val
                
                # Now steps and smoothed have exactly the same length!
                axs[idx].plot(steps, smoothed, color=color, linewidth=2.5, label='Trend (EMA)')
            
            axs[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
            axs[idx].set_xlabel('Timesteps', fontsize=12)
            axs[idx].set_ylabel('Value', fontsize=12)
            axs[idx].grid(True, linestyle='--', alpha=0.7)
            axs[idx].legend()
        else:
            axs[idx].text(0.5, 0.5, f"Data not found:\n{tag}", ha='center', va='center', color='red')
            axs[idx].set_title(title)

    # Remove the 8th plot (bottom right) because it's extra
    fig.delaxes(axs[7])

    plt.tight_layout()
    plt.show()