import src.recursive_ppo as rppo
import src.plot as plot
from src.set_seed import set_seed

if __name__ == "__main__":

    # 0. Set a global seed for reproducibility
    SEED = 42
    set_seed(SEED)

    # 1. Train the oracle agent with early stopping on different maze size
    # IMPORTANT: I already tasted different combination and those are the ones that seem to work best for each size.
    parameters = [
        (3, 100, 0.0003, 1),   # size, max_steps, learning_rate, verbose_level
        (6, 500, 0.0003, 1),
        (9, 1000, 0.0005, 1)
    ]

    # If you want to plot the logs, set this flag to True. After each training, the logs will be automatically 
    # plotted and saved in the same folder as the logs (./tensorboard_logs/).
    wanna_plot = True

    # This loop will go throw the list of parameters, train each agent, and plot the logs if requested. 
    for size, max_steps, lr, verbose in parameters:
        print(f"\n\n=== Training Oracle Agent for size={size}, max_steps={max_steps}, lr={lr} ===")
        model = rppo.train_oracle(size=size, max_steps=max_steps, lr=lr, verbose_level=verbose)

        if wanna_plot:
            log_folder = f'./tensorboard_logs/RecurrentPPO_{size}_{max_steps}_{lr}'
            plot.plot_tensorboard_logs(log_folder)