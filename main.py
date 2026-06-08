import src.recurrent_ppo as rppo
import src.plot as plot

from src.set_seed import set_seed
from src.generate_trajectories import collect_trajectories
from src.fsc import train_fsc, load_fsc
from src.lstm import train_lstm, load_lstm
from src.make_gif import collect_9_trajectories_for_grid_gif, make_trajectories_grid_gif
from src.recurrent_ppo import load_rppo
from src.success_rate import run_fsc_episode, run_rppo_episode, run_lstm_episode
from src.inference_time import inference_time_fsc, inference_time_lstm
from src.robustness import robustness_fsc, robustness_lstm

import sys
import pickle

if __name__ == "__main__":

    trainPPO = '--trainPPO' in sys.argv
    trainFSC = '--trainFSC' in sys.argv
    trainLSTM = '--trainLSTM' in sys.argv
    wanna_plot = '--plot' in sys.argv
    collect_trajectories_flag = '--collect' in sys.argv
    
    # 0. Set a global seed for reproducibility
    SEED = 42
    set_seed(SEED)

    # 1. Train the Recursive PPO agent with early stopping on different maze size
    # IMPORTANT: I already tasted different combination and those are the ones that seem to work best for each size.
    parameters = [
        (3, 100, 0.0003, 1, 200), # size, max_steps, learning_rate, verbose_level, n_episodes_to_collect
        (6, 500, 0.0003, 1, 1500),
        (9, 1000, 0.0005, 1, 3000)
    ]

    # This loop will go throw the list of parameters, train each agent, and plot the logs if requested. 
    if trainPPO: 
        for size, max_steps, lr, verbose, n_episodes in parameters:
            print(f"=== Training Recursive PPO Agent for size={size}, max_steps={max_steps}, lr={lr} === \n")
            model = rppo.train_oracle(size=size, max_steps=max_steps, lr=lr, verbose_level=verbose)

            if wanna_plot:
                print(f"\n=== Plotting Recurrent PPO training logs for size={size}x{size} ===")
                log_folder = f'./tensorboard_logs/RecurrentPPO_{size}_{max_steps}_{lr}'
                plot.plot_tensorboard_logs(log_folder)

    # 2. Collect trajectories for the trained model. if requested, otherwise load them from disk.
            if collect_trajectories_flag:
                print(f"\n== Collecting trajectories for {size}x{size} (Target: {n_episodes} episodes) ===\n")
                collect_trajectories(model_path=f'./models/RecurrentPPO_{size}_{max_steps}_{lr}/best_model.zip', grid_size=size, max_steps=max_steps, n_episodes_to_collect=n_episodes)
                print(f"Trajectories collected and saved for size={size}x{size}.\n")
    elif collect_trajectories_flag:
        print("Recurrent PPO training skipped. Models are gonna be loaded from disk for trajectory collection.\n")

        for size, max_steps, lr, _, n_episodes in parameters:
            print(f"== Collecting trajectories for {size}x{size} (Target: {n_episodes} episodes) ===\n")
            collect_trajectories(model_path=f'./models/RecurrentPPO_{size}_{max_steps}_{lr}/best_model.zip', grid_size=size, max_steps=max_steps, n_episodes_to_collect=n_episodes)
            print(f"Trajectories collected and saved for size={size}x{size}.\n")
    else:
        print("=== No training or trajectory collection requested for Recurrent PPO. ===\n")

    # 3. Train the FSC on the collected trajectories for each size with different M values (number of states in the FSC).
    parameters = [
        (3, "./data/Trajectories_3x3.pkl"),
        (6, "./data/Trajectories_6x6.pkl"),
        (9, "./data/Trajectories_9x9.pkl")
    ]
    parameters_fsc = [
        (1,2,3,4,5,6,7,8,9), # M for 3x3
        (1,2,4,8,16), # M for 6x6
        (1,2,4,8,16) # M for 9x9
    ]

    if trainFSC:
        for (size, path), M_values in zip(parameters, parameters_fsc):

            all_fsc = [] # Save all models
            all_losses = [] # save all best losses for each M
            all_histories = [] # Save all training histories for each M

            for M in M_values:
                print(f"=== Training FSC for size={size}x{size} with M={M} ===\n")
                with open(path, 'rb') as f:
                    dataset = pickle.load(f)
                fsc, best_loss, history = train_fsc(dataset,
                                               size = size,
                                               M = M,
                                               A = 4, # Number of actions
                                               Y = 16, # Number of possible observations 
                                               n_restarts = 5,
                                               max_epochs = 500,
                                               lr = 1e-2, # Default learning rate
                                               patience = 30,
                                               batch_size = 32,
                                               verbose = True,
                                               )
                
                all_losses.append(best_loss)
                all_histories.append(history)
                all_fsc.append(fsc)

    # 4. Plot the training history and the graphs of the FSCs for each size     
            if wanna_plot:
                print(f"=== Plotting FSC training history for size={size}x{size} ===\n")
                plot.plot_fsc_training_history(size, M_values, all_losses)

                print(f"=== Plotting best FSC graph of size={size}x{size} for all M values ===\n")
                for fsc in all_fsc:
                    plot.plot_fsc_graph(fsc, size)

    else:
        print("=== No FSC training requested. Models are gonna be loaded from disk for evaluation. ===\n")

    # The parameters for each size are already been tested and selected. 
    # The lr is cutted durig the execution based on how the training goes.
    parameters_lstm = [
        (3, "./data/Trajectories_3x3.pkl", 32, 1, 1e-3, 50, 32), # size, path, hidden_size, num_layers, lr, patience, batch_size
        (6, "./data/Trajectories_6x6.pkl", 64, 2, 1e-3, 50, 32),
        (9, "./data/Trajectories_9x9.pkl", 128, 2, 1e-3, 50, 32)
    ]

    # 5. Train the LSTM on the collected trajectories for each size
    if trainLSTM:

        all_lstm = [] # Save all LSTM models
        all_histories = [] # Save all training histories for each LSTM

        for size, path, hidden_size, num_layers, lr, patience, batch_size in parameters_lstm:
            print(f"=== Training LSTM for size={size}x{size} ===\n")
            with open(path, 'rb') as f:
                dataset = pickle.load(f)
            model, (history_train, history_val) = train_lstm(dataset,
                                                                size=size,
                                                                hidden_size=hidden_size,
                                                                num_layers=num_layers,
                                                                max_epochs=500,
                                                                lr=lr,
                                                                patience=patience,
                                                                batch_size=batch_size,
                                                                verbose=True)

            all_lstm.append(model)
            all_histories.append((history_train, history_val))

    # 6. Plot the training history for each size
            if wanna_plot:
                print(f"=== Plotting LSTM training history for size={size}x{size} ===\n")
                plot.plot_lstm_training_history(size, history_train, history_val)
    else:
        print("=== No LSTM training requested. Models are gonna be loaded from disk for evaluation. ===\n")

    # In the next part I will execute different evaluation scripts:
    # - Trajectory visualization: for each model and size, I will create a GIF that shows the agents playing in the environment. 
    #   The GIFs will be composed of a 3x3 grid of video, where each cell shows the behavior of one model (FSC, LSTM, RPPO) on a specific size (3x3, 6x6, 9x9).
    # - Success rate: 100 episodes for each model and size, average reward, average success rate and average steps.
    #   The plot will be a group bar plot, on the X-axis the different sizes, on the Y-axis the success rate (in %) and each group will have 3 bars (FSC, LSTM, RPPO).
    # - Parametric efficiency: the number of parameters of each model (FSC, LSTM) for each size (3x3, 6x6, 9x9).
    #   The plot will be a table, where the rows are the different sizes and the columns are the different models.
    # - Scale and Computation complexcity: for each model and size, I will measue the average training time and inference time. 
    #   The plot will be a line graph, on the X-axis the size, on the Y-axis the time (in seconds) in log scale and each line will represent a model (FSC, LSTM).
    # - Robustness: for each model and size, I will test the agents in an enviroment with some randomness (10% error in the action execution) and masure the success rate.
    #   The plot will be a heathmap of the visit, a grid of the labirinth where the color of each cell represents the number of times the agent has visited that cell during the evaluation episodes.
    # - Interpretability: for each model and size, I will create a visualization of the internal state of the model (FSC) during the execution of an episode.

    ######################################################################
    #                           Make the GIF                             # 
    ######################################################################

    print("=== Preparing GIF for Trajectories Visualization ===\n")

    # Load FSC models with M = 4, which is the overall best M value for all sizes.
    M = 4
    fsc_models_by_size = {
        3: load_fsc(f'./models/FSC_3/FSC_{M}states.pt', M=M),
        6: load_fsc(f'./models/FSC_6/FSC_{M}states.pt', M=M),
        9: load_fsc(f'./models/FSC_9/FSC_{M}states.pt', M=M),
    }

    # Load the best LSTM models saved with the validation loss during training.
    lstm_models_by_size = {
        3: load_lstm('./models/LSTM_3/best_model.pt', obs_dim=4, hidden_size=32, n_actions=4, num_layers=1),
        6: load_lstm('./models/LSTM_6/best_model.pt', obs_dim=4, hidden_size=64, n_actions=4, num_layers=2),
        9: load_lstm('./models/LSTM_9/best_model.pt', obs_dim=4, hidden_size=128, n_actions=4, num_layers=2),
    }

    # Load the best Recurrent PPO models saved during training.
    rppo_models_by_size = {
        3: load_rppo("models/RecurrentPPO_3_100_0.0003/best_model"),
        6: load_rppo("models/RecurrentPPO_6_500_0.0003/best_model"),
        9: load_rppo("models/RecurrentPPO_9_1000_0.0005/best_model"),
    }

    trajectories_9 = collect_9_trajectories_for_grid_gif(
        fsc_models_by_size=fsc_models_by_size,
        lstm_models_by_size=lstm_models_by_size,
        rppo_models_by_size=rppo_models_by_size,
        sizes=(3, 6, 9),
        max_steps_by_size={3: 100, 6: 100, 9: 100},
        model_order=("FSC", "LSTM", "RPPO"),
    )

    make_trajectories_grid_gif(
        trajectories_9,
        gif_path="./plots/Comparison/Simulation_grid.gif",
        model_order=["FSC", "LSTM", "RPPO"],
        size_order=[3, 6, 9],
        fps=6
    )

    ######################################################################
    #                           Success Rate                             # 
    ######################################################################

    print("=== Evaluating Success Rate for each model and size ===\n")

    number_of_episodes = 1000
    results_fsc = {}
    results_lstm = {}
    results_rppo = {}

    print("Evaluating Recurrent PPO models...")
    for size, model in rppo_models_by_size.items():
        total_reward, total_steps, total_success = 0, 0, 0
        for i in range(number_of_episodes):
            reward, steps, success = run_rppo_episode(model, grid_size=size, max_steps=100)
            total_reward += reward
            total_steps += steps
            total_success += success

        results_rppo[size] = {
            "average_reward": total_reward / number_of_episodes,
            "average_steps": total_steps / number_of_episodes,
            "success_rate": total_success / number_of_episodes * 100
        }

    print("Evaluating FSC models...")
    for size, model in fsc_models_by_size.items():
        total_reward, total_steps, total_success = 0, 0, 0
        for i in range(number_of_episodes):
            reward, steps, success = run_fsc_episode(model, grid_size=size, max_steps=100, verbose=False)
            total_reward += reward
            total_steps += steps
            total_success += success

        results_fsc[size] = {
            "average_reward": total_reward / number_of_episodes,
            "average_steps": total_steps / number_of_episodes,
            "success_rate": total_success / number_of_episodes * 100
        }
    
    print("Evaluating LSTM models...")
    for size, model in lstm_models_by_size.items():
        total_reward, total_steps, total_success = 0, 0, 0
        for i in range(number_of_episodes):
            reward, steps, success = run_lstm_episode(model, grid_size=size, max_steps=100)
            total_reward += reward
            total_steps += steps
            total_success += success

        results_lstm[size] = {
            "average_reward": total_reward / number_of_episodes,
            "average_steps": total_steps / number_of_episodes,
            "success_rate": total_success / number_of_episodes * 100
        }
    
    print("\n=== Success Rate Results ===\n")
    plot.plot_model_comparison_bars(results_rppo, results_fsc, results_lstm)


    ######################################################################
    #               Scale and Computational Complexity                   # 
    ######################################################################

    print("\n=== Evaluating Scale and Computational Complexity for each model and size ===\n")

    inference_times_fsc = {}
    inference_times_lstm = {}

    number_of_episodes = 1000

    for size, fsc in fsc_models_by_size.items():
        inference_time = inference_time_fsc(fsc, grid_size=size, max_steps=100, n_episodes=number_of_episodes)
        inference_times_fsc[size] = inference_time / number_of_episodes # Average time per episode

    for size, lstm in lstm_models_by_size.items():
        inference_time = inference_time_lstm(lstm, grid_size=size, max_steps=100, n_episodes=number_of_episodes)
        inference_times_lstm[size] = inference_time / number_of_episodes # Average time per episode

    print("=== Inference Time Results (Average time per episode in seconds) ===\n")
    for size in inference_times_fsc.keys():
        print(f"Size: {size}x{size} - FSC Inference Time: {inference_times_fsc[size]} sec, LSTM Inference Time: {inference_times_lstm[size]} sec")

    plot.plot_inference_time_comparison(inference_times_fsc, inference_times_lstm)

    ######################################################################
    #                             Robustness                             # 
    ######################################################################

    print("\n=== Evaluating Robustness for each model and size ===\n")

    number_of_episodes = 1000
    results_fsc = {}
    results_lstm = {}
    randomness = [0.1, 0.2, 0.3, 0.4, 0.5] # Different levels of randomness to test

    for random in randomness:
        print(f"Evaluating FSC models with randomness {random}...")
        for size, model in fsc_models_by_size.items():
            total_reward, total_steps, total_success = 0, 0, 0
            for i in range(number_of_episodes):
                reward, steps, success = robustness_fsc(model, grid_size=size, randomness=random, n_episodes=number_of_episodes)
                total_reward += reward
                total_steps += steps
                total_success += success

            results_fsc[size] = {
                "average_reward": total_reward / number_of_episodes,
                "average_steps": total_steps / number_of_episodes,
                "success_rate": total_success / number_of_episodes * 100
            }
        
        print(f"Evaluating LSTM models with randomness {random}...")
        for size, model in lstm_models_by_size.items():
            total_reward, total_steps, total_success = 0, 0, 0
            for i in range(number_of_episodes):
                reward, steps, success = robustness_lstm(model, grid_size=size, randomness=random, n_episodes=number_of_episodes)
                total_reward += reward
                total_steps += steps
                total_success += success

            results_lstm[size] = {
                "average_reward": total_reward / number_of_episodes,
                "average_steps": total_steps / number_of_episodes,
                "success_rate": total_success / number_of_episodes * 100
            }
        
        print(f"\n=== Success Rate Results with randomness {random} ===\n")
        plot.plot_robustness_bars(results_fsc, results_lstm, random, save_dir="./plots/Comparison")