import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class FSC(nn.Module):
    """
    Finite State Controller parametrizied with pytorch. 
    The rough parameters (theta_*) are free in R.
    The log-probabilities are obtained via F.log_softmax, which:
    - ensures stochasticity (values sum to 1)
    - maintains the computational graph for autograd

    Normalization axes for log-softmax:
    theta_pi  (M,)         → dim=0  → rho(m_0)         sum over M
    theta_pie (A, M)       → dim=0  → pi(a|m)           sum over A for each m
    theta_g   (M, M, A, Y) → dim=0  → g(m'|m, a, y)    sum over M_next for each (m,a,y)

    Input:
    - M: Number of internal states (memory states)
    - A: Number of actions
    - Y: Number of possible observations

    Output:
    - nll_trajectory(traj_a, traj_y): Computes the Negative Log-Likelihood of a given 
      trajectory of actions and observations, which can be used for training the FSC via gradient descent
    """

    def __init__(self, M: int, A: int, Y: int):
        super().__init__()
        self.M = M
        self.A = A
        self.Y = Y
        self.theta_pi  = nn.Parameter(torch.randn(M))
        self.theta_pie = nn.Parameter(torch.randn(A, M))
        self.theta_g   = nn.Parameter(torch.randn(M, M, A, Y))

    def nll_trajectory(self, traj_a: torch.Tensor, traj_y: torch.Tensor) -> torch.Tensor:
        """
        Computes the Negative Log-Likelihood (NLL) of a single trajectory using the forward algorithm (alpha recursion).

        Input:
        - traj_a : LongTensor (T,) - actions (0..A-1)
        - traj_y : LongTensor (T,) - observation indices (0..Y-1)
        
        Output:
        - scalar Tensor - NLL of the trajectory (with autograd graph)
        """
        log_rho = F.log_softmax(self.theta_pi,  dim=0)   # (M,)
        log_pie = F.log_softmax(self.theta_pie,  dim=0)   # (A, M)
        log_g   = F.log_softmax(self.theta_g,    dim=0)   # (M, M, A, Y)

        log_alpha = log_rho  # (M,)

        for a_t, y_t in zip(traj_a, traj_y):
            log_E_t   = log_pie[a_t]               # (M,)        – emission (action)
            log_T_t   = log_g[:, :, a_t, y_t]     # (M_next, M_prev) – transition
            # Broadcasting: (M_next, M_prev) + (1, M_prev) + (1, M_prev)
            inside    = log_T_t + log_alpha.unsqueeze(0) + log_E_t.unsqueeze(0)
            log_alpha = torch.logsumexp(inside, dim=1)  # (M_next,)

        return -torch.logsumexp(log_alpha, dim=0)  # scalar

def obs_to_index(obs):
    '''
    Converts a binary observation (array of 4 bits) into an integer index (0 to 15).

    Input:
    - obs: A binary array of shape (4,) representing the presence of walls

    Output:
    - An integer index corresponding to the binary observation
    '''
    return int("".join(obs.astype(str)), 2)

def train_fsc(dataset,
                size: int,
                M: int = 8,
                A: int = 4,
                Y: int = 16,
                n_restarts: int = 5,
                max_epochs: int = 500,
                lr: float = 1e-2,
                patience: int = 50,
                batch_size: int = 64,
                verbose: bool = True,
                ):
    """
    Train an FSC via NLL minima + Adam + random restarts.
    At each restart, the parameters are re-initialized with torch.randn.
    Early stopping is triggered if the loss does not decrease by at least 1e-4
    for `patience` consecutive epochs.
    Returns the FSC with the lowest NLL among all restarts.

    Input:
    - dataset: List of trajectories, where each trajectory is a dict with keys 'observations', 'actions', 'rewards'
    - size: Size of the gridworld (used for naming the saved model)
    - M: Number of internal states (memory states)
    - A: Number of actions
    - Y: Number of possible observations
    - n_restarts: Number of random restarts for training
    - max_epochs: Maximum number of epochs per restart
    - lr: Learning rate for Adam optimizer
    - patience: Number of epochs to wait for improvement before early stopping
    - batch_size: Number of trajectories per training batch
    - verbose: If True, prints training progress and final NLL

    Output:
    - best_fsc: The FSC instance with the lowest NLL found across all restarts
    - best_loss: The lowest NLL achieved
    - history: A list of lists, where each inner list contains the epoch losses for a restart
    """
    
    # Prepare the dataset: binary observations to integer indices (0–15)
    traj_a_list = [torch.tensor(t['actions'], dtype=torch.long) for t in dataset]
    traj_y_list = [
        torch.tensor([obs_to_index(o) for o in t['observations']], dtype=torch.long)
        for t in dataset
    ]
    N = len(traj_a_list)

    best_loss       = float('inf')
    best_state_dict = {}
    history         = []  # history[r] = list of losses per epoch for restart r

    for restart in range(n_restarts):
        fsc       = FSC(M, A, Y)
        optimizer = torch.optim.Adam(fsc.parameters(), lr=lr)

        no_improve      = 0
        best_epoch_loss = float('inf')
        epoch_losses    = []

        for epoch in range(max_epochs):
            # Shuffle indices to create random mini-batches
            indices = torch.randperm(N)
            epoch_nll_sum = 0.0

            for start in range(0, N, batch_size):
                batch_idx = indices[start : start + batch_size]

                optimizer.zero_grad()

                nll_list = [
                    fsc.nll_trajectory(traj_a_list[int(i)], traj_y_list[int(i)])
                    for i in batch_idx
                ]
                loss = torch.stack(nll_list).mean()

                loss.backward()
                # Gradient clipping: prevent gradient explosion on long trajectories
                torch.nn.utils.clip_grad_norm_(fsc.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_nll_sum += loss.item() * len(batch_idx)

            epoch_loss = epoch_nll_sum / N
            epoch_losses.append(epoch_loss)

            # Early stopping
            if epoch_loss < best_epoch_loss - 1e-4:
                best_epoch_loss = epoch_loss
                no_improve      = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                break

        history.append(epoch_losses)

        if verbose:
            print(f"Restart {restart+1}/{n_restarts} | Epoch: {len(epoch_losses):4d} | NLL: {best_epoch_loss:.4f}")

        if best_epoch_loss < best_loss:
            best_loss       = best_epoch_loss
            best_state_dict = {k: v.clone().detach() for k, v in fsc.state_dict().items()}

    # Recreate the FSC and load the parameters from the best restart
    best_fsc = FSC(M, A, Y)
    best_fsc.load_state_dict(best_state_dict)

    if not os.path.exists(f'./models/FSC_{size}/'):
        os.makedirs(f'./models/FSC_{size}', exist_ok=True)
    torch.save(best_fsc.state_dict(), f'./models/FSC_{size}/FSC_{M}states.pt')

    if verbose:
        print(f"\nBest NLL: {best_loss:.4f}\n")

    return best_fsc, best_loss, history

def load_fsc(path, M):
    '''
    Loads a trained FSC from the specified path.

    Input:
    - path: The file path to the saved FSC model (e.g., './models/FSC_3/FSC_8states.pt')
    - M: The number of internal states (memory states) used in the FSC architecture

    Output:
    - model: The loaded FSC model.
    '''
    fsc_loaded = FSC(M=M, A=4, Y=16)
    fsc_loaded.load_state_dict(torch.load(path, weights_only=True))
    fsc_loaded.eval()
    return fsc_loaded