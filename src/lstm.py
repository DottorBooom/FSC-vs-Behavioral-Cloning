import os

from matplotlib.pylab import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMPolicy(nn.Module):
    """
    This class implements a simple LSTM-based policy network for Behavioral Cloning.
    It takes as input a sequence of observations and outputs action logits for each time step.
    The architecture consists of:
    - An LSTM layer to process the sequence of observations and capture temporal dependencies.
    - A LayerNorm for stabilizing training.
    - A Dropout layer for regularization.
    - A Linear layer to produce the final action logits.

    Input:
    - obs_dim: Dimension of the input observation (default: 4 for the gridworld)
    - hidden_size: Number of hidden units in the LSTM (default: 32)
    - n_actions: Number of possible actions (default: 4 for the gridworld)
    - num_layers: Number of LSTM layers (default: 1)
    """

    def __init__(self, obs_dim: int = 4, hidden_size: int = 32,
                n_actions: int = 4, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(hidden_size, 
                            hidden_size,
                            num_layers=num_layers, 
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden_size, n_actions)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, obs_seq: torch.Tensor, hidden=None):
        """
        obs_seq : (T, obs_dim)  - sequence of observations in a single episode
        hidden  : None or (h, c) with shape (num_layers, 1, hidden_size)
        Returns : logits (T, n_actions), new hidden state
        """
        x = obs_seq.unsqueeze(0)           # (1, T, obs_dim)
        x = self.encoder(x)                # (1, T, hidden_size)
        out, hidden = self.lstm(x, hidden)  # out: (1, T, H)
        out = self.layer_norm(out)
        out = self.dropout(out)
        logits = self.head(out.squeeze(0))  # (T, n_actions)
        return logits, hidden

    def predict_action(self, obs: np.ndarray, hidden=None, temperature: float = 0.8):
        """
        Predicts the action for a single step at runtime.
        obs    : binary array (4,)
        Returns: integer action, new hidden state
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,4)
            x = self.encoder(x)  # (1,1,H)
            out, hidden = self.lstm(x, hidden)
            logits = self.head(out.squeeze(0))   # (1, 4)

            # Apply temperature scaling for more stable action selection during evaluation
            probs = F.softmax(logits / temperature, dim=-1)

            # Sample action from the probability distribution
            action = int(torch.multinomial(probs, 1).item())
        return action, hidden
    
def train_lstm(dataset,
                size: int,
                hidden_size: int = 32,
                num_layers: int = 1,
                max_epochs: int = 500,
                lr: float = 1e-3,
                patience: int = 20,
                batch_size: int = 32,
                temperature: float = 0.8,
                verbose: bool = True,
                ):
    """
    Train LSTM via cross-entropy on expert actions.
    The training is supervised: for each trajectory, the LSTM receives the sequence of observations and must predict the correct action at each step.
    Returns the trained model and the training history (loss curves) of the training and validation sets.

    Input:
    - dataset: A list of trajectories, where each trajectory is a dictionary containing 'observations' (array of shape (T, 4)) and 'actions' (array of shape (T,)).
    - size: Size of the gridworld (used for naming the saved model and plots)
    - hidden_size: Number of hidden units in the LSTM (default: 32)
    - num_layers: Number of LSTM layers (default: 1)
    - max_epochs: Maximum number of training epochs (default: 500)
    - lr: Learning rate for the optimizer (default: 1e-3)
    - patience: Number of epochs to wait for improvement before early stopping (default: 20)
    - batch_size: Number of trajectories per training batch (default: 32)
    - verbose: If True, prints training progress and final loss values
    """

    # Convert dataset: obs float, actions long
    obs_list = [torch.tensor(t['observations'], dtype=torch.float32) for t in dataset]
    act_list = [torch.tensor(t['actions'],      dtype=torch.long)    for t in dataset]
    N = len(obs_list)

    # Split data into train and validation (80% train, 20% val)
    split_idx = int(0.8 * N)
    obs_list_train, obs_list_val = obs_list[:split_idx], obs_list[split_idx:]
    act_list_train, act_list_val = act_list[:split_idx], act_list[split_idx:]

    model     = LSTMPolicy(obs_dim=4, hidden_size=hidden_size,
                            n_actions=4, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    no_improve      = 0
    history_train         = []
    history_val           = []

    # Early stopping parameters
    best_val_loss = float('inf')

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                           mode='min', 
                                                           factor=0.5, 
                                                           patience=10)

    if not os.path.exists(f'./models/LSTM_{size}/'):
        os.makedirs(f'./models/LSTM_{size}', exist_ok=True)

    for epoch in range(max_epochs):

        model.train()
        train_running_loss = 0.0

        indices = torch.randperm(split_idx)

        for start in range(0, split_idx, batch_size):
            batch_idx = indices[start: min(start + batch_size, split_idx)]
            optimizer.zero_grad()

            loss_list = [
                F.cross_entropy(
                    model(obs_list_train[int(i)])[0],   # logits (T, 4)
                    act_list_train[int(i)],               # targets (T,)
                    label_smoothing=0.1
                )
                for i in batch_idx
            ]
            loss = torch.stack(loss_list).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_running_loss += loss.item() * len(batch_idx)

        avg_train_loss = train_running_loss / split_idx

        model.eval()
        val_running_loss = 0.0

        with torch.no_grad():
            for start in range(0, N-split_idx, batch_size):
                batch_idx = torch.arange(start, min(start + batch_size, N-split_idx))
                val_loss_list = [
                    F.cross_entropy(
                        model(obs_list_val[int(i)])[0],
                        act_list_val[int(i)]
                    )
                    for i in batch_idx
                ]
                loss = torch.stack(val_loss_list).mean()
                
                val_running_loss += loss.item() * len(batch_idx)

        avg_val_loss = val_running_loss / (N - split_idx)
        history_train.append(avg_train_loss)
        history_val.append(avg_val_loss)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if epoch >= max_epochs*0.2: # Warm-up period
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improve = 0
                torch.save(model.state_dict(), f'./models/LSTM_{size}/best_model.pt')  # Save the best model
            else:
                no_improve += 1
                if no_improve > patience:
                    print("Early stopping triggered")
                    break
        
        scheduler.step(avg_val_loss)

    model.load_state_dict(torch.load(f'./models/LSTM_{size}/best_model.pt'))  # Load the best model
    model.eval()

    if verbose:
        print(f"\nBest Validation Loss: {best_val_loss:.4f}\n")

    return model, (history_train, history_val)

def load_lstm(path, obs_dim: int = 4, hidden_size: int = 32, n_actions: int = 4, num_layers: int = 1):
    '''
    Load a trained LSTM model from disk.

    Input:
    - path: str, path to the saved LSTM model checkpoint
    - obs_dim: int, dimension of the input observations (default: 4)
    - hidden_size: int, number of hidden units in the LSTM (default: 32)
    - n_actions: int, number of possible actions (default: 4)
    - num_layers: int, number of LSTM layers (default: 1)
    
    Output:
    - model: The loaded LSTMPolicy model ready for inference.
    '''

    lstm_loaded = LSTMPolicy(
        obs_dim=obs_dim,
        hidden_size=hidden_size,
        n_actions=n_actions,
        num_layers=num_layers
    )
    lstm_loaded.load_state_dict(torch.load(path, map_location='cpu'), strict=True)
    lstm_loaded.eval()

    return lstm_loaded