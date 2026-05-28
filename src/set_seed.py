import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    # 1. Standard Python
    random.seed(seed)
    
    # 2. Numpy
    np.random.seed(seed)
    
    # 3. PyTorch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    
    # GPU configurations (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False