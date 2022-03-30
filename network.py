import torch.nn as nn
import torch.nn.functional as F

class NetworkModuleBiLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential([
            nn.Embedding()
        ])