import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkModuleBiLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=1750, embedding_dim=300)
        self.rnn = nn.LSTM(input_size=300, hidden_size=100,
                           batch_first=True, bidirectional=True)
        self.flat = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(in_features=48000, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, input1, input2):
        tmp1 = self.emb(input1)
        tmp1, (h, c) = self.rnn(tmp1)
        tmp2 = self.emb(input2)
        tmp2, (h, c) = self.rnn(tmp2)
        tmp1 = self.flat(tmp1)
        tmp2 = self.flat(tmp2)
        input = torch.cat((tmp1, tmp2), 1)
        output = self.seq(input)
        return output


class NetworkModuleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=1750, embedding_dim=300)
        self.trans = nn.Transformer()
        self.flat = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(in_features=48000, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )