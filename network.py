import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NetworkModuleBiLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=1706,
                                embedding_dim=128, padding_idx=0)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64,
                            batch_first=True, bidirectional=True)
        self.flat = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(in_features=23040, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, s1, s2, len1, len2):
        s1 = self.emb(s1)
        s1 = pack_padded_sequence(
            input=s1, lengths=len1, batch_first=True, enforce_sorted=False)
        s1, _ = self.lstm(s1)
        s1, _ = pad_packed_sequence(s1, batch_first=True, total_length=90)
        s2 = self.emb(s2)
        s2 = pack_padded_sequence(
            input=s1, lengths=len2, batch_first=True, enforce_sorted=False)
        s2, _ = self.lstm(s2)
        s2, _ = pad_packed_sequence(s2, batch_first=True, total_length=90)
        s1 = self.flat(s1)
        s2 = self.flat(s2)
        input = torch.cat((s1, s2), 1)
        output = self.seq(input)
        return output


class NetworkModuleTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=1706,
                                embedding_dim=128, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
        self.trans = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=6)
        self.flat = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(in_features=23040, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, s1, s2, len1, len2):
        s1 = self.emb(s1)
        s1 = self.trans(s1)
        s2 = self.emb(s2)
        s2 = self.trans(s2)
        s1 = self.flat(s1)
        s2 = self.flat(s2)
        input = torch.cat((s1, s2), 1)
        output = self.seq(input)
        return output


class NetworkModuleBERT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
