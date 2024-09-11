import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentNeuralNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float,
        device: torch.device,
        hidden_dim: int = 200,
    ) -> None:
        super(RecurrentNeuralNetwork, self).__init__()

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            dropout=0.3,
            num_layers=5,
            device=device,
        )
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        self.layer = nn.Linear(hidden_dim, vocab_size, device=device)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # calculate sequence lengths, and pack the sequence
        lengths = (inp.sum(dim=-1) != 0).sum(dim=1).cpu()
        packed_input = pack_padded_sequence(
            inp, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        # unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # get the last non-padded output for each sequence
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        hidden = self.gelu(last_output)
        hidden = self.dropout(hidden)
        hidden = self.layer(hidden)

        return self.softmax(hidden)
