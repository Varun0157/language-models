import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentNeuralNetwork(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float,
        hidden_dim: int = 300,
    ) -> None:
        super(RecurrentNeuralNetwork, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # Calculate sequence lengths
        lengths = (inp.sum(dim=-1) != 0).sum(dim=1).cpu()

        # Pack the sequence
        packed_input = pack_padded_sequence(
            inp, lengths, batch_first=True, enforce_sorted=False
        )

        # Pass through LSTM
        packed_output, _ = self.lstm(packed_input)

        # Unpack the sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Get the last non-padded output for each sequence
        last_output = output[torch.arange(output.size(0)), lengths - 1]

        # Apply dropout and linear layer
        hidden = self.dropout(last_output)
        hidden = self.layer(hidden)

        return self.softmax(hidden)
