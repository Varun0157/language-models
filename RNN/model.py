import torch
import torch.utils.data
import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RecurrentNeuralNetwork(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float,
        hidden_dim: int = 300,
    ) -> None:
        super(RecurrentNeuralNetwork, self).__init__()

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer = torch.nn.Linear(hidden_dim, vocab_size)
        # self.gelu = torch.nn.GELU()

        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(inp)
        # take the output of the last sequence
        lstm_out = lstm_out[:, -1, :]  # (32, 5, 300) -> (32, 300)
        lstm_out = self.dropout(lstm_out)

        hidden = self.layer(lstm_out)
        # hidden = self.gelu(hidden)

        return self.softmax(hidden)
