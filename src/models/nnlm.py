import torch
import torch.utils.data
import torch.optim


class NeuralNetworkLanguageModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float,
        device: torch.device,
        sent_len: int = 5,
        hidden_dim: int = 300,
    ) -> None:
        super(NeuralNetworkLanguageModel, self).__init__()

        self.layer1 = torch.nn.Linear(
            embedding_dim * sent_len, hidden_dim, device=device
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer2 = torch.nn.Linear(hidden_dim, vocab_size, device=device)
        self.gelu = torch.nn.GELU()

        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.view(inp.size(0), -1)

        # first layer
        hidden = self.layer1(inp)
        hidden = self.gelu(hidden)
        hidden = self.dropout(hidden)

        # second layer
        pred = self.layer2(hidden)

        # softmax
        return self.softmax(pred)
