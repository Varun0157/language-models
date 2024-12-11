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

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim * sent_len, hidden_dim, device=device),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(hidden_dim, vocab_size, device=device),
            torch.nn.LogSoftmax(dim=1),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = inp.view(inp.size(0), -1)
        return self.layers(inp)
