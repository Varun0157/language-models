import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0)]


class TransformerDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        d_model=512,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # Linear layer to project concatenated embeddings to d_model dimension
        self.input_projection = nn.Linear(5 * embedding_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.input_projection.weight.data.uniform_(-initrange, initrange)
        self.input_projection.bias.data.zero_()
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        # src shape: (batch_size, 5, embedding_dim)
        batch_size, _, _ = src.shape

        # Reshape and project the input
        src = src.view(batch_size, -1)  # (batch_size, 5 * embedding_dim)
        src = self.input_projection(src)  # (batch_size, d_model)
        src = src.unsqueeze(0)  # (1, batch_size, d_model)

        src = self.pos_encoder(src)

        # We don't need tgt_mask or memory_mask for single-step prediction
        output = self.transformer_decoder(src, src)
        output = self.fc_out(output)

        # apply softmax to get probabilities
        output = nn.functional.log_softmax(output, dim=-1)

        return output.squeeze(0)  # (batch_size, vocab_size)
