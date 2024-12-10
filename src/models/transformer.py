import math

import torch
from torch import nn

MAX_SEQUENCE_LENGTH = 400  # max len is 353


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dropout_rate: float,
        device: torch.device,
        embedding_dim=300,
        num_layers=2,
        num_heads=6,
    ):
        super(TransformerModel, self).__init__()
        self.device = device

        self.pos_encoder = PositionalEncoding(
            max_len=MAX_SEQUENCE_LENGTH, d_model=embedding_dim
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.4,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # as in RNN's
        lengths = (x.sum(dim=-1) != 0).sum(dim=1)
        non_zero_mask = lengths > 0
        x = x[non_zero_mask]
        lengths = lengths[non_zero_mask]

        if len(lengths) == 0:
            raise ValueError("All sequences in the batch have length 0")

        # Generate input sequence mask
        max_len = x.size(1)
        input_mask = generate_square_subsequent_mask(max_len).to(self.device)

        # Apply positional encoding
        x = self.pos_encoder(x)

        # Apply transformer decoder
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)

        # Get the last non-padded output for each sequence, as in RNN
        last_output = x[torch.arange(x.size(0)), lengths - 1]

        # Apply final linear layer and softmax
        out = self.dropout(last_output)
        out = self.linear(out)
        out = self.softmax(out)

        return out
