import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAX_SEQUENCE_LENGTH = 400  # max len is 353


def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
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
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        dropout_rate: float,
        device: torch.device,
        embed_dim=300,
        num_layers=2,
        num_heads=6,
    ):
        super(TransformerModel, self).__init__()
        self.device = device

        self.pos_encoder = PositionalEncoding(
            max_len=MAX_SEQUENCE_LENGTH, d_model=embed_dim
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=0.4,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Calculate sequence lengths
        lengths = (x.sum(dim=-1) != 0).sum(dim=1)

        # Filter out any sequences with length 0
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

        # Get the last non-padded output for each sequence
        last_output = x[torch.arange(x.size(0)), lengths - 1]

        # Apply final linear layer and softmax
        out = self.dropout(last_output)
        out = self.linear(out)
        out = self.softmax(out)

        return out
