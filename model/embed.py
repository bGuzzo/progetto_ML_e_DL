"""
This module provides the embedding layers for the Anomaly Transformer model.

It includes the `PositionalEmbedding` which generates sinusoidal positional encodings, the `TokenEmbedding` which embeds the input features using a 1D convolution, and the `DataEmbedding` which combines these two to create the final input embedding for the model.
"""
import math

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """
    Implements the sinusoidal positional encoding as described in "Attention Is All You Need".

    This layer adds information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so they can be summed.
    """
    def __init__(self, d_model, max_len=5000):
        """
        Initializes the PositionalEmbedding layer.

        Args:
            d_model (int): The dimensionality of the model (and the embedding).
            max_len (int, optional): The maximum sequence length. Defaults to 5000.
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # The sine and cosine functions are used to create the positional encodings.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The positional encoding tensor of shape (1, sequence_length, d_model).
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Implements a token embedding using a 1D convolution.

    This layer projects the input features into the model's dimension `d_model`.
    """
    def __init__(self, c_in, d_model):
        """
        Initializes the TokenEmbedding layer.

        Args:
            c_in (int): The number of input features.
            d_model (int): The dimensionality of the model.
        """
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # The token embedding is implemented as a 1D convolution with a kernel size of 3.
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Initialize the weights of the convolutional layer using Kaiming normalization.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Performs the forward pass of the token embedding.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, c_in).

        Returns:
            torch.Tensor: The embedded tensor of shape (batch_size, sequence_length, d_model).
        """
        # The input is permuted to be compatible with the 1D convolution.
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    """
    Combines the token embedding and positional embedding to create the final input embedding.

    This is the entry point for the data embedding process in the Anomaly Transformer.
    """

    def __init__(self, c_in, d_model, dropout=0.0):
        """
        Initializes the DataEmbedding layer.

        Args:
            c_in (int): The number of input features.
            d_model (int): The dimensionality of the model.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Performs the forward pass of the data embedding.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, c_in).

        Returns:
            torch.Tensor: The final embedded tensor of shape (batch_size, sequence_length, d_model).
        """
        # The final embedding is the sum of the token embedding and the positional embedding.
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)