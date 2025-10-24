"""
This module provides the building blocks for a standard Transformer Encoder, used for comparison with the Anomaly Transformer.

It includes the `SelfAttEncoder` which is a stack of encoder layers, and the `SelfAttEncoderLayer` which represents a single encoder layer with a self-attention mechanism and a feed-forward network.
"""
from torch import nn
import torch.nn.functional as Func


class SelfAttEncoder(nn.Module):
    """
    A stack of self-attention encoder layers, forming the encoder of a standard Transformer model.

    This module iteratively processes the input sequence through a series of `SelfAttEncoderLayer` instances.
    """
    def __init__(self, attn_layers, norm_layer=None):
        """
        Initializes the SelfAttEncoder.

        Args:
            attn_layers (list): A list of `SelfAttEncoderLayer` instances.
            norm_layer (nn.Module, optional): A normalization layer to be applied after the last encoder layer. Defaults to None.
        """
        super(SelfAttEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        """
        Performs the forward pass of the self-attention encoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor of the encoder of shape (batch_size, sequence_length, d_model).
        """
        # Process the input through each encoder layer.
        for attn_layer in self.attn_layers:
            x = attn_layer(x)
        # Apply the final normalization layer if it exists.
        if self.norm is not None:
            x = self.norm(x)
        return x


class SelfAttEncoderLayer(nn.Module):
    """
    A single layer of a standard Transformer encoder.

    Each layer consists of a self-attention mechanism followed by a position-wise feed-forward network.
    Residual connections and layer normalization are applied around each of the two sub-layers.
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        """
        Initializes the SelfAttEncoderLayer.

        Args:
            attention (nn.Module): The self-attention module to be used.
            d_model (int): The dimensionality of the input and output of the layer.
            d_ff (int, optional): The dimensionality of the inner-layer of the feed-forward network. Defaults to 4 * d_model.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(SelfAttEncoderLayer, self).__init__()
        # The self-attention block.
        self.attention = attention
        # The dimensionality of the feed-forward network.
        d_ff = d_ff or 4 * d_model
        print(f"Using Feed Forward Layer of size {d_ff}")
        self.inner_ff = nn.Linear(in_features=d_model, out_features=d_ff)
        self.outer_ff = nn.Linear(in_features=d_ff, out_features=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = Func.relu

    def forward(self, x):
        """
        Performs the forward pass of the encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The output tensor of the layer of shape (batch_size, sequence_length, d_model).
        """
        # Compute the self-attention output.
        attn_out = self.attention(x, x, x)
        # Apply the first residual connection and layer normalization.
        norm1_in = x + self.dropout(attn_out)
        norm1_out = self.norm1(norm1_in)
        # Compute the output of the feed-forward network.
        inner_ff_out = self.dropout(self.relu(self.inner_ff(norm1_out)))
        outer_ff_out = self.outer_ff(inner_ff_out)
        ff_out = self.dropout(outer_ff_out)
        # Apply the second residual connection and layer normalization.
        return self.norm2(ff_out + norm1_out)