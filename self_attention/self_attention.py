"""
This module implements a standard multi-head self-attention layer.

This layer is used in the baseline Transformer Encoder model for comparison with the Anomaly Transformer.
It uses the `nn.MultiheadAttention` module from PyTorch for a convenient and efficient implementation of the self-attention mechanism.
"""
from torch import nn


class SelfAttentionLayer(nn.Module):
    """
    A standard multi-head self-attention layer.

    This class implements the self-attention mechanism as described in "Attention Is All You Need".
    It takes queries, keys, and values as input, and computes the attention-weighted sum of the values.
    The implementation uses the `nn.MultiheadAttention` module from PyTorch for efficiency.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initializes the SelfAttentionLayer.

        Args:
            d_model (int): The dimensionality of the model.
            n_heads (int): The number of attention heads.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
        """
        super(SelfAttentionLayer, self).__init__()
        # The core of this layer is the PyTorch MultiheadAttention module.
        self.inner_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        # Linear projections for queries, keys, and values.
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        """
        Performs the forward pass of the self-attention layer.

        Args:
            queries (torch.Tensor): The query tensor of shape (seq_len, batch_size, d_model).
            keys (torch.Tensor): The key tensor of shape (seq_len, batch_size, d_model).
            values (torch.Tensor): The value tensor of shape (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: The output of the self-attention layer of shape (seq_len, batch_size, d_model).
        """
        # Project the queries, keys, and values.
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        # Apply the multi-head attention.
        self_att_out, _ = self.inner_attention(queries, keys, values)
        return self_att_out