"""
This module implements the core attention mechanisms for the Anomaly Transformer model.

It includes the novel `AnomalyAttention` mechanism, which is the heart of the Anomaly Transformer,
and the `AttentionLayer` which is a wrapper that implements the multi-head attention mechanism.
The `AnomalyAttention` computes both the series-association (learned from the data) and a prior-association
(based on a chosen kernel, e.g., Gaussian), which are fundamental for calculating the association discrepancy
for anomaly detection.
"""
from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from model.kernel import KernelType, apply_kernel


class TriangularCausalMask():
    """
    A triangular causal mask for self-attention mechanisms.

    This mask is used to prevent positions from attending to subsequent positions in a sequence.
    This is crucial for autoregressive models where the prediction for the current time step
    can only depend on the previous time steps.
    """
    def __init__(self, B, L, device="cpu"):
        """
        Initializes the TriangularCausalMask.

        Args:
            B (int): The batch size.
            L (int): The sequence length.
            device (str, optional): The device to create the mask on. Defaults to "cpu".
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """Returns the causal mask."""
        return self._mask


class AnomalyAttention(nn.Module):
    """
    The Anomaly-Attention mechanism as proposed in the Anomaly Transformer paper.

    This attention mechanism calculates two types of associations:
    1.  **Series-Association**: Learned from the data using a standard scaled dot-product attention.
    2.  **Prior-Association**: A prior belief about the attention distribution, typically based on the
        relative distance between time points, modeled by a kernel function (e.g., Gaussian).

    The discrepancy between these two associations is then used to identify anomalies.
    """
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False, kernel_type=KernelType.GAUSSIAN):
        """
        Initializes the AnomalyAttention module.

        Args:
            win_size (int): The size of the attention window.
            mask_flag (bool, optional): Whether to apply a causal mask. Defaults to True.
            scale (float, optional): The scaling factor for the attention scores. If None, it defaults to 1/sqrt(d_keys). Defaults to None.
            attention_dropout (float, optional): The dropout rate for the attention scores. Defaults to 0.0.
            output_attention (bool, optional): Whether to output the attention scores. Defaults to False.
            kernel_type (KernelType, optional): The type of kernel to use for the prior-association. Defaults to KernelType.GAUSSIAN.
        """
        super(AnomalyAttention, self).__init__()
        self.kernel_type = kernel_type
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        # Pre-calculate the distance matrix for the prior-association.
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        """
        Performs the forward pass of the Anomaly-Attention.

        Args:
            queries (torch.Tensor): The query tensor of shape (B, L, H, E).
            keys (torch.Tensor): The key tensor of shape (B, S, H, E).
            values (torch.Tensor): The value tensor of shape (B, S, H, D).
            sigma (torch.Tensor): The learned scale parameter for the prior-association kernel of shape (B, L, H).
            attn_mask (torch.Tensor, optional): The attention mask.

        Returns:
            A tuple containing:
                - torch.Tensor: The output of the attention layer of shape (B, L, H, D).
                - torch.Tensor or None: The series-association scores if `output_attention` is True.
                - torch.Tensor or None: The prior-association scores if `output_attention` is True.
                - torch.Tensor or None: The learned sigma values if `output_attention` is True.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Calculate the raw attention scores (dot product between queries and keys).
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply the causal mask if specified.
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Scale the attention scores.
        attn = scale * scores

        # Calculate the prior-association.
        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        # The learned sigma is passed through a sigmoid and a scaling function to ensure it's in a reasonable range.
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # The distance matrix is repeated for each batch and head.
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        # Apply the chosen kernel function to get the prior-association.
        prior = apply_kernel(self.kernel_type, prior, sigma)

        # Calculate the series-association by applying softmax to the attention scores.
        series = self.dropout(torch.softmax(attn, dim=-1))
        # The final output is the weighted sum of the values, where the weights are the series-association scores.
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """
    A wrapper for the attention module that incorporates multi-head attention and projections.

    This layer takes the queries, keys, and values, and projects them into multiple heads.
    It then applies the inner attention mechanism (e.g., AnomalyAttention) and combines the results
    from all heads.
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        Initializes the AttentionLayer.

        Args:
            attention (nn.Module): The inner attention module to be used (e.g., AnomalyAttention).
            d_model (int): The dimensionality of the input and output of the layer.
            n_heads (int): The number of attention heads.
            d_keys (int, optional): The dimensionality of the keys. Defaults to d_model // n_heads.
            d_values (int, optional): The dimensionality of the values. Defaults to d_model // n_heads.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        # Linear projections for queries, keys, values, and the sigma parameter.
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        # The final output projection.
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """
        Performs the forward pass of the multi-head attention layer.

        Args:
            queries (torch.Tensor): The query tensor of shape (B, L, D).
            keys (torch.Tensor): The key tensor of shape (B, S, D).
            values (torch.Tensor): The value tensor of shape (B, S, D).
            attn_mask (torch.Tensor, optional): The attention mask.

        Returns:
            A tuple containing:
                - torch.Tensor: The output of the attention layer of shape (B, L, D).
                - torch.Tensor or None: The series-association scores.
                - torch.Tensor or None: The prior-association scores.
                - torch.Tensor or None: The learned sigma values.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        # Project and reshape the queries, keys, and values for multi-head attention.
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # Project the input to get the sigma parameter for each head.
        sigma = self.sigma_projection(x).view(B, L, H)

        # Apply the inner attention mechanism.
        out, series, prior, sigma = self.inner_attention(
                queries,
                keys,
                values,
                sigma,
                attn_mask
        )
        # Concatenate the heads and apply the final output projection.
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma