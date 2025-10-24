"""
This module implements the Anomaly Transformer model, a novel architecture for unsupervised time series anomaly detection.

The Anomaly Transformer, proposed by Xu et al. (2022), is based on the Transformer architecture and introduces the concept of Anomaly-Attention to explicitly distinguish between normal and anomalous patterns in time series data. The model leverages the idea that anomalies, due to their rarity, have difficulty establishing strong associations with the entire time series, a concept termed "adjacent-concentration bias".

This module contains the implementation of the main `AnomalyTransformer` class, along with the `Encoder` and `EncoderLayer` classes that form its core components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding
from .kernel import KernelType


class EncoderLayer(nn.Module):
    """
    Represents a single layer of the Anomaly Transformer's encoder.

    Each encoder layer consists of a multi-head self-attention mechanism (AnomalyAttention)
    followed by a position-wise feed-forward network. Residual connections and layer
    normalization are applied around each of the two sub-layers. This implementation also
    includes an optional LSTM network that can be used as an alternative to the feed-forward network.
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", l_lstm=0):
        """
        Initializes the EncoderLayer.

        Args:
            attention (nn.Module): The attention module to be used (e.g., AttentionLayer with AnomalyAttention).
            d_model (int): The dimensionality of the input and output of the layer (the model's dimension).
            d_ff (int, optional): The dimensionality of the inner-layer of the feed-forward network. Defaults to 4 * d_model.
            dropout (float, optional): The dropout rate. Defaults to 0.1.
            activation (str, optional): The activation function to use in the feed-forward network, "relu" or "gelu". Defaults to "relu".
            l_lstm (int, optional): The number of layers in the optional LSTM network. If 0, the feed-forward network is used. Defaults to 0.
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # The position-wise feed-forward network is implemented as two 1D convolutions.
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        # Define an optional LSTM network as an alternative to the FeedForward network.
        self.l_lstm = l_lstm
        if l_lstm != 0:
            # The LSTM network has the same input and hidden size as the model's dimension.
            self.lstm1 = nn.LSTM(input_size=d_model, dropout=dropout, hidden_size=d_model, num_layers=l_lstm)
            print(f"Using a {self.l_lstm} layer LSTM RNN network instead of a Feed-Forward in the Encoder Layer")

    def forward(self, x, attn_mask=None):
        """
        Performs the forward pass of the encoder layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            attn_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor of the layer of shape (batch_size, sequence_length, d_model).
                - torch.Tensor: The series-association scores from the attention mechanism.
                - torch.Tensor: The prior-association scores from the attention mechanism.
                - torch.Tensor: The learned sigma values from the attention mechanism.
        """
        # Apply the Anomaly-Attention mechanism.
        new_x, attn, mask, sigma = self.attention(
                x, x, x,
                attn_mask=attn_mask
        )
        # Apply the residual connection and dropout to the output of the attention layer.
        x = x + self.dropout(new_x)
        # Apply the first layer normalization.
        y = x = self.norm1(x)
        # Apply the position-wise feed-forward network or the optional LSTM network.
        if self.l_lstm == 0:
            # Apply the feed-forward network.
            y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
            y = self.dropout(self.conv2(y).transpose(-1, 1))
        else:
            # Apply the LSTM network.
            y_t, (h_t, c_t) = self.lstm1(x)
            y = y_t
        # Apply the residual connection and the second layer normalization.
        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    """
    Represents the encoder of the Anomaly Transformer, which is a stack of EncoderLayer instances.

    The encoder processes the input sequence and generates a continuous representation that is then used for reconstruction.
    """
    def __init__(self, attn_layers, norm_layer=None):
        """
        Initializes the Encoder.

        Args:
            attn_layers (list): A list of EncoderLayer instances.
            norm_layer (nn.Module, optional): A normalization layer to be applied after the last encoder layer. Defaults to None.
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Performs the forward pass of the encoder.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            attn_mask (torch.Tensor, optional): The attention mask. Defaults to None.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The output tensor of the encoder of shape (batch_size, sequence_length, d_model).
                - list: A list of the series-association scores from each encoder layer.
                - list: A list of the prior-association scores from each encoder layer.
                - list: A list of the learned sigma values from each encoder layer.
        """
        series_list = []
        prior_list = []
        sigma_list = []
        # Process the input through each encoder layer.
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        # Apply the final normalization layer if it exists.
        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(nn.Module):
    """
    The main Anomaly Transformer model for unsupervised time series anomaly detection.

    This model is an implementation of the Anomaly Transformer as described in the paper
    "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy" by Xu et al. (2022).
    It uses a stack of encoder layers with a novel Anomaly-Attention mechanism to learn the associations
    within the time series and identify anomalies based on the discrepancy between the learned series-association
    and a prior-association.
    """

    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True, kernel_type=KernelType.GAUSSIAN, l_lstm=0):
        """
        Initializes the AnomalyTransformer model.

        Args:
            win_size (int): The size of the input window.
            enc_in (int): The number of input features (dimensionality of the input time series).
            c_out (int): The number of output features.
            d_model (int, optional): The dimensionality of the model's hidden states. Defaults to 512.
            n_heads (int, optional): The number of attention heads in the multi-head attention mechanism. Defaults to 8.
            e_layers (int, optional): The number of encoder layers. Defaults to 3.
            d_ff (int, optional): The dimensionality of the inner-layer of the feed-forward network. Defaults to 512.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
            activation (str, optional): The activation function to use, "relu" or "gelu". Defaults to 'gelu'.
            output_attention (bool, optional): Whether to output the attention scores. Defaults to True.
            kernel_type (KernelType, optional): The type of kernel to use for the prior-association. Defaults to KernelType.GAUSSIAN.
            l_lstm (int, optional): The number of layers in the optional LSTM network in each encoder layer. If 0, the feed-forward network is used. Defaults to 0.
        """
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Input embedding layer, which combines token embedding and positional embedding.
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # The encoder, which is a stack of encoder layers.
        self.encoder = Encoder(
                [
                    EncoderLayer(
                            AttentionLayer(
                                    AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention,
                                                     kernel_type=kernel_type),
                                    d_model, n_heads),
                            d_model,
                            d_ff,
                            dropout=dropout,
                            activation=activation,
                            l_lstm=l_lstm
                    ) for l in range(e_layers)
                ],
                norm_layer=torch.nn.LayerNorm(d_model)
        )

        # The final projection layer to map the encoder output to the desired output dimension.
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        Performs the forward pass of the Anomaly Transformer model.

        Args:
            x (torch.Tensor): The input time series of shape (batch_size, sequence_length, num_features).

        Returns:
            If `output_attention` is True, returns a tuple containing:
                - torch.Tensor: The reconstructed output of shape (batch_size, sequence_length, c_out).
                - list: A list of the series-association scores from each encoder layer.
                - list: A list of the prior-association scores from each encoder layer.
                - list: A list of the learned sigma values from each encoder layer.
            Otherwise, returns:
                - torch.Tensor: The reconstructed output of shape (batch_size, sequence_length, c_out).
        """
        # Apply the input embedding.
        enc_out = self.embedding(x)
        # Pass the embedded input through the encoder.
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        # Project the encoder output to the final output dimension.
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out