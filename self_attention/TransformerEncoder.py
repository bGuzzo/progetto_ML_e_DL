"""
This module implements a standard Transformer Encoder model.

This model is used for comparison purposes with the Anomaly Transformer, to evaluate the effectiveness of the standard self-attention mechanism for anomaly detection in time series, as opposed to the novel Anomaly-Attention mechanism.

To ensure a fair comparison, this model uses the same data embedding and a similar architecture to the Anomaly Transformer, but with a standard multi-head self-attention mechanism instead of the Anomaly-Attention.
"""
from torch import nn

from model.embed import DataEmbedding
from self_attention.self_att_encoder import SelfAttEncoder, SelfAttEncoderLayer
from self_attention.self_attention import SelfAttentionLayer


class TransformerEncoder(nn.Module):
    """
    A standard Transformer Encoder model for time series reconstruction.

    This model is composed of a stack of self-attention encoder layers. It is designed to serve as a baseline to compare against the Anomaly Transformer, using a conventional self-attention mechanism instead of the Anomaly-Attention.
    """

    def __init__(self, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=None, dropout=0.0):
        """
        Initializes the TransformerEncoder model.

        The default parameters are chosen to be consistent with the Anomaly Transformer for a fair comparison.

        Args:
            enc_in (int): The number of input features.
            c_out (int): The number of output features.
            d_model (int, optional): The dimensionality of the model's hidden states. Defaults to 512.
            n_heads (int, optional): The number of attention heads. Defaults to 8.
            e_layers (int, optional): The number of encoder layers. Defaults to 3.
            d_ff (int, optional): The dimensionality of the inner-layer of the feed-forward network. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super(TransformerEncoder, self).__init__()
        # The data embedding is the same as the one used in the Anomaly Transformer.
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        # The encoder is a stack of standard self-attention encoder layers.
        self.encoder = SelfAttEncoder(
                [
                    SelfAttEncoderLayer(
                            SelfAttentionLayer(d_model=d_model, n_heads=n_heads, dropout=dropout),
                            d_model=d_model,
                            d_ff=d_ff,
                            dropout=dropout
                    ) for l in range(e_layers)
                ],
                norm_layer=nn.LayerNorm(d_model)
        )
        # The final projection layer to map the encoder output to the desired output dimension.
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        Performs the forward pass of the Transformer Encoder.

        Args:
            x (torch.Tensor): The input time series of shape (batch_size, sequence_length, num_features).

        Returns:
            torch.Tensor: The reconstructed output of shape (batch_size, sequence_length, c_out).
        """
        # Embed the input time series.
        enc_out = self.embedding(x)
        # Pass the embedded input through the self-attention encoder.
        enc_out = self.encoder(enc_out)
        # Project the encoder output to the final output dimension.
        return self.projection(enc_out)