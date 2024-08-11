from torch import nn

from model.embed import DataEmbedding
from self_attention.self_att_encoder import SelfAttEncoder, SelfAttEncoderLayer
from self_attention.self_attention import SelfAttentionLayer


class TransformerEncoder(nn.Module):

    # Use the same default params as the Anomaly Transformer
    def __init__(self, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=None, dropout=0.0):
        super(TransformerEncoder, self).__init__()
        # Use the same data encoder as the Anomaly Transformer
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        # Self-Attention Encoder
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
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        # Embed the input
        enc_out = self.embedding(x)
        # Apply the model
        enc_out = self.encoder(enc_out)
        # Re-Map the output to the original dimensionality
        return self.projection(enc_out)
