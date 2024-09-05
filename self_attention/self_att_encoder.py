from torch import nn
import torch.nn.functional as Func


# Encoder stack for the classical self-attention model
class SelfAttEncoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(SelfAttEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x):
        for attn_layer in self.attn_layers:
            # Load the subsequent layer with the previous attention output
            x = attn_layer(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


# Single encoder Layer
class SelfAttEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1):
        super(SelfAttEncoderLayer, self).__init__()
        # Load self-attention block
        self.attention = attention
        # Define Feed Forward size
        d_ff = d_ff or 4 * d_model
        print(f"Using Feed Forward Layer of size {d_ff}")
        self.inner_ff = nn.Linear(in_features=d_model, out_features=d_ff)
        self.outer_ff = nn.Linear(in_features=d_ff, out_features=d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = Func.relu

    def forward(self, x):
        # Compute self-attention output
        attn_out = self.attention(x, x, x)
        # Sum with dropout
        norm1_in = x + self.dropout(attn_out)
        norm1_out = self.norm1(norm1_in)
        # Compute Feed Forward network
        inner_ff_out = self.dropout(self.relu(self.inner_ff(norm1_out)))
        outer_ff_out = self.outer_ff(inner_ff_out)
        ff_out = self.dropout(outer_ff_out)
        return self.norm2(ff_out + norm1_out)
