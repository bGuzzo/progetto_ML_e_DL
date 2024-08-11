from torch import nn


# Classical self-attention implementation, map the input to Q, K, V and return the computed attention
class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        # Use Pytorch Multi Head Attention for convenience
        self.inner_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        # Query, Key and Value projection
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        queries = self.query_projection(queries)
        keys = self.key_projection(keys)
        values = self.value_projection(values)
        self_att_out, _ = self.inner_attention(queries, keys, values)
        return self_att_out
