"""
This package implements the Anomaly Transformer model and its constituent components.

The Anomaly Transformer is a novel architecture for unsupervised time series anomaly detection based on the attention mechanism. This package provides the implementation of the main model, as well as the individual modules that compose it, such as the Anomaly Attention mechanism, the positional embedding, and the feed-forward layers.

The key modules in this package are:
- `AnomalyTransformer.py`: The main model implementation.
- `attn.py`: The implementation of the Anomaly Attention mechanism and its variants.
- `embed.py`: The implementation of the positional and token embeddings.
- `loss_func.py`: The implementation of the loss functions used for training the model.
- `optimizer.py`: The implementation of the optimizer used for training the model.
- `kernel.py`: The implementation of the kernel functions used in the prior-association.
"""
