import torch
import torch.nn as nn
from utils import MultiHeadAttention,FeedForwardNetwork,PositionalEncoding,InputEmbedding,LayerNormalisation,ResidualConnections



class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block, feed_forward_block, dropout):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnections(dropout) for _ in range(2)])
    
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        return self.residual_connections[1](x, self.feed_forward_block)

class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalisation()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)