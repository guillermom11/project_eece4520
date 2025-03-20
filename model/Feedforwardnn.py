import torch
from torch import nn

class FeedForwardNN(nn.Module):

    def __init__(self,d_model,bias=False,dropout=0.2):
        """
        Arguments:
        d_model: size of embedding dimension
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.c_fc    = nn.Linear(d_model, 4 * d_model, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x) # [B, T, 4*d]
        x = self.gelu(x)
        x = self.c_proj(x) # [B, T, d]
        x = self.dropout(x)
        return x