import torch
from torch import nn
from model.FeedForwardNN import FeedForwardNN

class DecoderTransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads,max_length,bias=False,dropout=0.2):
        """
        Arguments:
        d_model: size of embedding dimension
        num_heads: number of attention heads
        max_length: maximum length of input sequences (in tokens)
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, bias=bias)
        self.ln_2 = nn.LayerNorm(d_model)
        self.ffnn = FeedForwardNN(d_model, bias, dropout)

    def forward(self, x,padding_mask=0):
        bs, l, h = x.shape
        mask = torch.triu(torch.ones(l, l, device=x.device), 1).bool()
        #residual connections
        norm_x = self.ln_1(x)
        #masking attention
        x = x + self.attn(norm_x,norm_x,norm_x,attn_mask=mask,key_padding_mask=padding_mask)[0]
        norm_x = self.ln_2(x)
        x = x + self.ffnn(norm_x)
        return x
    