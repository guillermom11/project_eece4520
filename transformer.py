import torch
from torch import nn
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=128):
        """_summary_

        Args:
            d_model (int, optional): size of embedding dimension. Defaults to 512.
            max_len (int, optional): max length of input seq. Defaults to 128.
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        term_for_div = 1 / torch.tensor(10000.0) ** (embedding_index / d_model)
        pe[:, 0::2] = torch.sin(position * term_for_div)
        pe[:, 1::2] = torch.cos(position * term_for_div)
        self.register_buffer("pe", pe)

    def forward(self, word_embeddings):
        return word_embeddings + self.pe[: word_embeddings.size(0), :]




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
        self.attn = nn.MultiheadAttention(d_model, num_heads, max_length, dropout, bias)
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
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, max_length, vocab_size,layers, bias=False,dropout=0.2):
        """
        Arguments:
        d_model: size of embedding dimension
        num_heads: number of attention heads
        max_length: maximum length of input sequences (in tokens)
        vocab_size: size of the token vocabulary
        layers: number of decoder-only blocks
        bias: whether or not to use bias in linear layers
        dropout: probability of dropout
        """
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, d_model), # token embeddings
            wpe=PositionEncoding(d_model, max_length), # position embeddings
            drop=nn.Dropout(dropout),
            blocks=nn.ModuleList([DecoderTransformerBlock(d_model, num_heads, max_length, bias, dropout) for _ in range(layers)]),
            ln_f=nn.LayerNorm(d_model),
            head=nn.Linear(d_model, vocab_size, bias=bias),
        ))

    def forward(self, idx, targets=None):
        
        device = idx.device
        _, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # generate token and position embeddings
        tok_emb = self.transformer.wte(idx) # [B, T, d]
        pos_emb = self.transformer.wpe(pos) # [T, d]
        x = self.transformer.drop(tok_emb + pos_emb)

        # pass through all decoder-only blocks
        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln_f(x) # final layer norm

        if targets is not None:
            # compute the loss if we are given targets
            logits = self.transformer.head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # only look at last token if performing inference
            logits = self.transformer.head(x[:, [-1], :])
            loss = None

        return logits, loss