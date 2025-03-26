import torch
from torch import nn
import torch.nn.functional as F


from model.DecoderTransformerBlock import DecoderTransformerBlock

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
            wpe=nn.Embedding(max_length, d_model), # position embeddings
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
        pos_emb = self.transformer.wpe(pos).unsqueeze(0).expand_as(tok_emb)
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