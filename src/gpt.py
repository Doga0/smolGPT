import torch
import torch.nn as nn
import math

class GPT_CONFIG:
  block_size:     int = 1024
  vocab_size:     int = 50257
  embed_dim:      int = 768
  num_heads:      int = 12
  num_layers:     int = 12
  dropout:        float = 0.1
  bias:           bool = True

class LayerNorm(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.eps = 1e-5
    self.scale = nn.Parameter(torch.ones(cfg.embed_dim))
    self.shift = nn.Parameter(torch.zeros(cfg.embed_dim))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    norm_x = (x - mean) / (std + self.eps)
    return self.scale * norm_x + self.shift

class GELU(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) *
     (x + 0.044715 * torch.pow(x, 3))
    ))

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.linear_1 = nn.Linear(cfg.embed_dim, cfg.embed_dim * 4, bias=cfg.bias)
    self.gelu = GELU()
    self.linear_2 = nn.Linear(cfg.embed_dim * 4, cfg.embed_dim, bias=cfg.bias)
    self.dropout = nn.Dropout(cfg.dropout)

  def forward(self, x):
    return self.linear_2(self.dropout(self.gelu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.embed_dim = cfg.embed_dim
    self.h = cfg.num_heads
    assert cfg.embed_dim % cfg.num_heads == 0, "embed_dim must be divisible by num_heads"

    self.d_k = cfg.embed_dim // cfg.num_heads # 512 / 8 = 64 by default
    self.w_q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    self.w_k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    self.w_v = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    # fully connected layer: 8*64x512 or 512x512
    self.w_o = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    self.dropout = nn.Dropout(cfg.dropout)
    self.register_buffer("mask", torch.triu(torch.ones(cfg.block_size, cfg.block_size), diagonal=1))

  def ScaledDotProductAttention(self, query, key, value, mask, dropout):
    d_k = query.shape[-1]

    # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)

    attention_scores = torch.softmax(attention_scores, dim=-1)

    if dropout is not None:
      attention_scores = self.dropout(attention_scores) # (batch, h, seq_len, seq_len)

    return torch.matmul(attention_scores, value), attention_scores


  def forward(self, x):
    batch, num_tokens, d_in = x.shape

    query = self.w_q(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    key = self.w_k(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    value = self.w_v(x) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

    # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
    query = query.view(batch, num_tokens, self.h, self.d_k).transpose(1, 2)
    key = key.view(batch, num_tokens, self.h, self.d_k).transpose(1, 2)
    value = value.view(batch, num_tokens, self.h, self.d_k).transpose(1, 2)

    mask = self.mask.bool()[:num_tokens, :num_tokens]

    x, self.attention_scores = self.ScaledDotProductAttention(query, key, value, mask, self.dropout)

    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

    return self.w_o(x)

class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.norm_1 = LayerNorm(cfg)
    self.attn = MultiHeadAttention(cfg)
    self.dropout = nn.Dropout(cfg.dropout)
    self.norm_2 = LayerNorm(cfg)
    self.ff = FeedForward(cfg)

  def forward(self, x):
    shortcut = x
    x = self.norm_1(x)
    x = self.attn(x)
    x = self.dropout(x)
    x = x + shortcut

    shortcut = x
    x = self.norm_2(x)
    x = self.ff(x)
    x = self.dropout(x)
    x = x + shortcut

    return x

class GPT(nn.Module):
  def __init__(self, cfg):
    super().__init__()

    self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
    self.pos_embed = nn.Embedding(cfg.block_size, cfg.embed_dim)
    self.dropout = nn.Dropout(cfg.dropout)

    self.blocks = nn.Sequential(
        *[TransformerBlock(cfg) for _ in range(cfg.num_layers)]
    )

    self.norm = LayerNorm(cfg)
    self.linear = nn.Linear(cfg.embed_dim, cfg.vocab_size)

  def forward(self, idx):
    batch_size, seq_len = idx.shape

    tok_embed = self.tok_embed(idx)
    pos_embed = self.pos_embed(torch.arange(seq_len, device=idx.device))
    x = self.dropout(tok_embed + pos_embed)
    x = self.blocks(x)
    x = self.norm(x)
    x = self.linear(x)

    return x

