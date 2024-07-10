import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
  def __init__(self, d_model, vocab_size):
    super().__init__()

    self.d_model = d_model
    self.vocab_size = vocab_size

    self.embedding = nn.Embedding(vocab_size, d_model)

  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, seq_len, dropout):
    super().__init__()

    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[: ,1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0) # (1, seq_len, d_model)

    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)

class LayerNormalization(nn.Module):
  def __init__(self, eps=10**-6):
    super().__init__()

    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(1))
    self.bias = nn.Parameter(torch.zeros(1))

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)

    return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff, dropout):
    super().__init__()

    self.linear_1 = nn.Linear(d_model, d_ff)
    self.relu = nn.ReLU()
    self.linear_2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, h, dropout):
    super().__init__()

    self.d_model = d_model
    self.h = h
    assert d_model % h == 0, "d_model must be divisible by h"

    self.d_k = d_model // h # 512 / 8 = 64 by default
    self.w_q = nn.Linear(d_model, d_model)
    self.w_k = nn.Linear(d_model, d_model)
    self.w_v = nn.Linear(d_model, d_model)

    # fully connected layer: 8*64x512 or 512x512
    self.w_o = nn.Linear(d_model, d_model)

    self.dropout = nn.Dropout(dropout)

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


  def forward(self, q, k, v, mask):
    query = self.w_q(q) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    key = self.w_k(k) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
    value = self.w_v(v) # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

    # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

    x, self.attention_scores = self.ScaledDotProductAttention(query, key, value, mask, self.dropout)

    # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
    x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

    return self.w_o(x)

class EncoderLayer(nn.Module):
  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()

    self.attention = MultiHeadAttention(d_model, h, dropout)
    self.feed_forward = FeedForward(d_model, d_ff, dropout)
    self.norm1 = LayerNormalization()
    self.norm2 = LayerNormalization()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, mask):
    attn_output = self.attention(x,x,x,mask)
    x = self.norm1(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))

    return x

class DecoderLayer(nn.Module):
  def __init__(self, d_model, d_ff, h, dropout):
    super().__init__()

    self.masked_attention = MultiHeadAttention(d_model, h, dropout)
    self.norm1 = LayerNormalization()
    self.attention = MultiHeadAttention(d_model, h, dropout)
    self.norm2 = LayerNormalization()
    self.feed_forward = FeedForward(d_model, d_ff, dropout)
    self.norm3 = LayerNormalization()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, enc_output, src_mask, trg_mask):
    masked_attn_output = self.masked_attention(x, x, x, trg_mask)
    x = self.norm1(x + self.dropout(masked_attn_output))
    attn_output = self.attention(x, enc_output, enc_output, src_mask)
    x = self.norm2(x + self.dropout(attn_output))
    ff_output = self.feed_forward(x)
    x = self.norm3(x + self.dropout(ff_output))

    return x

class Transformer(nn.Module):
  def __init__(self, d_model, seq_len, d_ff, h, num_layers, src_vocab_size, trg_vocab_size, dropout):
    super().__init__()

    self.src_embed = Embedding(d_model, src_vocab_size)
    self.trg_embed = Embedding(d_model, trg_vocab_size)
    self.pos = PositionalEncoding(d_model, seq_len, dropout)

    self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, d_ff, h, dropout) for _ in range(num_layers)])
    self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, d_ff, h, dropout) for _ in range(num_layers)])

    self.linear = nn.Linear(d_model, trg_vocab_size)
    self.dropout = nn.Dropout(dropout)

  def generate_mask(self, src, trg):
    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
    trg_mask = (trg != 0).unsqueeze(1).unsqueeze(3)
    seq_length = trg.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    trg_mask = trg_mask & nopeak_mask
    return src_mask, trg_mask

  def forward(self, src, trg):
    src_mask, trg_mask = self.generate_mask(src, trg)

    src_embed = self.dropout(self.pos(self.src_embed(src)))
    trg_embed = self.dropout(self.pos(self.trg_embed(trg)))

    enc_output = src_embed
    for encoder_layer in self.encoder_layers:
      enc_output = encoder_layer(enc_output, src_mask)

    dec_output = trg_embed
    for decoder_layer in self.decoder_layers:
      dec_output = decoder_layer(dec_output, enc_output, src_mask, trg_mask)

    return self.linear(dec_output)