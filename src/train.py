import torch
import torch.nn as nn
import math
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import sentencepiece as spm
import generate

device = (
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Using {device} device")

num_of_gpus = torch.cuda.device_count()
print("Number of gpus:", num_of_gpus)

# config parameters
dataset = load_dataset("Mursel/Turkish-wikipedia-10k")

epochs = 6
batch_size = 2
learning_rate = 6e-4
min_lr = 6e-5

betas = (0.9, 0.95)
device_type = 'cuda' if 'cuda' in device else 'cpu'
weight_decay=0.1

freq = 1

class GPT_CONFIG:
  block_size:     int = 256
  vocab_size:     int = 32000
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

    self.d_k = cfg.embed_dim // cfg.num_heads
    self.w_q = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    self.w_k = nn.Linear(cfg.embed_dim, cfg.embed_dim)
    self.w_v = nn.Linear(cfg.embed_dim, cfg.embed_dim)

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

    tok_emb = self.tok_embed(idx)
    pos_emb = self.pos_embed(torch.arange(seq_len, device=idx.device))
    x = self.dropout(tok_emb + pos_emb)
    x = self.blocks(x)
    x = self.norm(x)
    logits = self.linear(x)

    return logits

class GPTDataset:
  def __init__(self, text, tokenizer, max_len, stride):
    super().__init__()

    self.input_ids = []
    self.target_ids = []

    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    for i in range(0, len(token_ids) - max_len, stride):
      input_chunk = token_ids[i: i + max_len]
      target_chunk = token_ids[i + 1: i + max_len + 1]

      self.input_ids.append(input_chunk)
      self.target_ids.append(target_chunk)

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return {
        "input_ids": torch.tensor(self.input_ids[idx]),
        "target_ids": torch.tensor(self.target_ids[idx])
    }

def dataloader(tokenizer, text, batch_size, max_len,
              stride, shuffle, drop_last,
              num_workers=0):

  dataset = GPTDataset(text, tokenizer, max_len, stride)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers)

  return dataloader

split_dataset = dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
val_dataset = split_dataset["test"]

train_text = " ".join([entry["poem"] for entry in train_dataset])
val_text = " ".join([entry["poem"] for entry in val_dataset])

text = train_text + val_text

tokenizer = spm.SentencePieceProcessor(model_file="/content/tokenizer_32k.model")

print("Characters: ", len(text))
print("Tokens: ", len(tokenizer.Encode(text)))

config = GPT_CONFIG()

print("n_vocab: ",tokenizer.n_vocab)

train_ratio = 0.75

train_size = int(len(text) * train_ratio)

train_dataset = text[:train_size]
val_dataset = text[train_size:]

print("Train size: ", len(train_dataset))
print("Val size: ", len(val_dataset))

train_loader = dataloader(tokenizer, train_dataset, batch_size, config.block_size, config.block_size,
                          drop_last=False, shuffle=False)

val_loader = dataloader(tokenizer, val_dataset, batch_size, config.block_size, config.block_size,
                        drop_last=False, shuffle=False)

total_steps = int(epochs * len(train_loader) / batch_size)
warmup_steps = int(0.1 * total_steps)

for batch in train_loader:
  x = batch['input_ids'].to(device)
  y = batch['target_ids'].to(device)
  print(x.shape, y.shape)

for batch in val_loader:
  x = batch['input_ids'].to(device)
  y = batch['target_ids'].to(device)
  print(x.shape, y.shape)

model = GPT(config)
model.to(device)

# learning rate warmup with cosine decay
def lr_scheduler(step):

  if total_steps < warmup_steps:
    raise ValueError("Total steps must be greater or equal to warm up steps.")

  if step < warmup_steps:
    return learning_rate * step / warmup_steps

  if step > total_steps:
    return min_lr

  learning_rate = min_lr + 0.5 * (learning_rate - min_lr) * (1 + torch.cos((torch.pi * step - warmup_steps) / float(total_steps- warmup_steps)))

  return learning_rate

optimizer = optim.AdamW(model.parameters(), learning_rate, betas, weight_decay)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

def train_one_epoch(model, train_loader, freq, epoch):

  lr = lr_scheduler(epoch)

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

  loss_total = 0.
  last_loss = 0.
  size = len(train_loader)
  step = 0

  for i in train_loader:
    inputs  = i['input_ids'].to(device)
    targets = i['target_ids'].to(device)

    if inputs.max() >= config.vocab_size or inputs.min() < 0:
      raise ValueError(f"Input ids out of range. Found min: {inputs.min().item()}, max: {inputs.max().item()}")

    optimizer.zero_grad()

    outputs = model(inputs)

    outputs = outputs.view(-1, outputs.size(-1))
    targets = targets.view(-1)

    loss = loss_fn(outputs, targets)
    loss.backward()

    optimizer.step()

    loss_total += loss.item()
    step += 1
    if step % freq == 0:
      last_loss = loss_total / freq
      print(f"  Batch: {step} | LR: {lr} | Loss: {last_loss}")
      loss_total = 0.

  return last_loss

def train(model, train_loader, val_loader, epochs, freq):

  train_losses = []
  val_losses = []

  for epoch in range(epochs):
    model.train()

    print(f"\nEPOCH {epoch + 1}")
    print("---------------------------------------------")

    avg_loss = train_one_epoch(freq, model, train_loader, epoch)

    val_steps = 0
    total_val_loss = 0.

    model.eval()

    with torch.no_grad():
      for i, val_data in enumerate(val_loader):
        val_inputs, val_targets = val_data['input_ids'].to(device), val_data['target_ids'].to(device)

        val_outputs = model(val_inputs)

        val_outputs = val_outputs.view(-1, val_outputs.size(-1))
        val_targets = val_targets.view(-1)

        loss = loss_fn(val_outputs, val_targets)

        total_val_loss += loss.item()
        val_steps += 1

    avg_val_loss = total_val_loss / val_steps
    print(f"LOSS Train: {avg_loss} Valid: {avg_val_loss}")

    train_losses.append(avg_loss)
    val_losses.append(avg_val_loss)

  return train_losses, val_losses

def plot_losses(epochs, train_losses, val_losses):
  plt.figure()

  fig, ax = plt.subplots()

  ax.plot(epochs, train_losses, label="Train Loss")
  ax.plot(epochs, val_losses, linestyle="-.", label="Validation loss")

  ax.set_xlabel("Epochs")
  ax.set_ylabel("Loss")
  ax.legend(loc="upper right")

  fig.tight_layout()

train_losses, val_losses = train(model, train_loader, val_loader)

torch.save(model.state_dict(), "/content/model.pth")
model = GPT(GPT_CONFIG)
model.load_state_dict(torch.load("/content/model.pth"))

plot_losses(range(1, epochs + 1), train_losses, val_losses)

def visualize_attn():
  pass

context = "Merhaba, ben"
enc = tokenizer.Encode(context)
enc_tensor = torch.tensor(enc).unsqueeze(0)

output = generate.greedy_decode(model, enc_tensor, 3, config.block_size)

print("Output: ", output)
print("Output length: ", len(output[0]))

decoded_text = tokenizer.Decode(output.squeeze(0).tolist())
print(decoded_text)