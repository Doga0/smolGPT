from transformer import Transformer
import torch
import torch.nn as nn
import torch.optim as optim

src_vocab_size = 5000
trg_vocab_size = 5000
d_model = 512
h = 8
num_layers = 6
d_ff = 2048
seq_len = 100
dropout = 0.1

model = Transformer(d_model, seq_len, d_ff, h, num_layers, src_vocab_size, trg_vocab_size, dropout)

src_data = torch.randint(1, src_vocab_size, (64, seq_len))
tgt_data = torch.randint(1, trg_vocab_size, (64, seq_len))

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

model.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = model(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, trg_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

model.eval()

# Random sample validation data
val_src_data = torch.randint(1, src_vocab_size, (64, seq_len))  # (batch_size, seq_length)
val_tgt_data = torch.randint(1, trg_vocab_size, (64, seq_len))  # (batch_size, seq_length)

with torch.no_grad():

    val_output = model(val_src_data, val_tgt_data[:, :-1])
    val_loss = criterion(val_output.contiguous().view(-1, trg_vocab_size), val_tgt_data[:, 1:].contiguous().view(-1))
    print(f"Validation Loss: {val_loss.item()}")