import torch

class Generate:
  def __init__(self):
    super().__init__()

  def greedy_decode(self, model, idx, max_new_tokens, block_size):

    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]

      with torch.no_grad():
        logits = model(idx_cond)
        logits = logits[:, -1, :]

      probs = torch.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

  def beam_decode(self):
    # to do
    pass