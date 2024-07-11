from generate import Generate
from tiktoken_ import Tokenizer
from gpt import GPT, GPT_CONFIG
import torch

context = "Once upon a time"
tokenizer = Tokenizer()
enc = tokenizer.encode(context)
enc_tensor = torch.tensor(enc).unsqueeze(0)

config = GPT_CONFIG()
model = GPT(config)
model.eval()

generate = Generate()
output = generate.greedy_decode(model, enc_tensor, 10, config.block_size)

print("Output: ", output)
print("Output length: ", len(output[0]))

decoded_text = tokenizer.decode(output.squeeze(0).tolist())
print(decoded_text)