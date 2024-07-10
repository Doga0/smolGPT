from bpe import BPE
from minbpe import MinBPE
from tiktoken_ import Tokenizer
from datasets import load_dataset
import os
import urllib.request


dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")

print("-----HF BPE TEST-----")

path = "/test/"

bpe = BPE(dataset, 5000)
bpe.train()

bpe.save(path)

bpe.decode(bpe.encode("This is a test!"))

print("---------------------")


print("-----MinBPE TEST-----")

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

with open("the-verdict.txt", "r") as f:
    text = f.read()

t = MinBPE()
t.train(text[:600], 512, True)
print(t.decode(t.encode("This is a test!")))

print("---------------------")


print("-----TIKTOKEN TEST-----")

tokenizer = Tokenizer()
tokenizer.decode(tokenizer.encode('This is a test!'))

print("---------------------")