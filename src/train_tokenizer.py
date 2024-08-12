import sentencepiece as spm
from datasets import load_dataset

dataset = load_dataset("Mursel/Turkish-wikipedia-100k")

train_dataset = "\n".join([entry["content"] for entry in dataset["train"]])
print(train_dataset[:1000])

file = "/content/corpus.txt"
with open(file, "w") as f:
    f.write(train_dataset)
    f.close()

with open(file, "r") as f:
    data = f.read()[:1000]
    print(data)
    f.close()

spm.SentencePieceTrainer.train(
    input=file,
    model_prefix="tokenizer_32k",
    vocab_size=32000,
    hard_vocab_limit=False
)

text = "Merhaba dünya!"
tokenizer = spm.SentencePieceProcessor(model_file="/vocab/tokenizer_32k.model")
enc = tokenizer.Encode(text)
print(enc)

dec = tokenizer.Decode(enc)
print(dec)
