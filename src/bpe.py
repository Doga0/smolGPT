from transformers import PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)

def get_training_corpus(dataset):
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

class BPE():
    def __init__(self, dataset, vocab_size, special_tokens=None):

      self.dataset = dataset
      self.vocab_size = vocab_size

      self.tokenizer = Tokenizer(models.BPE())
      self.special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens

    def train(self):
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        trainer = trainers.BpeTrainer(
          vocab_size = self.vocab_size,
          special_tokens = self.special_tokens
        )
        self.tokenizer.train_from_iterator(get_training_corpus(self.dataset), trainer=trainer)

        self.tokenizer.post_processor = processors.ByteLevel(
          trim_offsets=False,
        )

        self.tokenizer.decoder = decoders.ByteLevel()

        wrapped_tokenizer = PreTrainedTokenizerFast(
          tokenizer_object=self.tokenizer,
          bos_token="<|endoftext|>",
          eos_token="<|endoftext|>",
        )

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def save(self, path):
        self.tokenizer.model.save(path)

