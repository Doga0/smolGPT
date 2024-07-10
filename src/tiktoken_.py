import tiktoken

encodings_ = 'cl100k_base'
model_ = 'gpt-3.5-turbo'

class Tokenizer:
  def __init__(self, encoding=None, model=None):
    self.encodings = encoding if encoding is not None else encodings_
    self.model = model if model is not None else model_
    self.tokenizer = tiktoken.get_encoding(self.encodings)
    self.tokenizer = tiktoken.encoding_for_model(self.model)

  def encode(self, text):
    return self.tokenizer.encode(text)

  def decode(self, ids):
    return self.tokenizer.decode(ids)

  def get_vocab(self):
    return self.tokenizer.n_vocab


