# Mostly inspired from my sensei https://github.com/karpathy/minbpe

import regex as re

def get_stats(ids, counts=False):
  """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
  """

  counts = {} if counts is False else counts

  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1

  return counts

def merge(ids, pair, idx):
  """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """

  new_ids = []
  i=0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      new_ids.append(idx)
      i += 2
    else:
      new_ids.append(ids[i])
      i += 1
  return new_ids


PATTERNS = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class MinBPE:
  def __init__(self, patterns=None):
    super().__init__()

    self.patterns = PATTERNS if patterns is None else patterns
    self.compiled_pattern = re.compile(self.patterns)
    self.special_tokens = {}
    self.inverse_special_tokens = {}

  def preprocess(self, text):
    pass

  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256, "Vocab size must include at least all chars in UTF-8"
    num_merges = vocab_size - 256

    text_chunks = re.findall(self.compiled_pattern, text)

    merges = {}
    ids = [list(ch.encode('utf-8')) for ch in text_chunks]
    vocab = {idx: bytes([idx]) for idx in range(256)} #int --> bytes

    for i in range(num_merges):

      stats = {}
      for chunk in ids:
        get_stats(chunk, stats)

      pair = max(stats, key=stats.get)
      idx = 256 + i

      ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]

      merges[pair] = idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

      if verbose:
        print(f"Merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

    self.merges = merges
    self.vocab = vocab

  def encode_chunks(self, text):
    tokens = list(text)

    while(len(tokens) >= 2):
      stats = get_stats(tokens)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

      if pair not in self.merges:
        break

      idx = self.merges[pair]
      tokens = merge(tokens, pair, idx)

    return tokens

  def encode_ordinary(self, text):
    text_chunks = re.findall(self.compiled_pattern, text)

    ids = []
    for chunk in text_chunks:
      chunks_bytes = chunk.encode('utf-8')
      ids.extend(self.encode_chunks(chunks_bytes))

    return ids

  def encode(self, text, allowed_special="none_raise"):

        special = None

        if allowed_special == "all":
            special = self.special_tokens

        elif allowed_special == "none":
            special = {}

        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)

        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}

        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids

  def decode(self, ids):

    part_bytes = []
    for idx in ids:
        if idx in self.vocab:
            part_bytes.append(self.vocab[idx])
        elif idx in self.inverse_special_tokens:
            part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
        else:
            raise ValueError(f"invalid token id: {idx}")

    tokens = b"".join(self.vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")

    return text

  def build_vocab(self):

    vocab = {idx: bytes([idx]) for idx in range(256)}

    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    
    for special, idx in self.special_tokens.items():
      vocab[idx] = special

    return vocab


