import hashlib
import os
from collections import Counter
from typing import Tuple, Union

import numpy as np
import torch
import torch as th


def load_data(path: str):
    fn = 'corpus.{}.data'.format(hashlib.md5(path.encode()).hexdigest())

    if os.path.exists(fn):
        print('Loading cached dataset...')
        corpus = torch.load(fn)

    else:
        print('Producing dataset...')
        corpus = Corpus(path)
        torch.save(corpus, fn)

    return corpus.train, corpus.valid, corpus.test


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'wiki.train.tokens'))
        self.valid = self.tokenize(os.path.join(path, 'wiki.valid.tokens'))
        self.test = self.tokenize(os.path.join(path, 'wiki.test.tokens'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


class LMLoader:
    def __init__(
            self,
            source: Union[th.Tensor, np.ndarray],
            device: str = "cpu",
            bptt: int = 10,
            batch_size: int = 20,
            evaluation: bool = False,
            to_device: bool = False,
    ):
        self.evaluation = evaluation
        self.bptt = bptt
        self.batch_size = batch_size
        self.device = device
        self.to_device = to_device

        if isinstance(source, np.ndarray):
            source = th.from_numpy(source)

        data = source.data.long()
        self.batches = self.batchify(data, batch_size)

    def batchify(self, data: th.Tensor, bsz: int) -> th.Tensor:
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()

        if self.to_device:
            data = data.to(self.device)

        return data

    def get_batch(self, i: int) -> Tuple[th.Tensor, th.Tensor]:
        seq_len = min(self.bptt, len(self.batches) - 1 - i)
        data = self.batches[i: i + seq_len]
        target = self.batches[i + 1: i + 1 + seq_len].view(-1)

        return (
            data.to(self.device, non_blocking=True),
            target.to(self.device, non_blocking=True)
        )

    def __len__(self):
        return self.batches.size(0) // self.bptt

    def __iter__(self):
        for i in range(0, self.batches.size(0) - 1, self.bptt):
            yield self.get_batch(i)
