# -*- coding: UTF-8 -*-
"""
@project:RefineCTR
"""

import torch


class RandomNormal(object):

    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.normal_(embed.weight, self.mean, self.std)
        return embed


class RandomUniform(object):
    def __init__(self, minval=0.0, maxval=1.0):
        self.minval = minval
        self.maxval = maxval

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.uniform_(embed.weight, self.minval, self.maxval)
        return embed


class XavierNormal(object):

    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_normal_(embed.weight, self.gain)
        return embed


class XavierUniform(object):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, vocab_size, embed_dim):
        embed = torch.nn.Embedding(vocab_size, embed_dim)
        torch.nn.init.xavier_uniform_(embed.weight, self.gain)
        return embed


class Pretrained(object):
    def __init__(self, embedding_weight, freeze=True):
        self.embedding_weight = torch.FloatTensor(embedding_weight)
        self.freeze = freeze

    def __call__(self, vocab_size, embed_dim):
        assert vocab_size == self.embedding_weight.shape[0] and embed_dim == self.embedding_weight.shape[1]
        embed = torch.nn.Embedding.from_pretrained(self.embedding_weight, freeze=self.freeze)
        return embed
