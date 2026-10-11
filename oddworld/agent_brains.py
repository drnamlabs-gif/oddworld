# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Agent brains for the world: each agent's genome IS the attention brain's 1,056 weights, inherited directly
(sexual blend + mutation in the world). No hand-built relational feature, no lifetime learning.

The world's objects use a 10-dim code (shape 4, colour 4, size 2). The brain reads 12 dims: shape and colour each
have one extra value that never occurs in the world. The probe tests genomes offline on puzzles from the world's
vocabulary and on puzzles with those unseen values; the world never uses it.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from attention_brain import N_W, init_theta, one_hot2, sample2, scores, scores_paired, unflatten

N_GENES = N_W
WORLD_TO_BRAIN = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 10, 11])   # world one-hot dim -> brain input dim


def to_brain_input(X):
    Y = torch.zeros(*X.shape[:-1], 12)
    Y[..., WORLD_TO_BRAIN] = X
    return Y


def random_genomes(n, gen):
    return init_theta(n, gen)


class AttentionBrains:
    def __init__(self):
        self.world = None
        tg = torch.Generator().manual_seed(4242)
        a, o = sample2(500, 5, tg, "train"); self.Xt, self.Ot = one_hot2(a), o
        a, o = sample2(500, 5, tg, "novel"); self.Xn, self.On = one_hot2(a), o

    @torch.no_grad()
    def __call__(self, slots, X):
        W = unflatten(self.world.genome[torch.as_tensor(slots)])
        return torch.softmax(scores_paired(W, to_brain_input(X)), -1).numpy()

    @torch.no_grad()
    def probe(self, G):
        """phi (greedy) of genomes G on training-vocabulary puzzles and on unseen-value puzzles. Never used by the world."""
        W = unflatten(G)
        def phi(X, O):
            return (((scores(W, X).argmax(-1) == O.unsqueeze(0)).float().mean(1) - 0.2) / 0.8).numpy()
        return phi(self.Xt, self.Ot), phi(self.Xn, self.On)
