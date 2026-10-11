# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Agent brains with the structured genome (structured_brain.py), used only by neutrality_check.py: 32 genes per
agent; at birth each agent draws its own weights from its genes and updates them after every eat from its own energy
change. Children inherit genes only.
"""

import sys
from pathlib import Path

import numpy as np
import torch

from puzzle import one_hot, sample
from structured_brain import D, F, H_MAX, decode, develop, random_genes, scores

N_GENES = 32


def random_structured_genomes(n, gen):
    G = torch.zeros(n, N_GENES)
    G[:, :31] = random_genes(n, gen)
    G[:, 31] = 0.5 * torch.randn(n, generator=gen)          # log beta
    return G


class StructuredBrains:
    def __init__(self, cap, seed, clip=5.0):
        self.cap, self.clip = cap, clip
        self.gen = torch.Generator().manual_seed(9000 + seed)
        self.Wq = torch.zeros(cap, H_MAX, D, F); self.Wk = torch.zeros(cap, H_MAX, D, F)
        self.r = torch.zeros(cap, H_MAX, 2)
        self.Wq0, self.Wk0, self.r0 = self.Wq.clone(), self.Wk.clone(), self.r.clone()
        self.base = torch.zeros(cap)
        self.world = None
        eg = torch.Generator().manual_seed(4242)
        a, o, _ = sample(500, 5, eg, "none")
        self.test_x, self.test_odd = one_hot(a), o

    def _dec(self, slots):
        return decode(self.world.genome[slots][:, :31])

    def beta(self, slots):
        return torch.exp(self.world.genome[slots][:, 31]).clamp(0.05, 20.0)

    def on_birth(self, s):
        sl = torch.tensor([s])
        Wq, Wk, r = develop(self._dec(sl), self.gen)
        self.Wq[s], self.Wk[s], self.r[s] = Wq[0], Wk[0], r[0]
        self.Wq0[s], self.Wk0[s], self.r0[s] = Wq[0], Wk[0], r[0]
        self.base[s] = 0.0

    @torch.no_grad()
    def __call__(self, slots, X):
        sl = torch.as_tensor(slots)
        dec = self._dec(sl)
        # every agent scores its own scene: run per-agent (P=n, B=1) via diagonal of a batched call
        sc = torch.stack([scores(self.Wq[s:s + 1], self.Wk[s:s + 1], self.r[s:s + 1],
                                 {k: v[i:i + 1] for k, v in dec.items()}, X[i:i + 1])[0, 0]
                          for i, s in enumerate(sl.tolist())])
        return torch.softmax(self.beta(sl).unsqueeze(1) * sc, -1).numpy()

    def learn(self, s, X, tried, action, reward):
        sl = torch.tensor([s])
        dec = self._dec(sl)
        Wq = self.Wq[s:s + 1].clone().requires_grad_(True)
        Wk = self.Wk[s:s + 1].clone().requires_grad_(True)
        r = self.r[s:s + 1].clone().requires_grad_(True)
        logits = self.beta(sl).unsqueeze(1) * scores(Wq, Wk, r, dec, X.unsqueeze(0))[:, 0]
        logits = logits.masked_fill(torch.as_tensor(tried).unsqueeze(0), -1e9)
        logp = torch.log_softmax(logits, -1)[0, action]
        adv = reward - float(self.base[s])
        self.base[s] = 0.9 * self.base[s] + 0.1 * reward
        gq, gk, gr = torch.autograd.grad(adv * logp, (Wq, Wk, r))
        with torch.no_grad():
            for W, g, eta in ((self.Wq, gq, dec["eta_qk"]), (self.Wk, gk, dec["eta_qk"]), (self.r, gr, dec["eta_r"])):
                n = float(g.norm())
                if n > self.clip:
                    g = g * (self.clip / n)
                W[s] += float(eta) * g[0]

    @torch.no_grad()
    def probe(self, slots):
        """Offline test (greedy, 500 fixed puzzles): phi of the birth weights ('innate') and of the
        current, learned weights. Never used for selection."""
        sl = torch.as_tensor(slots)
        dec = self._dec(sl)

        def phi(Wq, Wk, r):
            pick = scores(Wq, Wk, r, dec, self.test_x).argmax(-1)
            return ((pick == self.test_odd.unsqueeze(0)).float().mean(1) - 0.2) / 0.8

        return phi(self.Wq0[sl], self.Wk0[sl], self.r0[sl]).numpy(), phi(self.Wq[sl], self.Wk[sl], self.r[sl]).numpy()
