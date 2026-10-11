# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""A structured, compact genome for a small attention brain, used only by neutrality_check.py (the control
experiments reported in the note's Section 5). The genome encodes structure and learning rules, not weights:
active heads, per-attribute initial gains, how similar keys are to queries, attention temperature, initial readout
on two per-object features (self-attention mass and attention entropy), and two learning rates. Weights are drawn
at birth from these distributions and updated during life by a reward-modulated rule (REINFORCE eligibility).
"""

import math
import sys
from pathlib import Path

import torch

from puzzle import one_hot, sample

H_MAX, D, F = 4, 4, 10
N_GENES = 31
N_GENES_WORLD = 32    # + gene 31: log choice "decisiveness" beta (policy = softmax(beta * scores))
BLOCKS = [(0, 4), (4, 8), (8, 10)]          # shape, color, size one-hot blocks


def decode(G):
    P = G.shape[0]
    return {
        "mask": (G[:, 0:4] > 0).float(),
        "gains": torch.nn.functional.softplus(G[:, 4:16]).view(P, H_MAX, 3),
        "tie": torch.sigmoid(G[:, 16]),
        "tau": torch.exp(G[:, 17:21]).clamp(0.05, 20.0),
        "r0": G[:, 21:29].view(P, H_MAX, 2),
        "eta_qk": torch.exp(G[:, 29]).clamp(1e-4, 10.0),
        "eta_r": torch.exp(G[:, 30]).clamp(1e-4, 10.0),
    }


def develop(dec, gen):
    """Birth: draw weights from the genome's distributions (stochastic)."""
    P = dec["mask"].shape[0]
    scale = torch.zeros(P, H_MAX, 1, F)
    for b, (lo, hi) in enumerate(BLOCKS):
        scale[..., lo:hi] = dec["gains"][:, :, b].view(P, H_MAX, 1, 1)
    Wq = torch.randn(P, H_MAX, D, F, generator=gen) * scale
    s = dec["tie"].view(P, 1, 1, 1)
    Wk = s * Wq + torch.sqrt(1 - s ** 2) * torch.randn(P, H_MAX, D, F, generator=gen) * scale
    r = dec["r0"] + 0.1 * torch.randn(P, H_MAX, 2, generator=gen)
    return Wq, Wk, r


def scores(Wq, Wk, r, dec, X):
    """Wq, Wk (P,H,D,F), r (P,H,2), X (B,5,F) -> scores (P,B,5)."""
    Q = torch.einsum("phdf,bnf->pbhnd", Wq, X)
    K = torch.einsum("phdf,bnf->pbhnd", Wk, X)
    S = torch.einsum("pbhnd,pbhmd->pbhnm", Q, K) / dec["tau"].view(-1, 1, H_MAX, 1, 1)
    A = torch.softmax(S, -1)
    self_mass = torch.diagonal(A, dim1=-2, dim2=-1)
    ent = -(A * torch.log(A.clamp_min(1e-9))).sum(-1) / math.log(5)
    feats = torch.stack([self_mass, ent], -1) * dec["mask"].view(-1, 1, H_MAX, 1, 1)
    return torch.einsum("pbhnk,phk->pbn", feats, r)


@torch.no_grad()
def phi_greedy(Wq, Wk, r, dec, X, odd):
    pick = scores(Wq, Wk, r, dec, X).argmax(-1)
    acc = (pick == odd.unsqueeze(0)).float().mean(1)
    return (acc - 0.2) / 0.8


def live(G, gen, steps=40, batch=8, clip=5.0):
    """One lifetime per genome: birth, then `steps` three-factor updates on fresh puzzles."""
    dec = decode(G)
    Wq, Wk, r = (t.requires_grad_(True) for t in develop(dec, gen))
    born = (Wq.detach().clone(), Wk.detach().clone(), r.detach().clone())
    for _ in range(steps):
        a, o, _ = sample(batch, 5, gen, "none")
        dist = torch.distributions.Categorical(logits=scores(Wq, Wk, r, dec, one_hot(a)))
        act = dist.sample()                                          # (P, B)
        rew = torch.where(act == o.unsqueeze(0), 1.0, -1.0)
        adv = rew - rew.mean(1, keepdim=True)                        # reward minus the individual's own baseline
        obj = (adv * dist.log_prob(act)).mean(1).sum()
        gq, gk, gr = torch.autograd.grad(obj, (Wq, Wk, r))
        with torch.no_grad():
            for W, g, eta in ((Wq, gq, dec["eta_qk"]), (Wk, gk, dec["eta_qk"]), (r, gr, dec["eta_r"])):
                n = g.flatten(1).norm(dim=1).clamp_min(1e-9)
                g = g * (clip / n).clamp(max=1.0).view(-1, *([1] * (g.dim() - 1)))
                W += eta.view(-1, *([1] * (g.dim() - 1))) * g
    return dec, born, (Wq.detach(), Wk.detach(), r.detach())


def hand_set():
    """A human's best guess: 3 heads, one per attribute (shape / color / size), Q = K (similarity
    attention), moderate temperature, readout rewarding self-attention ('nobody looks like me'),
    modest learning rates. The 4th head is off."""
    g = torch.zeros(N_GENES)
    g[0:4] = torch.tensor([1.0, 1.0, 1.0, -1.0])
    gains = torch.full((H_MAX, 3), -4.0)                             # softplus(-4) ~ 0.02: almost ignore
    gains[0, 0] = gains[1, 1] = gains[2, 2] = 2.0                    # softplus(2) ~ 2.1: head h reads attribute h
    g[4:16] = gains.flatten()
    g[16] = 6.0                                                      # tie ~ 1: Wk ~ Wq
    g[17:21] = math.log(1.0)
    r0 = torch.zeros(H_MAX, 2); r0[:3, 0] = 3.0                      # + self_mass
    g[21:29] = r0.flatten()
    g[29] = math.log(0.05); g[30] = math.log(0.3)
    return g


def random_genes(n, gen):
    """Prior for sampling: broad, uninformed ranges for every gene."""
    G = torch.empty(n, N_GENES)
    G[:, 0:4] = torch.randn(n, 4, generator=gen)
    G[:, 4:16] = torch.randn(n, 12, generator=gen)
    G[:, 16] = torch.randn(n, generator=gen)
    G[:, 17:21] = 0.7 * torch.randn(n, 4, generator=gen)
    G[:, 21:29] = torch.randn(n, 8, generator=gen)
    G[:, 29:31] = torch.empty(n, 2).uniform_(math.log(1e-3), math.log(1.0), generator=gen)
    return G
