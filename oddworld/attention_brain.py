# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The generic attention brain carried by every agent. It has no hand-built relational feature: the
architecture offers only general operations, so any relational computation has to be found by evolution.

Brain, per individual (all weights explicit tensors so a population runs batched):
    e_i  = We x_i                                  (D)
    head h:  A_h = softmax((Wq_h e)(Wk_h e)^T / sqrt(dh)),   o_h = A_h (Wv_h e)
    r_i  = e_i + sum_h Wo_h o_h,i                  (residual)
    s_i  = w2 . relu(W1 r_i + b1)                  (score per object)

The input code is one value larger than the world in shape and colour (5, 5, 2): shape 4 and colour 4 never occur
in the world, so their input weights are never under selection. "Novel" puzzles put these unseen values in every
scene: a brain that compares objects should still work, one that memorised value-specific features should not.
"""

import math
import sys
from pathlib import Path

import torch

from puzzle import _attr_column

TRAIN_SIZES = (4, 4, 2)
FULL_SIZES = (5, 5, 2)
F = sum(FULL_SIZES)                     # 12 input dims; dims 4 (shape 4) and 9 (colour 4) unseen in training
D, H, DH, FH = 12, 3, 4, 24


def sample2(batch, n, gen, mode="train"):
    """mode "train": values from TRAIN_SIZES only. "novel": FULL_SIZES and every scene contains an unseen value
    (shape 4 or colour 4) somewhere. Same well-posed construction as puzzle.py."""
    sizes = TRAIN_SIZES if mode == "train" else FULL_SIZES
    outs, odds, need = [], [], batch
    while need > 0:
        b = max(2 * need, 64)
        k = torch.randint(0, 3, (b,), generator=gen)
        oi = torch.randint(0, n, (b,), generator=gen)
        attrs = torch.stack([_attr_column(b, n, s, k == a, oi, gen) for a, s in enumerate(sizes)], 2)
        keep = torch.ones(b, dtype=torch.bool) if mode == "train" else ((attrs[..., 0] == 4) | (attrs[..., 1] == 4)).any(1)
        outs.append(attrs[keep]); odds.append(oi[keep]); need -= int(keep.sum())
    return torch.cat(outs)[:batch], torch.cat(odds)[:batch]


def one_hot2(attrs):
    return torch.cat([torch.nn.functional.one_hot(attrs[..., a], s).float() for a, s in enumerate(FULL_SIZES)], -1)


SHAPES = {"We": (D, F), "Wq": (H, DH, D), "Wk": (H, DH, D), "Wv": (H, DH, D), "Wo": (H, D, DH),
          "W1": (FH, D), "b1": (FH,), "w2": (FH,)}
N_W = sum(math.prod(s) for s in SHAPES.values())


def unflatten(theta):
    """theta (P, N_W) -> dict of (P, *shape)."""
    out, i = {}, 0
    for k, s in SHAPES.items():
        n = math.prod(s); out[k] = theta[:, i:i + n].view(-1, *s); i += n
    return out


def scores(W, X):
    """W: dict of (P, ...) weights, X (B, N, F) -> scores (P, B, N)."""
    e = torch.einsum("pdf,bnf->pbnd", W["We"], X)
    q = torch.einsum("phkd,pbnd->pbhnk", W["Wq"], e)
    k = torch.einsum("phkd,pbnd->pbhnk", W["Wk"], e)
    v = torch.einsum("phkd,pbnd->pbhnk", W["Wv"], e)
    A = torch.softmax(torch.einsum("pbhnk,pbhmk->pbhnm", q, k) / math.sqrt(DH), -1)
    o = torch.einsum("pbhnm,pbhmk->pbhnk", A, v)
    r = e + torch.einsum("phdk,pbhnk->pbnd", W["Wo"], o)
    hdn = torch.relu(torch.einsum("pjd,pbnd->pbnj", W["W1"], r) + W["b1"][:, None, None, :])
    return torch.einsum("pj,pbnj->pbn", W["w2"], hdn)


def init_theta(P, gen, scale=1.0):
    """Standard fan-in init (what 'random weights at generation 0' means here)."""
    parts = []
    for k, s in SHAPES.items():
        fan_in = s[-1] if len(s) > 1 else 1
        std = scale / math.sqrt(fan_in) if k != "b1" else 0.0
        parts.append(torch.randn(P, math.prod(s), generator=gen) * std)
    return torch.cat(parts, 1)


@torch.no_grad()
def phi(theta, X, odd):
    pick = scores(unflatten(theta), X).argmax(-1)
    return ((pick == odd.unsqueeze(0)).float().mean(1) - 0.2) / 0.8


def scores_paired(W, X):
    """Each individual p scores ITS OWN scene: W dict of (P, ...), X (P, N, F) -> (P, N). (Used in the world.)"""
    e = torch.einsum("pdf,pnf->pnd", W["We"], X)
    q = torch.einsum("phkd,pnd->phnk", W["Wq"], e)
    k = torch.einsum("phkd,pnd->phnk", W["Wk"], e)
    v = torch.einsum("phkd,pnd->phnk", W["Wv"], e)
    A = torch.softmax(torch.einsum("phnk,phmk->phnm", q, k) / math.sqrt(DH), -1)
    o = torch.einsum("phnm,phmk->phnk", A, v)
    r = e + torch.einsum("phdk,phnk->pnd", W["Wo"], o)
    hdn = torch.relu(torch.einsum("pjd,pnd->pnj", W["W1"], r) + W["b1"][:, None, :])
    return torch.einsum("pj,pnj->pn", W["w2"], hdn)
