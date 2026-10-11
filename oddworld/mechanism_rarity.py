# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Follow-up: the evolved brains ignore WHERE attention goes (uniform attention gives the same phi), so is the
computation 'how common are my attribute values in this scene'? For each object, freq_j = fraction of objects
sharing its value on attribute j. Rules: the exact answer is the object with the smallest min_j freq_j (odd value
has freq 1/5, everything else >= 2/5); a cheaper 'sum of frequencies' rule is only approximate.
Per individual: how well do frequencies explain its scores (R^2), and which rule does its choice pattern match?"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

from mechanism import RUNS, forward
from attention_brain import one_hot2, sample2, unflatten

torch.set_num_threads(4)
g = torch.Generator().manual_seed(12)
for mode in ("train", "novel"):
    a, odd = sample2(1500, 5, g, mode)
    freq = torch.stack([(a[..., j].unsqueeze(2) == a[..., j].unsqueeze(1)).float().mean(-1) for j in range(3)], -1)  # (B,5,3)
    sum_rule = (-freq.sum(-1)).argmax(-1)
    min_rule = (-freq.min(-1).values + 1e-4 * torch.rand(freq.shape[:2], generator=g)).argmax(-1)
    acc = lambda pick: float(((pick == odd).float().mean() - 0.2) / 0.8)
    print(f"[{mode}] rule phi: min-frequency {acc(min_rule):+.3f}   sum-frequency {acc(sum_rule):+.3f}")
    for s in (0, 1, 2):
        G = torch.load(RUNS / f"selection_seed{s}" / "final_genomes.pt", weights_only=False)["genomes"]
        with torch.no_grad():
            sc, _ = forward(unflatten(G), one_hot2(a))                     # (P, B, 5)
        sc = sc - sc.mean(-1, keepdim=True)
        F = torch.cat([freq, torch.ones_like(freq[..., :1])], -1).reshape(-1, 4)       # per object: 3 freqs + bias
        F = F - 0  # design matrix shared across individuals
        r2, agree_sum, agree_min, w = [], [], [], []
        for p in range(len(G)):
            y = sc[p].reshape(-1)
            beta = torch.linalg.lstsq(F, y.unsqueeze(1)).solution.squeeze(1)
            r2.append(float(1 - ((F @ beta - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()))
            pick = sc[p].argmax(-1)
            agree_sum.append(float((pick == sum_rule).float().mean())); agree_min.append(float((pick == min_rule).float().mean()))
            w.append(beta[:3].numpy())
        w = np.array(w)
        print(f"  seed {s}: R^2(scores ~ attribute frequencies) median {np.median(r2):.2f} | choice agrees with sum-rule {np.mean(agree_sum):.2f}, "
              f"min-rule {np.mean(agree_min):.2f} | frequency weights (shape, colour, size) {np.median(w, 0).round(2)}")
