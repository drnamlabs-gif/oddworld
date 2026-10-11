# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""What did evolution build? Mechanism analysis of the final genomes of the selection runs
(every living individual, 3 seeds). Offline only, on fixed puzzle sets.

  1. accuracy by odd attribute (shape / colour / size), training symbols vs never-seen symbols
  2. causal ablations: each head removed; attention replaced by uniform (no relation info) or by identity (self only)
  3. what each head attends by: mean attention between two objects that share vs differ on each attribute
  4. the odd object's self-attention vs the common objects' (a hand-built 'self-attention mass' feature, rediscovered?)
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

from attention_brain import DH, H, one_hot2, sample2, unflatten

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"
ATTR = ["shape", "colour", "size"]
torch.set_num_threads(4)


def forward(W, X, mode="full", drop_head=None):
    e = torch.einsum("pdf,bnf->pbnd", W["We"], X)
    q = torch.einsum("phkd,pbnd->pbhnk", W["Wq"], e)
    k = torch.einsum("phkd,pbnd->pbhnk", W["Wk"], e)
    v = torch.einsum("phkd,pbnd->pbhnk", W["Wv"], e)
    A = torch.softmax(torch.einsum("pbhnk,pbhmk->pbhnm", q, k) / math.sqrt(DH), -1)
    n = X.shape[1]
    if mode == "uniform":
        A = torch.full_like(A, 1 / n)
    elif mode == "identity":
        A = torch.eye(n).expand_as(A)
    o = torch.einsum("pbhnm,pbhmk->pbhnk", A, v)
    if drop_head is not None:
        o = o.clone(); o[:, :, drop_head] = 0
    r = e + torch.einsum("phdk,pbhnk->pbnd", W["Wo"], o)
    hdn = torch.relu(torch.einsum("pjd,pbnd->pbnj", W["W1"], r) + W["b1"][:, None, None, :])
    return torch.einsum("pj,pbnj->pbn", W["w2"], hdn), A


def phi_of(s, odd):
    return ((s.argmax(-1) == odd.unsqueeze(0)).float().mean(1) - 0.2) / 0.8          # per individual


@torch.no_grad()
def analyse(G, sets):
    W = unflatten(G)
    out = {}
    for name, (a, odd, kk) in sets.items():
        X = one_hot2(a)
        s, A = forward(W, X)
        out[f"phi_{name}"] = float(phi_of(s, odd).mean())
        out[f"phi_{name}_by_attr"] = {ATTR[j]: float(phi_of(s[:, kk == j], odd[kk == j]).mean()) for j in range(3)}
        if name != "train":
            continue
        out["ablate_uniform"] = float(phi_of(forward(W, X, "uniform")[0], odd).mean())
        out["ablate_identity"] = float(phi_of(forward(W, X, "identity")[0], odd).mean())
        out["ablate_head"] = [float(phi_of(forward(W, X, drop_head=h)[0], odd).mean()) for h in range(H)]
        # attention between two DIFFERENT objects, split by whether they share each attribute (mean over individuals)
        offd = ~torch.eye(5, dtype=torch.bool)
        prof = []
        for h in range(H):
            Ah = A[:, :, h]                                                        # (P, B, n, n)
            row = {}
            for j in range(3):
                same = (a[..., j].unsqueeze(2) == a[..., j].unsqueeze(1)) & offd     # (B, n, n)
                diff = (a[..., j].unsqueeze(2) != a[..., j].unsqueeze(1))
                row[ATTR[j]] = float(Ah[:, same].mean() / Ah[:, diff].mean())         # >1: attends to objects alike on j
            sm = torch.diagonal(Ah, dim1=-2, dim2=-1)                               # (P, B, n) self-attention
            isodd = torch.nn.functional.one_hot(odd, 5).bool()
            row["self_odd"] = float(sm[:, isodd].mean()); row["self_common"] = float(sm[:, ~isodd].mean())
            row["drop_if_removed"] = out["phi_train"] - out["ablate_head"][h]
            prof.append(row)
        out["heads"] = prof
    return out


def main():
    g = torch.Generator().manual_seed(11)
    sets = {}
    for name, mode in (("train", "train"), ("novel", "novel")):
        a, odd = sample2(1500, 5, g, mode)
        # recover the odd attribute: the one on which the answer is unique
        kk = torch.zeros(len(a), dtype=torch.long)
        for j in range(3):
            cnt = (a[..., j] == a[torch.arange(len(a)), odd, j].unsqueeze(1)).sum(1)
            kk[cnt == 1] = j
        sets[name] = (a, odd, kk)
    res = {}
    for cond in ("selection", "neutral"):
        for s in (0, 1, 2):
            G = torch.load(RUNS / f"{cond}_seed{s}" / "final_genomes.pt", weights_only=False)["genomes"]  # our own file
            r = analyse(G, sets); r["n"] = len(G); res[f"{cond}_s{s}"] = r
            print(f"\n== {cond} seed {s} (n={len(G)}) ==")
            print(f"  phi train {r['phi_train']:+.3f}  novel {r['phi_novel']:+.3f}")
            print("  by odd attribute, train:", {k: round(v, 2) for k, v in r["phi_train_by_attr"].items()},
                  " novel:", {k: round(v, 2) for k, v in r["phi_novel_by_attr"].items()})
            print(f"  attention -> uniform: {r['ablate_uniform']:+.3f}   -> identity (self only): {r['ablate_identity']:+.3f}")
            for h, row in enumerate(r["heads"]):
                print(f"  head {h}: remove -> phi drops {row['drop_if_removed']:+.3f} | alike/unlike attention ratio "
                      + " ".join(f"{a}={row[a]:.2f}" for a in ATTR) + f" | self-attn odd {row['self_odd']:.2f} vs common {row['self_common']:.2f}")
    json.dump(res, open(HERE.parent / "results" / "mechanism.json", "w"), indent=1)


if __name__ == "__main__":
    main()
