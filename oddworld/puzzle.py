# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""The odd-one-out puzzle over symbolic objects.

Each episode shows N objects with 3 discrete attributes (shape 4, color 4,
size 2). One attribute k is the "odd"
dimension: all objects share value v on k except one, which has v' != v.
On every other attribute, values are assigned so that NO object is unique
(either all equal, or split into groups of size >= 2). So the answer is
well-posed (exactly one object is unique on anything), and no absolute
feature value identifies it: v, v' and k are all random per episode. A policy
that scores each object on its own features alone cannot beat chance; it has
to compare objects with each other.

Not embodied: there's no movement, the agent picks one object directly. That
isolates the question "can attention represent the relation" from navigation.
"""

import torch

ATTR_SIZES = (4, 4, 2)          # shape, color, size
ATTR_NAMES = ("shape", "color", "size")
FEAT_DIM = sum(ATTR_SIZES)      # one-hot concat = 10

# (shape, color) combos never shown in training; used for the compositional test
HOLDOUT_COMBOS = ((0, 3), (1, 2), (2, 1), (3, 0))


def _attr_column(batch, n, n_vals, is_odd, odd_idx, gen):
    """Values (B, N) for one attribute. Odd rows: one unique value at odd_idx.
    Other rows: all-same or a (m, n-m) split with m in [2, n-2], so nobody is unique."""
    v = torch.randint(0, n_vals, (batch,), generator=gen)
    v_alt = (v + torch.randint(1, n_vals, (batch,), generator=gen)) % n_vals   # != v

    odd_vals = v.unsqueeze(1).expand(batch, n).clone()
    odd_vals[torch.arange(batch), odd_idx] = v_alt

    perm = torch.argsort(torch.rand(batch, n, generator=gen), dim=1)
    m = torch.randint(2, n - 1, (batch,), generator=gen)                      # group size of u1
    rank = torch.empty_like(perm)
    rank.scatter_(1, perm, torch.arange(n).expand(batch, n))
    all_same = torch.rand(batch, generator=gen) < 0.5
    second = torch.where(all_same, v, v_alt)
    split_vals = torch.where(rank < m.unsqueeze(1), v.unsqueeze(1), second.unsqueeze(1))

    return torch.where(is_odd.unsqueeze(1), odd_vals, split_vals)


def sample(batch, n, gen, holdout_mode="exclude"):
    """Returns attrs (B, N, 3) long, odd_idx (B,), odd_attr (B,).

    holdout_mode: "exclude" -> no object anywhere uses a HOLDOUT combo (training);
                  "answer"  -> the answer object always has a HOLDOUT combo (test);
                  "none"    -> no constraint.
    """
    out_attrs, out_odd, out_k = [], [], []
    need = batch
    holdout = torch.tensor(HOLDOUT_COMBOS)
    while need > 0:
        b = max(need * 2, 64)
        k = torch.randint(0, 3, (b,), generator=gen)
        odd_idx = torch.randint(0, n, (b,), generator=gen)
        cols = [_attr_column(b, n, s, k == a, odd_idx, gen) for a, s in enumerate(ATTR_SIZES)]
        attrs = torch.stack(cols, dim=2)                                       # (b, n, 3)

        combo = attrs[..., 0] * 4 + attrs[..., 1]                              # (b, n)
        held = torch.isin(combo, holdout[:, 0] * 4 + holdout[:, 1])
        if holdout_mode == "exclude":
            keep = ~held.any(dim=1)
        elif holdout_mode == "answer":
            keep = held[torch.arange(b), odd_idx]
        else:
            keep = torch.ones(b, dtype=torch.bool)

        out_attrs.append(attrs[keep]); out_odd.append(odd_idx[keep]); out_k.append(k[keep])
        need -= int(keep.sum())

    attrs = torch.cat(out_attrs)[:batch]
    odd = torch.cat(out_odd)[:batch]
    k = torch.cat(out_k)[:batch]
    # shuffle token order isn't needed (odd_idx already uniform), but keep it explicit:
    return attrs, odd, k


def one_hot(attrs):
    parts = [torch.nn.functional.one_hot(attrs[..., a], s).float() for a, s in enumerate(ATTR_SIZES)]
    return torch.cat(parts, dim=-1)                                            # (B, N, 10)


def check_well_posed(attrs, odd):
    """Every episode: exactly one object is unique on some attribute, and it's `odd`."""
    B, N, A = attrs.shape
    unique_any = torch.zeros(B, N, dtype=torch.bool)
    for a in range(A):
        same = (attrs[..., a].unsqueeze(2) == attrs[..., a].unsqueeze(1)).sum(-1)   # (B, N)
        unique_any |= same == 1
    ok = (unique_any.sum(1) == 1) & unique_any[torch.arange(B), odd]
    return bool(ok.all()), float(ok.float().mean())


def heuristic_accuracies(attrs, odd, gen):
    """Accuracy of non-relational policies (score each object on its own features,
    random tie-break). The best of these defines the heuristic ceiling φ_heur."""
    B, N, _ = attrs.shape
    noise = torch.rand(B, N, generator=gen) * 1e-3
    res = {"random": float((torch.randint(0, N, (B,), generator=gen) == odd).float().mean())}
    for a, s in enumerate(ATTR_SIZES):
        for val in range(s):
            score = (attrs[..., a] == val).float() + noise
            res[f"prefer_{ATTR_NAMES[a]}={val}"] = float((score.argmax(1) == odd).float().mean())
    res["largest"] = float(((attrs[..., 2].float() + noise).argmax(1) == odd).float().mean())
    res["smallest"] = float(((-attrs[..., 2].float() + noise).argmax(1) == odd).float().mean())
    return res
