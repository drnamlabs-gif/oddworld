# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Is the neutral control really neutral? Per agent: innate phi at birth vs number of offspring and lifespan.
Spearman rho should be ~0 in neutral, > 0 under selection.
  python neutrality_check.py --condition neutral --density 1 --K 115 --basal 0.06
"""
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr

from structured_agent_brains import N_GENES, StructuredBrains, random_structured_genomes
from world import VecWorld, Economy

ap = argparse.ArgumentParser()
ap.add_argument("--condition", default="neutral"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--density", type=int, default=1); ap.add_argument("--K", type=float, default=115)
ap.add_argument("--basal", type=float, default=0.06); ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--mode", default="shuffle")
a = ap.parse_args()
torch.set_num_threads(2)
br = StructuredBrains(150, a.seed)
eco = Economy(basal=a.basal, neutral=a.condition == "neutral", density=bool(a.density), K=a.K, neutral_mode=a.mode)
w = VecWorld(19, 150, 80, br, seed=a.seed, genome_dim=N_GENES, economy=eco, choice="sample_ior", mutation_sigma=0.15,
             init_genomes=random_structured_genomes(80, torch.Generator().manual_seed(100 + a.seed)), layout="hex")
uid = np.arange(80); nxt = 80                 # unique id per individual (slots get reused)
phi, kids, born, died = {}, {}, {}, {}
ph, _ = br.probe(torch.arange(80))
for i in range(80): phi[i], kids[i], born[i] = float(ph[i]), 0, 0
uid = np.concatenate([uid, np.full(70, -1)])
nb, nd = 0, 0
for t in range(a.steps):
    w.step()
    for (_, s, *_r) in w.deaths[nd:]:
        died[uid[s]] = t
    nd = len(w.deaths)
    for (_, c, p) in w.births[nb:]:
        kids[uid[p]] += 1
        uid[c] = nxt; phi[nxt] = float(br.probe(torch.tensor([c]))[0][0]); kids[nxt] = 0; born[nxt] = t; nxt += 1
    nb = len(w.births)
done = [i for i in died]                      # completed lives only
x = np.array([phi[i] for i in done]); k = np.array([kids[i] for i in done]); L = np.array([died[i] - born[i] for i in done])
r1, p1 = spearmanr(x, k); r2, p2 = spearmanr(x, L)
print(f"{a.condition} ({a.mode}): {len(done)} completed lives, final pop {w.population}")
print(f"  innate phi vs offspring: rho={r1:+.3f} (p={p1:.2g});  vs lifespan: rho={r2:+.3f} (p={p2:.2g})")
q = np.quantile(x, [0.25, 0.75])
print(f"  mean offspring, phi bottom quartile {k[x <= q[0]].mean():.2f} vs top quartile {k[x >= q[1]].mean():.2f}")
