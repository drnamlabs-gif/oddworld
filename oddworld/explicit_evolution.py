# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Can EVOLUTION with an explicit fitness (before any implicit-selection world) find relational reasoning in the attention brain
with directly inherited weights (1056 genes)? Same budget per algorithm.
  es      OpenAI-ES (antithetic, Adam), the reachability upper bound for evolution
  ga      asexual: truncation selection (top 25%) + Gaussian mutation, the closest to ALife selection
  ga_sex  same, children = blend of two parents + mutation (recombination purges mutations)
Fitness = greedy phi on a fresh batch of 256 training puzzles each generation. Logs best/mean phi and novel-value phi.
  python explicit_evolution.py --algo ga --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import torch

from attention_brain import N_W, init_theta, one_hot2, phi, sample2

HERE = Path(__file__).resolve().parents[1] / "results"
ap = argparse.ArgumentParser()
ap.add_argument("--algo", choices=["es", "ga", "ga_sex"], default="ga")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gens", type=int, default=1500)
ap.add_argument("--pop", type=int, default=64)
ap.add_argument("--sigma", type=float, default=0.03)
args = ap.parse_args()
torch.set_num_threads(2)
gen = torch.Generator().manual_seed(args.seed)
tg = torch.Generator().manual_seed(7)
a, o = sample2(1000, 5, tg, "train"); Xt, Ot = one_hot2(a), o
a, o = sample2(1000, 5, tg, "novel"); Xn, On = one_hot2(a), o

P = args.pop
log, t0 = [], time.time()
if args.algo == "es":
    mu = init_theta(1, gen)[0]
    m, v, lr, sig = torch.zeros(N_W), torch.zeros(N_W), 0.02, 0.05
else:
    pop = init_theta(P, gen)
for g in range(args.gens + 1):
    a, o = sample2(256, 5, gen, "train"); X = one_hot2(a)
    if args.algo == "es":
        eps = torch.randn(P // 2, N_W, generator=gen)
        cand = torch.cat([mu + sig * eps, mu - sig * eps])
        f = phi(cand, X, o)
        rank = torch.argsort(torch.argsort(f)).float() / (P - 1) - 0.5            # centred ranks
        grad = ((rank[:P // 2] - rank[P // 2:]).unsqueeze(1) * eps).mean(0) / sig
        m = 0.9 * m + 0.1 * grad; v = 0.999 * v + 0.001 * grad ** 2
        mu = mu + lr * (m / (1 - 0.9 ** (g + 1))) / ((v / (1 - 0.999 ** (g + 1))).sqrt() + 1e-8)
        champ = mu.unsqueeze(0)
    else:
        f = phi(pop, X, o)
        order = torch.argsort(f, descending=True)
        elite = pop[order[:P // 4]]
        champ = pop[order[:1]]
        i1 = torch.randint(0, P // 4, (P - 1,), generator=gen)
        if args.algo == "ga_sex":
            i2 = torch.randint(0, P // 4, (P - 1,), generator=gen)
            kids = (elite[i1] + elite[i2]) / 2
        else:
            kids = elite[i1].clone()
        kids += args.sigma * torch.randn(kids.shape, generator=gen)
        pop = torch.cat([champ, kids])                                             # 1 elite kept
    if g % 25 == 0:
        ct, cn = float(phi(champ, Xt, Ot)[0]), float(phi(champ, Xn, On)[0])
        log.append({"gen": g, "evals": (g + 1) * P, "fit_mean": float(f.mean()), "fit_max": float(f.max()),
                    "champ_train": ct, "champ_novel": cn, "norm": float(champ.norm())})
        if g % 250 == 0:
            print(f"{args.algo} s{args.seed} gen {g:5d}  pop mean {f.mean():+.3f} max {f.max():+.3f}  champ train {ct:+.3f} novel {cn:+.3f}  |w| {champ.norm():.1f} ({time.time() - t0:.0f}s)", flush=True)
out = HERE / "explicit_evolution"; out.mkdir(parents=True, exist_ok=True)
json.dump(log, open(out / f"evo_{args.algo}_seed{args.seed}.json", "w"))
torch.save(champ, out / f"evo_{args.algo}_seed{args.seed}_champ.pt")
