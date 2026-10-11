# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Capacity check for the attention brain, before any bottom-up evolution.
  1. generation-0 baseline: phi of 1000 random weight sets (criterion 3: should be ~<= phi_heur = 0.25)
  2. explicit-reward reachability: REINFORCE on the weights, 5 seeds; puzzles needed; phi on train and on
     scenes with never-seen symbol values (criterion 2)
"""

import json
import time
from pathlib import Path

import torch

from attention_brain import N_W, init_theta, one_hot2, phi, sample2, scores, unflatten

HERE = Path(__file__).resolve().parents[1] / "results"; HERE.mkdir(exist_ok=True)
torch.set_num_threads(4)

tg = torch.Generator().manual_seed(7)
a, o = sample2(2000, 5, tg, "train"); Xt, Ot = one_hot2(a), o
a, o = sample2(2000, 5, tg, "novel"); Xn, On = one_hot2(a), o
res = {"n_weights": N_W}

# 1. generation 0
th = init_theta(1000, torch.Generator().manual_seed(1))
p0 = torch.cat([phi(th[i:i + 100], Xt[:500], Ot[:500]) for i in range(0, 1000, 100)])
res["gen0"] = {"median": float(p0.median()), "q90": float(p0.quantile(0.9)), "max": float(p0.max()),
               "frac_above_heur": float((p0 > 0.25).float().mean())}
print("generation 0 (random weights):", res["gen0"], flush=True)

# 2. REINFORCE on the weights
P, steps, batch = 5, 3000, 256
gen = torch.Generator().manual_seed(0)
theta = init_theta(P, gen).requires_grad_(True)
opt = torch.optim.Adam([theta], lr=3e-3)
curve, hit = [], {0.5: [None] * P, 0.9: [None] * P}
t0 = time.time()
for step in range(steps + 1):
    if step % 100 == 0:
        pt, pn = phi(theta.detach(), Xt, Ot), phi(theta.detach(), Xn, On)
        curve.append({"step": step, "puzzles": step * batch, "train": pt.tolist(), "novel": pn.tolist()})
        for thr in hit:
            for i in range(P):
                if hit[thr][i] is None and pt[i] >= thr:
                    hit[thr][i] = step * batch
        print(f"step {step:5d}  phi train {pt.median():+.3f} [{pt.min():+.2f},{pt.max():+.2f}]  novel {pn.median():+.3f}  ({time.time() - t0:.0f}s)", flush=True)
    if step == steps:
        break
    a, o = sample2(batch, 5, gen, "train")
    logits = scores(unflatten(theta), one_hot2(a))                 # (P, B, 5)
    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    rew = (act == o.unsqueeze(0)).float()
    adv = rew - rew.mean(1, keepdim=True)
    loss = -(adv * dist.log_prob(act)).mean(1).sum() - 0.01 * dist.entropy().mean(1).sum()
    opt.zero_grad(); loss.backward(); opt.step()
res["rl"] = {"final_train": curve[-1]["train"], "final_novel": curve[-1]["novel"],
             "puzzles_to_0.5": hit[0.5], "puzzles_to_0.9": hit[0.9], "curve": curve}
torch.save(theta.detach(), HERE / "rl_theta.pt")
json.dump(res, open(HERE / "capacity.json", "w"), indent=1)
print("saved capacity.json")
