# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""One Oddworld run: implicit selection (energy = the true payoff of each eat) or the lottery control
(energy = the payoff of a random untried object, independent of the choice). Defaults are the configuration of
the technical note: 19 clusters on a hexagonal lattice, density-dependent upkeep (K = 115, basal 0.02), sexual
reproduction (mate radius 4, blend + Gaussian mutation sigma = 0.03), greedy choice with inhibition of return.
Writes ../runs/<condition>_seed<seed>/log.jsonl (one row per 2,000 steps) and final_genomes.pt.

  python run.py --condition selection --seed 0
  python run.py --condition neutral --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from agent_brains import N_GENES, AttentionBrains, random_genomes
from world import DT, VecWorld, Economy

RUNS = Path(__file__).resolve().parents[1] / "runs"
LOG_EVERY, PROBE_EVERY = 2000, 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["selection", "neutral"], default="selection")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2000000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.set_num_threads(2)

    out = RUNS / f"{args.condition}_seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    brains = AttentionBrains()
    init = random_genomes(80, torch.Generator().manual_seed(100 + args.seed))
    eco = Economy(basal=0.02, neutral=(args.condition == "neutral"), density=True, K=115.0, neutral_mode="lottery")
    w = VecWorld(19, 150, len(init), brains, layout="hex", mate_radius=4.0, seed=args.seed, genome_dim=N_GENES,
                 economy=eco, choice="greedy", mutation_sigma=0.03, init_genomes=init, reproduction="sexual")
    brains.world = w

    log = open(out / "log.jsonl", "w")
    t0, last_e, last_b, last_d, last_fc = time.time(), 0, 0, 0, 0
    for t in range(1, args.steps + 1):
        w.step()
        if t % LOG_EVERY == 0:
            L = np.flatnonzero(w.alive)
            fc = [c for _, c in w.first_choices[last_fc:]]
            ev = [e[2] for e in w.events[last_e:]]
            row = {"t": t, "sim_s": t * DT, "pop": len(L),
                   "births": len(w.births) - last_b, "deaths": len(w.deaths) - last_d,
                   "phi_first": (np.mean(fc) - 0.2) / 0.8 if fc else None, "n_first": len(fc),
                   "phi_events": (np.mean(ev) - 0.2) / 0.8 if ev else None,
                   "gen_mean": float(w.generation[L].mean()) if len(L) else None,
                   "gen_max": int(w.generation[L].max()) if len(L) else None,
                   "lineages": int(len(np.unique(w.lineage[L]))) if len(L) else 0,
                   "energy_mean": float(w.energy[L].mean()) if len(L) else None}
            if len(L):
                Z = w.genome[torch.as_tensor(L)]
                row["znorm_mean"] = float(Z.norm(dim=1).mean())
                row["diversity"] = float(torch.pdist(Z[:60]).mean()) if len(L) > 1 else 0.0
                if t % PROBE_EVERY == 0:
                    pick = torch.as_tensor(np.random.default_rng(t).choice(L, min(40, len(L)), replace=False))
                    pt, pn = brains.probe(w.genome[pick])
                    row["innate_phi_mean"], row["innate_phi_max"] = float(pt.mean()), float(pt.max())
                    row["novel_phi_mean"], row["novel_phi_max"] = float(pn.mean()), float(pn.max())
            last_e, last_b, last_d, last_fc = len(w.events), len(w.births), len(w.deaths), len(w.first_choices)
            log.write(json.dumps(row) + "\n"); log.flush()
            if not len(L):
                break
        if t % 100000 == 0:
            print(f"{args.condition} seed {args.seed}: t={t} pop={w.population} ({time.time() - t0:.0f}s wall)", flush=True)
    L = np.flatnonzero(w.alive)
    if len(L):
        torch.save({"genomes": w.genome[torch.as_tensor(L)], "lineage": w.lineage[L], "generation": w.generation[L]}, out / "final_genomes.pt")
    print(f"done {args.condition} seed {args.seed}: pop={len(L)} wall={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
