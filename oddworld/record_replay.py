# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Re-run an Oddworld run EXACTLY (same construction and seeds as run.py, so the world is the same
trajectory as the logged run) while recording what the gif and the replay viewer need:
  - every agent's phi at birth (training symbols and never-seen symbols), measured offline, never used by the world
  - a timelapse snapshot every 4000 steps (positions, phi, lineage, generation)
  - the full genome table every 250k steps (so clips can be recorded from before / during / after the transition)

  python record_replay.py run --condition selection --seed 0
  python record_replay.py clip --condition selection --seed 0 --at 1000000      # 120 s fine-grained clip from a snapshot
"""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from agent_brains import N_GENES, AttentionBrains, random_genomes
from world import DT, RING_R, VecWorld, Economy

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "runs" / "replay"


def make_world(condition, seed, init=None, world_seed=None, n_clusters=19, cap=150):
    brains = AttentionBrains()
    # K scales with the number of clusters so agents per cluster (~6) stays as in the 19-cluster evolution world
    eco = Economy(basal=0.02, neutral=(condition == "neutral"), density=True, K=115.0 * n_clusters / 19, neutral_mode="lottery")
    if init is None:
        init = random_genomes(80, torch.Generator().manual_seed(100 + seed))
    w = VecWorld(n_clusters, cap, len(init), brains, layout="hex", mate_radius=4.0, seed=seed if world_seed is None else world_seed,
                 genome_dim=N_GENES, economy=eco, choice="greedy", mutation_sigma=0.03, init_genomes=init, reproduction="sexual")
    brains.world = w
    return w, brains


class PhiTracker:
    """phi (train, novel) of every agent, measured once at birth."""
    def __init__(self, w, brains):
        self.w, self.b = w, brains
        self.phi = np.zeros((w.cap, 2))
        self.nb = 0
        self.measure(np.flatnonzero(w.alive))

    def measure(self, slots):
        if len(slots):
            pt, pn = self.b.probe(self.w.genome[torch.as_tensor(slots)])
            self.phi[slots, 0], self.phi[slots, 1] = pt, pn

    def after_step(self):
        new = sorted({b[1] for b in self.w.births[self.nb:]})
        self.nb = len(self.w.births)
        self.measure(np.array(new, dtype=np.int64))
        return new


def run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    w, b = make_world(a.condition, a.seed)
    tr = PhiTracker(w, b)
    snaps, genomes = [], {}
    for t in range(0, a.steps + 1):
        if t:
            w.step(); tr.after_step()
        L = np.flatnonzero(w.alive)
        if t % 4000 == 0:
            snaps.append({"t": t, "x": w.x[L].round(2).tolist(), "z": w.z[L].round(2).tolist(),
                          "phi": tr.phi[L, 0].round(3).tolist(), "novel": tr.phi[L, 1].round(3).tolist(),
                          "lin": w.lineage[L].tolist(), "gen": w.generation[L].tolist()})
        if t % 250000 == 0 and len(L):
            genomes[t] = {"genomes": w.genome[torch.as_tensor(L)].clone(), "lineage": w.lineage[L].copy(),
                          "generation": w.generation[L].copy()}
        if t % 100000 == 0:
            print(f"{a.condition} s{a.seed}: t={t} pop={len(L)} phi={tr.phi[L, 0].mean() if len(L) else 0:.3f}", flush=True)
        if not len(L):
            break
    json.dump({"condition": a.condition, "seed": a.seed, "dt": DT, "every": 4000, "cx": w.cx.tolist(), "cz": w.cz.tolist(),
               "world_R": w.world_R, "snaps": snaps}, open(OUT / f"timelapse_{a.condition}_seed{a.seed}.json", "w"))
    torch.save(genomes, OUT / f"genomes_{a.condition}_seed{a.seed}.pt")
    print("saved", flush=True)


def b64(x):
    return base64.b64encode(np.ascontiguousarray(x).tobytes()).decode()


def clip(a):
    snap = torch.load(OUT / f"genomes_{a.condition}_seed{a.seed}.pt", weights_only=False)[a.at]   # our own file
    # a smaller world for viewing: a random sample of the population at that time, same density per cluster
    if a.pick == "top":
        # the n individuals with the highest offline phi (probe), for a demonstration of what evolution reached
        phi_all, _ = AttentionBrains().probe(snap["genomes"])
        pick = torch.as_tensor(np.argsort(-phi_all)[:a.n].copy())
    else:
        pick = torch.randperm(len(snap["genomes"]), generator=torch.Generator().manual_seed(a.seed + a.at))[:a.n]
    G = snap["genomes"][pick]
    w, b = make_world(a.condition, a.seed, init=G, world_seed=a.seed + 50, n_clusters=a.clusters, cap=a.cap)
    w.lineage[:len(G)] = snap["lineage"][pick.numpy()]; w.generation[:len(G)] = snap["generation"][pick.numpy()]
    w.age[:len(G)] = w.rng.uniform(0, w.eco.max_age / 2, len(G))
    tr = PhiTracker(w, b)
    for _ in range(a.warmup):
        w.step(); tr.after_step()
    slots, genomes = [], []
    def info(frame, ss):
        for s in ss:
            slots.append({"f": frame, "s": int(s), "phi": round(float(tr.phi[s, 0]), 3), "novel": round(float(tr.phi[s, 1]), 3),
                          "lin": int(w.lineage[s]), "gen": int(w.generation[s]), "gi": len(genomes)})
            genomes.append(w.genome[s].numpy().astype(np.float32))   # the viewer re-runs this agent's brain in JS
    info(0, np.flatnonzero(w.alive))
    e0, d0, b0 = len(w.events), len(w.deaths), len(w.births)
    n_frames = a.steps // a.every
    # per agent per frame: x*100, z*100, heading*1000, energy*1000, home cluster, target object (-1 none), tried bitmask
    pos = np.full((n_frames, w.cap, 7), -32768, dtype=np.int16)
    clusters0 = [{"attrs": w.attrs[c].tolist(), "ang": w.oang[c].round(4).tolist()} for c in range(w.M)]
    regens, eats, births, deaths = [], [], [], []
    vers = w.version.copy()
    for f in range(n_frames):
        for _ in range(a.every):
            w.step(); info(f, tr.after_step())
        L = np.flatnonzero(w.alive)
        pos[f, L, 0] = np.round(w.x[L] * 100); pos[f, L, 1] = np.round(w.z[L] * 100)
        pos[f, L, 2] = np.round(((w.h[L] + np.pi) % (2 * np.pi) - np.pi) * 1000)
        pos[f, L, 3] = np.round(np.clip(w.energy[L], 0, 3) * 1000)
        pos[f, L, 4] = w.home[L]; pos[f, L, 5] = w.target[L]
        pos[f, L, 6] = (w.tried[L] * (1 << np.arange(5))).sum(1)
        for c in np.flatnonzero(w.version != vers):
            regens.append({"f": f, "c": int(c), "attrs": w.attrs[c].tolist(), "ang": w.oang[c].round(4).tolist()})
        vers = w.version.copy()
        eats += [[f, int(s), int(ok), c, o] for (_t, s, ok, _k, c, o) in w.events[e0:]]
        deaths += [[f, int(d[1])] for d in w.deaths[d0:]]
        births += [[f, int(bb[1]), int(bb[2])] for bb in w.births[b0:]]
        e0, d0, b0 = len(w.events), len(w.deaths), len(w.births)
    rows = [json.loads(l) for l in open(HERE.parent / "runs" / f"{a.condition}_seed{a.seed}" / "log.jsonl")]
    probes = [r for r in rows if "innate_phi_mean" in r]
    curve = {"sim_s": [r["sim_s"] for r in rows[::10]], "pop": [r["pop"] for r in rows[::10]],
             "probe_s": [r["sim_s"] for r in probes], "innate": [round(r["innate_phi_mean"], 3) for r in probes],
             "novel": [round(r["novel_phi_mean"], 3) for r in probes]}
    gen_at = next((r["gen_mean"] for r in rows if r["t"] >= a.at), rows[-1]["gen_mean"]) if a.at else 0
    # reference scores from PyTorch, so the viewer's JS forward can be checked against them
    from agent_brains import to_brain_input
    from attention_brain import scores_paired, unflatten
    gref = torch.tensor(np.stack(genomes[:3]))
    xref = to_brain_input(w.X[0]).unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        ref = {"attrs": w.attrs[0].tolist(), "scores": scores_paired(unflatten(gref), xref).numpy().round(5).tolist()}
    out = OUT / f"clip_{a.condition}_seed{a.seed}_t{a.at}_n{a.n}{a.suffix}.json"
    json.dump({"condition": a.condition, "seed": a.seed, "at_step": a.at, "at_s": a.at * DT, "gen_at": round(gen_at or 0),
               "dt": DT * a.every, "n_frames": n_frames, "cap": w.cap, "n_clusters": w.M, "n_sample": a.n, "ring_R": RING_R, "cx": w.cx.round(3).tolist(), "cz": w.cz.round(3).tolist(),
               "world_R": w.world_R, "clusters0": clusters0, "regens": regens, "eats": eats, "births": births, "deaths": deaths,
               "slots": slots, "pos": b64(pos), "stride": 7, "genomes": b64(np.stack(genomes)), "n_w": int(genomes[0].size),
               "check": ref, "curve": curve}, open(out, "w"))
    print("saved", out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "clip"])
    ap.add_argument("--condition", default="selection"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None); ap.add_argument("--at", type=int, default=2000000)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--pick", choices=["random", "top"], default="random")
    ap.add_argument("--clusters", type=int, default=7); ap.add_argument("--n", type=int, default=40); ap.add_argument("--cap", type=int, default=50); ap.add_argument("--warmup", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.mode == "run":
        a.steps = a.steps or 2000000; run(a)
    else:
        a.steps = a.steps or 2400; clip(a)
