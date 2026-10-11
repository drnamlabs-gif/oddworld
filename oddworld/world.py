# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Vectorized embodied world for ALife: same rules as embodied/world.py and the 3D
viewer (odd-one-out clusters, solid bodies with contact resolution, scripted orbit
navigation, eat-on-touch, scene regenerates only when its odd object is eaten),
but arrays over a fixed-capacity population with an alive mask, so 60-100 agents
and hundreds of generations are affordable.

The ALife layer (energy, death, asexual reproduction with mutation) is on top and
can be switched off (alife=False) to compare against the old world with fixed brains.

Brains are pluggable: brain_fn(slots, X) -> p, where X is (n, 5, 10) one-hot scenes.
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from puzzle import one_hot, sample

N_OBJ = 5
RING_R = 3.6
RA = 0.95
ORBIT_R = RING_R + 2.4
OBJ_R = np.array([0.75, 0.8, 0.8, 0.92])
SIZE_SCALE = np.array([0.8, 1.25])
DT = 0.05
REGEN_STEPS = 14
COOLDOWN = 0.8
SWITCH_P = 0.35
CHANCE = 1 / N_OBJ


def wrap_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Economy:
    """Energy rules (ALife layer). Units: energy; rates per second of sim time."""
    gain_correct: float = 1.0
    loss_wrong: float = 0.15
    basal: float = 0.0035          # per second, AT population K; scales with pop / K (crowding)
    K: float = 80.0                # soft carrying capacity: random agents break even here
    move: float = 0.003            # per unit distance travelled
    start_energy: float = 1.0
    repro_threshold: float = 2.0
    child_energy: float = 0.9      # transferred from parent to child
    repro_overhead: float = 0.1
    max_energy: float = 3.0
    max_age: float = 400.0         # seconds
    neutral: bool = False          # control: payoff decoupled from the agent's choice (see neutral_mode)
    neutral_mode: str = "lottery"  # "lottery": eater gets a random-pick payoff (food not conserved);
                                   # "shuffle": the TRUE payoff of the eat goes to a random living agent (food conserved)
    density: bool = True           # True: upkeep scales with pop / K. False: constant upkeep, carrying capacity set by food


class VecWorld:
    def __init__(self, n_clusters, cap, n_init, brain_fn, seed, genome_dim=0, economy=None,
                 alife=True, mutation_sigma=0.3, init_genomes=None, choice="greedy", reproduction="asexual",
                 mate_radius=4.0, mate_threshold=1.5, layout="ring"):
        """choice="greedy": eat the highest-scoring object not yet tried in this scene (inhibition of
        return: an object eaten wrongly is never chosen again in the same scene). Genome differences show
        fully in behaviour, and no scene can trap an agent. choice="sample": the old rule (sample from p)."""
        self.choice = choice
        # reproduction="sexual": two nearby agents both above mate_threshold make one child whose genome is the
        # MEAN of theirs plus mutation (intermediate recombination); each parent pays half the child's energy.
        self.reproduction = reproduction
        self.mate_radius, self.mate_threshold = mate_radius, mate_threshold
        self.rng = np.random.default_rng(seed)
        self.tgen = torch.Generator().manual_seed(seed)
        self.brain_fn = brain_fn
        self.alife = alife
        self.eco = economy or Economy()
        self.sigma = mutation_sigma
        M = self.M = n_clusters
        self.cap = cap

        if layout == "hex":
            # clusters on a hexagonal lattice (M nearest sites: 1, 7, 19 are full hexagons), spaced so orbits never overlap;
            # agents move between lattice neighbours
            sp = 2 * (ORBIT_R + RA) + 2.1
            sites = [(sp * (q + r / 2), sp * r * math.sqrt(3) / 2) for q in range(-4, 5) for r in range(-4, 5)]
            sites.sort(key=lambda p: (round(math.hypot(*p), 3), math.atan2(p[1], p[0])))
            self.cx, self.cz = np.array([p[0] for p in sites[:M]]), np.array([p[1] for p in sites[:M]])
            dd = np.hypot(self.cx[:, None] - self.cx[None, :], self.cz[:, None] - self.cz[None, :])
            self.neighbors = [np.flatnonzero((dd[c] > 0) & (dd[c] < sp * 1.05)) for c in range(M)]
            self.ring_R = float(np.hypot(self.cx, self.cz).max())
            self.world_R = self.ring_R + ORBIT_R + 4.0
        else:
            # clusters on a ring, spaced so orbits never overlap
            self.ring_R = max(14.0, (2 * (ORBIT_R + RA) + 1.5) / (2 * math.sin(math.pi / M)))
            self.world_R = self.ring_R + ORBIT_R + 4.0
            ang = np.arange(M) * 2 * np.pi / M + 0.4
            self.cx, self.cz = self.ring_R * np.cos(ang), self.ring_R * np.sin(ang)
            self.neighbors = [np.array([(c - 1) % M, (c + 1) % M]) for c in range(M)]
        self.version = np.zeros(M, dtype=np.int64)
        self.regen = np.zeros(M, dtype=np.int64)
        self.first_done = np.zeros(M, dtype=bool)
        self.attrs = np.zeros((M, N_OBJ, 3), dtype=np.int64)
        self.odd = np.zeros(M, dtype=np.int64)
        self.odd_attr = np.zeros(M, dtype=np.int64)
        self.ox = np.zeros((M, N_OBJ)); self.oz = np.zeros((M, N_OBJ))
        self.oang = np.zeros((M, N_OBJ)); self.orad = np.zeros((M, N_OBJ))
        self.X = torch.zeros(M, N_OBJ, 10)
        for c in range(M):
            self._regen(c)

        # agents: fixed-capacity slots
        f = lambda v=0.0: np.full(cap, v, dtype=float)
        self.alive = np.zeros(cap, dtype=bool)
        self.x, self.z, self.h = f(), f(), f()
        self.px, self.pz = f(), f()
        self.home = np.zeros(cap, dtype=np.int64)
        self.target = np.full(cap, -1, dtype=np.int64)
        self.key_home = np.full(cap, -1, dtype=np.int64)
        self.key_ver = np.full(cap, -1, dtype=np.int64)
        self.p = np.full((cap, N_OBJ), 1 / N_OBJ)
        self.tried = np.zeros((cap, N_OBJ), dtype=bool)    # objects already eaten (wrongly) in the current scene
        self.need_p = np.zeros(cap, dtype=bool)             # brain changed (lifetime learning): recompute p, keep `tried`
        self.cool, self.energy, self.age = f(), f(), f()
        self.intent, self.blocked = f(), f()
        self.flip = np.zeros(cap, dtype=bool)
        self.block_count = np.zeros(cap, dtype=np.int64)
        self.detour, self.detour_dir = f(), f()
        self.lineage = np.zeros(cap, dtype=np.int64)
        self.generation = np.zeros(cap, dtype=np.int64)
        self.eats = np.zeros(cap, dtype=np.int64)
        self.correct = np.zeros(cap, dtype=np.int64)
        self.genome = torch.zeros(cap, genome_dim) if genome_dim else None
        if hasattr(brain_fn, "world"):
            brain_fn.world = self                           # brains read genomes at birth (on_birth)

        for k in range(n_init):
            c = k % M
            toward = math.atan2(-self.cz[c], -self.cx[c]) + 0.5 * (k // M)
            g = init_genomes[k] if init_genomes is not None else None
            self._spawn(k, self.cx[c] + math.cos(toward) * (RING_R + 2.6), self.cz[c] + math.sin(toward) * (RING_R + 2.6),
                        c, g, lineage=k, generation=0, energy=self.eco.start_energy,
                        age=self.rng.uniform(0, self.eco.max_age / 2))     # staggered, so founders don't all die at once

        self.t = 0
        self.events = []          # (t, slot, correct, odd_attr, cluster, object)
        self.first_choices = []   # (t, correct)
        self.births, self.deaths = [], []   # (t, slot, ...)
        self.solved = 0

    # ---------- clusters ----------
    def _regen(self, c):
        a, o, k = sample(1, N_OBJ, self.tgen, "none")
        self.attrs[c] = a[0].numpy(); self.odd[c] = int(o[0]); self.odd_attr[c] = int(k[0])
        self.X[c] = one_hot(a)[0]
        self.version[c] += 1
        self.first_done[c] = False
        self.regen[c] = 0
        angs = np.arange(N_OBJ) * 2 * np.pi / N_OBJ + self.rng.uniform(0, 0.3, N_OBJ)
        self.oang[c] = angs
        self.ox[c] = self.cx[c] + RING_R * np.cos(angs)
        self.oz[c] = self.cz[c] + RING_R * np.sin(angs)
        self.orad[c] = OBJ_R[self.attrs[c, :, 0]] * SIZE_SCALE[self.attrs[c, :, 2]]

    # ---------- agents ----------
    def _spawn(self, s, x, z, home, genome, lineage, generation, energy, age=0.0):
        self.alive[s] = True
        self.x[s] = self.px[s] = x; self.z[s] = self.pz[s] = z
        self.h[s] = self.rng.uniform(0, 2 * np.pi)
        self.home[s] = home; self.target[s] = -1; self.key_home[s] = -1; self.key_ver[s] = -1
        self.tried[s] = False
        self.cool[s] = 0; self.energy[s] = energy; self.age[s] = age
        self.intent[s] = self.blocked[s] = 0; self.flip[s] = False; self.block_count[s] = 0
        self.detour[s] = self.detour_dir[s] = 0
        self.lineage[s] = lineage; self.generation[s] = generation
        self.eats[s] = self.correct[s] = 0
        if self.genome is not None and genome is not None:
            self.genome[s] = genome
        self.need_p[s] = False
        if hasattr(self.brain_fn, "on_birth"):
            self.brain_fn.on_birth(s)                       # develop this agent's weights from its genes

    def _think(self):
        live = np.flatnonzero(self.alive)
        h = self.home[live]
        stale = live[(self.key_home[live] != h) | (self.key_ver[live] != self.version[h])]
        if len(stale):
            self.key_home[stale] = self.home[stale]
            self.key_ver[stale] = self.version[self.home[stale]]
            self.target[stale] = -1
            self.tried[stale] = False                       # new scene (or new cluster): nothing tried yet
        recompute = np.union1d(stale, live[self.need_p[live]]).astype(np.int64)
        if len(recompute):
            self.p[recompute] = self.brain_fn(recompute, self.X[self.home[recompute]])
            self.need_p[recompute] = False
        need = live[(self.target[live] < 0) & (self.regen[self.home[live]] == 0)]
        if len(need):
            if self.choice == "greedy":
                self.target[need] = np.where(self.tried[need], -np.inf, self.p[need]).argmax(1)
            elif self.choice == "sample_ior":
                # sample from p among objects not yet tried (exploration, needed for lifetime learning)
                pm = np.where(self.tried[need], 0.0, self.p[need]) + 1e-12 * ~self.tried[need]
                cum = np.cumsum(pm, axis=1)
                u = self.rng.random(len(need))[:, None] * cum[:, -1:]
                self.target[need] = np.minimum((u > cum).sum(1), N_OBJ - 1)
            else:
                cum = np.cumsum(self.p[need], axis=1)
                u = self.rng.random(len(need))[:, None] * cum[:, -1:]
                self.target[need] = np.minimum((u > cum).sum(1), N_OBJ - 1)

    def _move(self):
        L = np.flatnonzero(self.alive)
        h = self.home[L]
        pending = self.regen[h] > 0
        self.target[L[pending]] = -1
        tgt = self.target[L]
        has_t = tgt >= 0
        dx, dz = self.x[L] - self.cx[h], self.z[L] - self.cz[h]
        rho, phi = np.hypot(dx, dz), np.arctan2(dz, dx)
        ti = np.where(has_t, tgt, 0)
        d = wrap_pi(self.oang[h, ti] - phi)
        orbit_phi = phi.copy()
        arriving = (~has_t) | (rho > ORBIT_R + 1.5)
        circling = has_t & ~arriving & (np.abs(d) > 0.25) & (rho > RING_R)
        orbit_phi[circling] = phi[circling] + np.sign(d[circling]) * np.minimum(np.abs(d[circling]), 0.6)
        final = has_t & ~arriving & ~circling
        tx = np.where(final, self.ox[h, ti], self.cx[h] + np.cos(orbit_phi) * ORBIT_R)
        tz = np.where(final, self.oz[h, ti], self.cz[h] + np.sin(orbit_phi) * ORBIT_R)
        det = self.detour[L] > 0
        tx = np.where(det, self.x[L] + np.cos(self.detour_dir[L]) * 3, tx)
        tz = np.where(det, self.z[L] + np.sin(self.detour_dir[L]) * 3, tz)
        self.detour[L[det]] -= DT

        desired = np.arctan2(tz - self.z[L], tx - self.x[L])
        self.h[L] = wrap_pi(self.h[L] + np.clip(wrap_pi(desired - self.h[L]), -2.6 * DT, 2.6 * DT))
        dist = np.hypot(tx - self.x[L], tz - self.z[L])
        speed = np.where(final, np.minimum(3.0, dist * 1.5),
                         np.where(has_t, np.minimum(3.0, dist * 1.5 + 0.5), np.minimum(3.0, dist * 1.2)))
        self.intent[L] = speed * DT
        fx, fz = np.cos(self.h[L]), np.sin(self.h[L])
        vx, vz = fx * speed, fz * speed

        # avoidance: other agents and objects of other (non-regenerating) clusters
        def avoid(ox, oz, rr, mask):
            ddx, ddz = ox - self.x[L][:, None], oz - self.z[L][:, None]
            dd = np.hypot(ddx, ddz)
            reach = rr + 1.6
            ahead = (ddx * fx[:, None] + ddz * fz[:, None]) / np.maximum(dd, 1e-4)
            ok = mask & (dd < reach) & (dd > 1e-4) & (ahead >= 0.2)
            side = np.where((fx[:, None] * ddz - fz[:, None] * ddx) >= 0, -1.0, 1.0)
            side = np.where(self.flip[L][:, None], -side, side)
            k = np.where(ok, (reach - dd) / reach * ahead * speed[:, None] * 1.6, 0.0)
            return (-fz[:, None] * side * k).sum(1), (fx[:, None] * side * k).sum(1)

        ax_, az_ = avoid(self.x[L][None, :], self.z[L][None, :], 2 * RA, ~np.eye(len(L), dtype=bool))
        vx += ax_; vz += az_
        oxs, ozs, ors = self.ox.reshape(-1), self.oz.reshape(-1), self.orad.reshape(-1)
        ocl = np.repeat(np.arange(self.M), N_OBJ)
        omask = (ocl[None, :] != h[:, None]) & (self.regen[ocl] == 0)[None, :]
        ax_, az_ = avoid(oxs[None, :], ozs[None, :], RA + ors[None, :], omask)
        vx += ax_; vz += az_
        self.x[L] += vx * DT; self.z[L] += vz * DT
        self.cool[L] = np.maximum(0.0, self.cool[L] - DT)
        return L, speed

    def _contacts(self):
        L = np.flatnonzero(self.alive)
        oxs, ozs, ors = self.ox.reshape(-1), self.oz.reshape(-1), self.orad.reshape(-1)
        ocl = np.repeat(np.arange(self.M), N_OBJ)
        solid = self.regen[ocl] == 0
        for _ in range(4):
            x, z = self.x[L], self.z[L]
            ddx, ddz = x[:, None] - x[None, :], z[:, None] - z[None, :]
            dd = np.hypot(ddx, ddz)
            np.fill_diagonal(dd, np.inf)
            ov = np.clip(2 * RA - dd, 0, None) / 2
            safe = np.maximum(dd, 1e-6)
            self.x[L] += (ov * ddx / safe).sum(1)
            self.z[L] += (ov * ddz / safe).sum(1)
            x, z = self.x[L], self.z[L]
            ddx, ddz = x[:, None] - oxs[None, :], z[:, None] - ozs[None, :]
            dd = np.hypot(ddx, ddz)
            ov = np.where(solid[None, :], np.clip(RA + ors[None, :] - dd, 0, None), 0.0)
            safe = np.maximum(dd, 1e-6)
            self.x[L] += (ov * ddx / safe).sum(1)
            self.z[L] += (ov * ddz / safe).sum(1)
            r = np.hypot(self.x[L], self.z[L]); rmax = self.world_R - RA - 0.3
            scale = np.where(r > rmax, rmax / np.maximum(r, 1e-6), 1.0)
            self.x[L] *= scale; self.z[L] *= scale

    def _blocked(self, L):
        moved = np.hypot(self.x[L] - self.px[L], self.z[L] - self.pz[L])
        stuck = (self.intent[L] > 0.02) & (moved < 0.25 * self.intent[L])
        self.blocked[L] = np.where(stuck, self.blocked[L] + DT, np.maximum(0.0, self.blocked[L] - DT))
        for s in L[self.blocked[L] > 1]:
            self.flip[s] = not self.flip[s]; self.blocked[s] = 0
            self.block_count[s] += 1
            if self.block_count[s] >= 3:
                self.target[s] = -1; self.block_count[s] = 0; self.detour[s] = 0.9
                self.detour_dir[s] = self.h[s] + (1 if self.rng.random() < 0.5 else -1) * (np.pi * 0.6 + self.rng.random() * 0.6)

    def _neighbor_cluster(self, c):
        nb = self.neighbors[c]
        return int(self.rng.choice(nb)) if len(nb) else int(c)

    def _eat(self, L):
        e = self.eco
        for s in L:
            c, t = self.home[s], self.target[s]
            if t < 0 or self.regen[c] or self.cool[s] > 0:
                continue
            if math.hypot(self.ox[c, t] - self.x[s], self.oz[c, t] - self.z[s]) > RA + self.orad[c, t] + 0.25:
                continue
            correct = int(t == self.odd[c])
            self.eats[s] += 1; self.correct[s] += correct
            self.events.append((self.t, s, correct, int(self.odd_attr[c]), int(c), int(t)))
            if not self.first_done[c]:
                self.first_choices.append((self.t, correct)); self.first_done[c] = True
            if self.alife and e.neutral and e.neutral_mode == "shuffle":
                # same food economy as selection (the eat's true payoff, the cluster regenerates as usual), but the payoff
                # goes to a random agent of the WHOLE population. (Giving it to a random agent at the same cluster leaks
                # selection: children live where their parents do, so a good lineage feeds its own kin; measured
                # rho(innate phi, offspring) = +0.058 vs +0.093 under selection, neutrality_check.py.)
                de = e.gain_correct if correct else -e.loss_wrong
                r = int(self.rng.choice(np.flatnonzero(self.alive)))
                self.energy[r] = min(e.max_energy, self.energy[r] + de)
                if hasattr(self.brain_fn, "learn"):
                    self.brain_fn.learn(int(s), self.X[c], self.tried[s].copy(), int(t), float(de) if r == s else 0.0)
                    self.need_p[s] = True
            elif self.alife:
                if e.neutral:
                    # payoff of a RANDOM pick among the objects not yet tried (same luck structure as a
                    # random eater), decoupled from what the agent actually chose: no selection on choice
                    untried = np.flatnonzero(~self.tried[s])
                    lucky = self.rng.choice(untried) == self.odd[c]
                    de = e.gain_correct if lucky else -e.loss_wrong
                else:
                    de = e.gain_correct if correct else -e.loss_wrong
                self.energy[s] = min(e.max_energy, self.energy[s] + de)
                if hasattr(self.brain_fn, "learn"):
                    # lifetime learning from the agent's own energy change (the only signal it has)
                    self.brain_fn.learn(int(s), self.X[c], self.tried[s].copy(), int(t), float(de))
                    self.need_p[s] = True
            if not correct:
                self.tried[s, t] = True                     # inhibition of return
            self.cool[s] = COOLDOWN
            if correct:
                self.regen[c] = REGEN_STEPS; self.solved += 1
            self.target[s] = -1
            if self.rng.random() < SWITCH_P:
                self.home[s] = self._neighbor_cluster(c)

    def _life(self, L, speed):
        e = self.eco
        crowd = len(L) / e.K if e.density else 1.0   # density-dependent upkeep (old) or constant (food sets capacity)
        self.energy[L] -= e.basal * crowd * DT + e.move * speed * DT
        self.age[L] += DT
        dead = L[(self.energy[L] <= 0) | (self.age[L] > e.max_age)]
        for s in dead:
            self.alive[s] = False
            self.deaths.append((self.t, int(s), float(self.age[s]), int(self.eats[s]), int(self.correct[s])))
        if self.reproduction == "sexual":
            self._mate()
            return
        ready = self.rng.permutation(np.flatnonzero(self.alive & (self.energy >= e.repro_threshold)))   # no slot-order bias
        for s in ready:
            free = np.flatnonzero(~self.alive)
            if not len(free):
                break
            c = int(free[0])
            child_g = None
            if self.genome is not None:
                child_g = self.genome[s] + torch.randn(self.genome.shape[1], generator=self.tgen) * self.sigma
            a = self.rng.uniform(0, 2 * np.pi)
            self._spawn(c, self.x[s] + math.cos(a) * 2 * RA, self.z[s] + math.sin(a) * 2 * RA, self.home[s], child_g,
                        lineage=self.lineage[s], generation=self.generation[s] + 1, energy=e.child_energy)
            self.energy[s] -= e.child_energy + e.repro_overhead
            self.births.append((self.t, c, int(s)))

    def _mate(self):
        e = self.eco
        ready = self.rng.permutation(np.flatnonzero(self.alive & (self.energy >= self.mate_threshold)))
        used = set()
        for s in ready:
            if s in used:
                continue
            others = np.array([o for o in ready if o != s and o not in used], dtype=np.int64)
            if not len(others):
                break
            d = np.hypot(self.x[others] - self.x[s], self.z[others] - self.z[s])
            if d.min() > self.mate_radius:
                continue
            p2 = int(others[d.argmin()])
            free = np.flatnonzero(~self.alive)
            if not len(free):
                break
            c = int(free[0])
            child_g = None
            if self.genome is not None:
                child_g = (self.genome[s] + self.genome[p2]) / 2 + torch.randn(self.genome.shape[1], generator=self.tgen) * self.sigma
            mx, mz = (self.x[s] + self.x[p2]) / 2, (self.z[s] + self.z[p2]) / 2
            self._spawn(c, mx, mz, self.home[s], child_g, lineage=self.lineage[s],
                        generation=max(self.generation[s], self.generation[p2]) + 1, energy=e.child_energy)
            half = (e.child_energy + e.repro_overhead) / 2
            self.energy[s] -= half; self.energy[p2] -= half
            used.update((s, p2))
            self.births.append((self.t, c, int(s), p2))

    def step(self):
        for c in np.flatnonzero(self.regen > 0):
            self.regen[c] -= 1
            if self.regen[c] == 0:
                self._regen(c)
        if not self.alive.any():
            self.t += 1
            return
        L = np.flatnonzero(self.alive)
        self.px[L], self.pz[L] = self.x[L], self.z[L]
        self._think()
        L, speed = self._move()
        self._contacts()
        self._blocked(L)
        self._eat(L)
        if self.alife:
            self._life(L, speed)
        self.t += 1

    @property
    def population(self):
        return int(self.alive.sum())
