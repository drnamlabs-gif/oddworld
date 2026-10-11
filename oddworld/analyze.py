# Dr. Nam H. Le. drnamlabs.com, bunnyquant.com
"""Did relational reasoning emerge? Per seed: phi on training-vocabulary puzzles and on
puzzles with never-seen symbol values (population probe of 40 living genomes), first time the population mean
crosses phi_heur + margin and theta = 0.5, and selection vs neutral on the final window (per-seed unit:
Welch t, Mann-Whitney U, Cohen's d, bootstrap CI)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

RUNS = Path(__file__).resolve().parents[1] / "runs"
SEEDS = [0, 1, 2]
cols = {"selection": "tab:red", "neutral": "tab:gray"}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
final = {c: {"train": [], "novel": []} for c in cols}
for cond, col in cols.items():
    for s in SEEDS:
        rows = [json.loads(l) for l in open(RUNS / f"{cond}_seed{s}" / "log.jsonl")]
        P = [r for r in rows if "innate_phi_mean" in r]
        t = np.array([r["sim_s"] for r in P]); tr = np.array([r["innate_phi_mean"] for r in P]); nv = np.array([r["novel_phi_mean"] for r in P])
        ls = ["-", "--", ":"][s]
        axs[0].plot(t, tr, color=col, ls=ls, lw=1.2, label=f"{cond} s{s}")
        axs[1].plot(t, nv, color=col, ls=ls, lw=1.2)
        axs[2].plot([r["sim_s"] for r in rows], [r["pop"] for r in rows], color=col, ls=ls, lw=0.8)
        k = max(1, len(P) // 10)
        final[cond]["train"].append(tr[-k:].mean()); final[cond]["novel"].append(nv[-k:].mean())
        def first(th):
            sm = np.convolve(tr, np.ones(5) / 5, mode="valid")
            i = np.flatnonzero(sm >= th)
            return f"{t[i[0] + 4]:.0f} s (gen ~{rows[min(len(rows) - 1, int(i[0] * 5 + 20))]['gen_mean']:.0f})" if len(i) else "never"
        print(f"{cond:9s} s{s}: final train {tr[-k:].mean():+.3f}  novel {nv[-k:].mean():+.3f}  max train {tr.max():+.2f}  "
              f"gen {rows[-1]['gen_max']}  pop(last) {rows[-1]['pop']}  cross 0.35: {first(0.35)}  cross 0.5: {first(0.5)}")
for ax, ttl in zip(axs, ["innate φ, seen symbols (population mean)", "innate φ, symbols never seen in evolution", "population"]):
    ax.set_title(ttl); ax.set_xlabel("simulated time (s)")
for ax in axs[:2]:
    ax.axhline(0.25, color="k", ls=":", lw=0.7); ax.axhline(0.5, color="k", ls="--", lw=0.5); ax.set_ylim(0.1, 1)
axs[0].legend(fontsize=7)
fig.suptitle("Relational reasoning emerges under implicit selection (3 seeds per condition; dotted line = best non-relational policy, φ = 0.25)", fontsize=11)
fig.tight_layout(); fig.savefig(RUNS / "emergence.png", dpi=130)
for m in ("train", "novel"):
    a, b = np.array(final["selection"][m]), np.array(final["neutral"][m])
    d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    rng = np.random.default_rng(0)
    boot = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(10000)]
    print(f"{m}: selection {a.round(3)} vs neutral {b.round(3)}  Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3g}  "
          f"MW p={stats.mannwhitneyu(a, b, alternative='greater').pvalue:.3g}  d={d:.2f}  diff 95% CI [{np.quantile(boot, .025):+.3f}, {np.quantile(boot, .975):+.3f}]")
print("saved", RUNS / "emergence.png")
