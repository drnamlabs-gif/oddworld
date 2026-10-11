# Oddworld: relational reasoning emerges under implicit selection in evolving attention agents

Dr. Nam H. Le. drnamlabs.com, bunnyquant.com

Code and data for the technical note (DOI: 10.5281/zenodo.23291253) by Nam H. Le (Department of Computer Science and Vermont Complex Systems Institute, University of Vermont; nam.le@uvm.edu), October 2026.

Requirements: Python 3.11+, PyTorch 2.x, NumPy, SciPy, Matplotlib.

## Layout
- `oddworld/`: all code
  - `puzzle.py`: the odd-one-out puzzle generator
  - `world.py`: the world (clusters, movement, eating, energy, death, sexual reproduction)
  - `attention_brain.py`: the generic attention brain; `agent_brains.py`: genomes = brain weights, plus the offline probe
  - `run.py`: one run (selection or lottery control) with the note's configuration as defaults
  - `analyze.py`: Figure 3 and the selection-vs-control statistics
  - `capacity.py`, `explicit_evolution.py`: the capacity check (Table 1)
  - `mechanism.py`, `mechanism_rarity.py`: the ablations and the rarity analysis (Section 4)
  - `neutrality_check.py` with `structured_brain.py`, `structured_agent_brains.py`: the control experiment of Section 5
  - `record_replay.py`: exact deterministic re-runs for the video and the 3D replay (first `python record_replay.py run --condition selection --seed 0`, then `clip --at <step>`)
- `runs/{selection,neutral}_seed{0,1,2}/`: logs (`log.jsonl`, one row per 2,000 steps) and final genomes of the six runs in the note
- `results/`: `capacity.json`, `mechanism.json`, `explicit_evolution/` (Table 1 and Section 4)

## Reproduce
```
cd oddworld
python run.py --condition selection --seed 0     # ~70 min on one CPU core; deterministic given the seed
python run.py --condition neutral --seed 0       # lottery control
python analyze.py                                # Figure 3 + statistics from ../runs
python mechanism.py && python mechanism_rarity.py
python capacity.py
python explicit_evolution.py --algo ga_sex --seed 0
python neutrality_check.py --condition neutral --mode shuffle
```

Code: MIT licence (see LICENSE). Text, figures, video and data: CC BY 4.0.
