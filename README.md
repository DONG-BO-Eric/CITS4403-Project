# Extended Schelling Model: Marriage, Openness and Cross-Group Households

CITS4403 Research Project

## Overview

Schelling's segregation model shows how mild individual preferences for similar neighbours can
produce strongly segregated neighbourhoods. In the classic model, agents can only **move**.
Real neighbourhoods also change through **household formation**: people pair up, and some of
those pairs cross group lines.

This project extends the Schelling model with marriage and individual openness. It asks whether
mixed (AB) households act as **connectors** between two otherwise separated communities, and how
the amount of marriage activity and individual openness strengthen or weaken that effect.

## Our contribution (beyond the textbook Schelling model)

The Schelling model of segregation is covered in the lectures and in *Think Complexity*. Our
starting point is that course version (`Cell2D`-based grid, satisfaction threshold `p`,
random relocation of unhappy agents). Everything below is our own extension:

- **Households.** Six cell states instead of three: empty, single A, single B, and AA, BB and
  AB couples, implemented with a `Person` class that tracks each agent's ID, location, group,
  openness, satisfaction and neighbour history.
- **Marriage process** (marriage parameter `m`). Two happy single agents who have been neighbours
  for at least three consecutive rounds become eligible to marry. Each is willing in a given step
  with probability `m`, and the total number of marriages is capped at `m` × the initial
  population. When a couple forms, one cell becomes the household and the other cell becomes a
  new single agent.
- **Openness.** A per-agent trait that (i) weights cross-group pairs (weight = product of the
  partners' openness) and (ii) enters the satisfaction score, so other-group neighbours count as
  partly similar. `fixed_openness` lets an experiment give every agent the same value.
- **New outcome measures.** Average size of 8-connected pure-group clusters, and the
  **AB boundary fraction**: the share of AB households that border both a pure-A and a pure-B cell.
- **Experiments.** Calibration of `p`; sweeps of `m`, of openness, and of the full openness × `m`
  grid; and comparisons across `p`.

## Repository structure

```
project-root/
├── src/
│   └── extended_schelling.py          # Model: Person and Schelling classes
├── utils/
│   ├── Cell2D.py                       # Cell2D base class and draw_array (course materials)
│   ├── utils.py                        # Plotting helpers (decorate, three_frame, number_frame, ...)
│   └── README.md
├── data/
│   └── README.md                       # No external data; all results come from simulation
├── notebooks/
│   └── extended_schelling_analysis.ipynb   # Model specification, experiments and results
├── tests/
│   ├── conftest.py
│   └── test_extended_schelling.py      # Tests of the model rules and edge cases (pytest)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.x (developed with Python 3.14).

```bash
git clone <repository-url>
cd project-root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Run the full analysis**

```bash
cd notebooks
jupyter notebook extended_schelling_analysis.ipynb
```

Run the cells from top to bottom. The first cell adds `src/` and `utils/` to the Python path.
Several experiments repeat many 100 × 100 simulations, so the later sections can take several
minutes each.

**Use the model directly** (from the project root)

```python
import sys
sys.path[:0] = ["src", "utils"]

import numpy as np
from extended_schelling import Schelling

np.random.seed(0)
grid = Schelling(n=100, p=0.76, m=0.54)        # optional: fixed_openness=0.8
for _ in range(150):
    segregation = grid.step()                  # returns mean fraction of similar neighbours

print(segregation, grid.marriages_done)
grid.draw()                                    # requires matplotlib
```

Model parameters:

| Parameter | Meaning | Default |
|---|---|---|
| `n` | Grid size (`n` × `n`, wrap-around boundary) | required |
| `p` | Satisfaction threshold an occupant needs in order to stay | required |
| `m` | Marriage parameter: per-step willingness probability and cap on total marriages | 0.1 |
| `fixed_openness` | If set, every agent uses this openness instead of a random value in {0.1, …, 0.9} | `None` |

## Testing

The model rules are checked with `pytest`. From the project root:

```bash
pytest tests/
```

The tests cover initialisation, the satisfaction formula for every household type, the
wrap-around boundary, agents with no neighbours, the happiness threshold, the segregation
measure on layouts with known answers, relocation (including a full grid with no empty
cells), neighbour tracking and marriage eligibility, openness-weighted pair selection, the
marriage cap, household formation, run-level invariants and reproducibility with a fixed seed.

One test is marked as an expected failure: if every agent has openness 0 and only
cross-group pairs are available, all selection weights are 0 and pair selection raises an
error. Openness must therefore be greater than 0; all experiments use openness of at least 0.1.

## Notebook guide

| Section | Content |
|---|---|
| 1–2 | Setup, and the full model specification with equations |
| 3 | Initial openness distribution and grid evolution |
| 4 | Calibration of the satisfaction threshold `p` (peak at `p` ≈ 0.76) |
| 5 | Effect of `m` on pure-group cluster size at `p` = 0.76 |
| 6 | How `p` changes the direction of the effect of `m` |
| 7 | AB boundary fraction as a function of `m` |
| 8 | AB boundary fraction as a function of uniform openness |
| 9 | Openness × `m` heatmap of the AB boundary fraction |
| 10 | Summary and limitations |

## Team and contributions

| Member | Student number | Main contributions |
|---|---|---|
| ZHAO YANG ZHAO | 25120743 | Main model implementation (`src/extended_schelling.py`: `Person` and `Schelling` classes, relocation, marriage and openness mechanics); analysis of how the satisfaction threshold `p` affects segregation, including calibration of `p` (Notebook Sections 3–4) |
| BO DONG | 24679419 | Analysis of clustering and AB-household bridging under different values of `m` and openness (Notebook Sections 5–9); restructuring of the repository; code-quality review; project report |

## Acknowledgements

The `Cell2D` base class and plotting helpers are adapted from the course materials, which are based
on Allen B. Downey, *Think Complexity*, 2nd edition (O'Reilly, 2018), MIT License.
`number_frame` in `utils/utils.py` is our own addition.
