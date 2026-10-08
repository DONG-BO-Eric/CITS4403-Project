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

Run the **Model testing** section of `notebooks/extended_schelling_analysis.ipynb` after the setup cells. It includes the main initialization, satisfaction, relocation and marriage checks, followed by grid reproducibility checks using the same parameters and seed.

## Notebook guide

| Section | Content |
|---|---|
| 1–2 | Setup, and the full model specification with equations |
| 3 | Model testing and initial openness distribution |
| 4 | Grid evolution |
| 5 | Calibration of the satisfaction threshold `p` (peak at `p` ≈ 0.76) |
| 6 | Effect of `m` on pure-group cluster size at `p` = 0.76 |
| 7 | How `p` changes the direction of the effect of `m` |
| 8 | AB boundary fraction as a function of `m` |
| 9 | AB boundary fraction as a function of uniform openness |
| 10 | Openness × `m` heatmap of the AB boundary fraction |
| 11 | Summary and limitations |

## Team and contributions

| Member | Student number | Main contributions |
|---|---|---|
| ZHAO YANG ZHAO | 25120743 | Main model implementation (`src/extended_schelling.py`: `Person` and `Schelling` classes, relocation, marriage and openness mechanics); analysis of how the satisfaction threshold `p` affects segregation, including calibration of `p` (Notebook Sections 4–5) |
| BO DONG | 24679419 | Analysis of clustering and AB-household bridging under different values of `m` and openness (Notebook Sections 6–10); restructuring of the repository; code-quality review; project report |

## Acknowledgements

The `Cell2D` base class and plotting helpers are adapted from the course materials, which are based
on Allen B. Downey, *Think Complexity*, 2nd edition (O'Reilly, 2018), MIT License.
`number_frame` in `utils/utils.py` is our own addition.
