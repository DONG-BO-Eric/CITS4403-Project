# Utility / helper modules

| File | Provides | Used by |
|---|---|---|
| `Cell2D.py` | `Cell2D` base class (`loop`, `draw`, `animate`) and `draw_array` | `src/extended_schelling.py` (base class of `Schelling`) |
| `utils.py` | `decorate`, `legend`, `three_frame`, `number_frame`, `set_palette`, `savefig`, `underride` | `notebooks/extended_schelling_analysis.ipynb` (plotting) and `Cell2D.py` |

`Cell2D.py` and most of `utils.py` come from the course materials, based on Allen B. Downey's
*Think Complexity* / *Complexity and Computation* (MIT License). `number_frame` is our own addition:
it draws a configurable number of frames side by side, running a given number of steps before each.
