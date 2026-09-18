# Fly give-up-circuit wiring vs. equilibrium learning in the war of attrition

Research codebase for the question: *when two learning agents play a war of attrition,
does constraining each agent's policy network to the real Drosophila give-up circuit's
wiring (PPL1 → MBON18/19 → Tk-GAL4^FruM, MaleCNS v1.0) change how well or how fast they
converge to the known equilibrium, compared to conventional networks of matched size?*

* `docs/theory.md` — derivations (Game A exponential NE, Game B `b(v) = v²/2`, grid
  equilibrium, asymmetric-update rest point) and metric definitions.

Setup (CPU only):

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or .venv/bin/pip
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest
```
