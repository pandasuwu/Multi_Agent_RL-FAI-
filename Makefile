PY ?= .venv/Scripts/python

.PHONY: reproduce test connectome tune experiments analysis figures review

reproduce:
	$(PY) scripts/reproduce.py

test:
	$(PY) -m pytest -q

connectome:
	$(PY) scripts/build_connectome.py --download

tune:
	$(PY) scripts/tune_lr.py configs/game_a_exact.yaml configs/game_b_exact.yaml configs/game_a_sampled.yaml configs/game_b_sampled.yaml

experiments:
	$(PY) scripts/run_experiment.py configs/game_a_exact.yaml configs/game_b_exact.yaml configs/game_a_sampled.yaml configs/game_b_sampled.yaml

analysis:
	$(PY) scripts/analyze.py

figures:
	$(PY) scripts/make_figures.py

review:
	$(PY) scripts/run_experiment.py configs/popnash_game_a_sampled.yaml configs/lrsweep_game_a_exact.yaml configs/lrsweep_game_b_exact.yaml
	$(PY) scripts/validate_asymmetric.py
	$(PY) scripts/path_diagnostics.py
	$(PY) scripts/analyze.py
